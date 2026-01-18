import pandas as pd
import numpy as np
from datetime import timedelta
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from sqlalchemy import text

def process_disposal_events(disposal_info):
    """
    前處理 Finlab 處置資料：
    1. 欄位標準化 (Stock_id, event_start_date, event_end_date)
    2. 排序並計算 Gap
    3. 標記 First/Second Disposal
    """
    # 強制轉為 Pandas DataFrame
    events = pd.DataFrame(disposal_info).copy()
    
    # 檢查是否需要 Reset Index (因應 Finlab 可能將 stock_id 設為 Index)
    if 'stock_id' not in events.columns and 'Stock_id' not in events.columns:
        events = events.reset_index()

    # 1. 欄位映射與標準化
    # 統一將 stock_id 轉為 Stock_id
    if 'stock_id' in events.columns:
        events = events.rename(columns={'stock_id': 'Stock_id'})
        
    if '處置開始時間' in events.columns:
        events = events.rename(columns={
            '處置開始時間': 'event_start_date',
            '處置結束時間': 'event_end_date',
            '分時交易': 'interval',
            '處置條件': 'condition'
        })
    elif 'date' in events.columns:
         events = events.rename(columns={'date': 'event_start_date'})
         events['event_end_date'] = events['event_start_date']

    # 確保日期格式
    events['event_start_date'] = pd.to_datetime(events['event_start_date'])
    events['event_end_date'] = pd.to_datetime(events['event_end_date'])
    
    # Debug info
    print(f"Columns before processing: {events.columns.tolist()}")

    # 2. 排序
    # Ensure Stock_id exists
    if 'Stock_id' not in events.columns:
        # Try finding case-insensitive match
        for col in events.columns:
            if col.lower() == 'stock_id':
                events = events.rename(columns={col: 'Stock_id'})
                break
    
    if 'Stock_id' not in events.columns:
         # Last resort: use index name if matches
         if events.index.name and events.index.name.lower() == 'stock_id':
             events = events.reset_index()
             events = events.rename(columns={events.columns[0]: 'Stock_id'}) # Assuming reset puts it at 0, or use name
         else:
             raise KeyError(f"Column 'Stock_id' not found. Available columns: {events.columns.tolist()}")

    events = events.sort_values(['Stock_id', 'event_start_date'])
    
    # 3. 動態層級判斷 (Dynamic Level Classification)
    # 定義: 若 Start <= Prev_End + 3.5 days，則屬於同一個 Chain
    # 為了解決 "Start <= Max(Prev_End_in_Chain)" 的問題，我們需要迭代檢查
    # 但為了向量化效能，我們可以先計算 gap，再用 cumsum 分組
    
    # 這裡稍微複雜，因為我們需要拿 "上一筆處置的結束日" 來比，
    # 但如果是連續重疊 (Chain)，我們應該拿 "目前 Chain 的最大結束日" 來比。
    # 由於 Pandas 向量化較難處理 "Chain Max End"，我們使用 python loop 確保邏輯正確 (資料量 3000 筆，Loop 很快)

    events['disposal_level'] = 1
    
    # 為了方便 Iteration，先重設 index
    events = events.reset_index(drop=True)
    
    current_stock = None
    chain_end_date = None
    current_level = 1
    
    levels = []
    
    for idx, row in events.iterrows():
        stock = row['Stock_id']
        start = row['event_start_date']
        end = row['event_end_date']
        
        if stock != current_stock:
            # New Stock -> Reset
            current_stock = stock
            chain_end_date = end
            current_level = 1
        else:
            # Same Stock
            # Check overlap/continuity
            # User requested strict overlap or close continuity (Chain)
            # Add small buffer (e.g. 3.5 days) to account for weekends/minor gaps
            limit_date = chain_end_date + pd.Timedelta(days=3.5)
            
            if start <= limit_date:
                # Continuous -> Increase Level
                current_level += 1
                # Extend chain end if this event ends later
                if end > chain_end_date:
                    chain_end_date = end
            else:
                # Broken Chain -> Reset Level
                current_level = 1
                chain_end_date = end
        
        levels.append(current_level)
        
    events['disposal_level'] = levels
    
    return events

def fetch_and_filter_price_range(client, stock_id, start_dates, end_dates, offset_days=3):
    """
    針對單檔股票，抓取並篩選出落在 [Start-Offset, End+Offset] 區間內的股價。
    支援多個處置區間（取聯集）。
    """
    if start_dates.empty or end_dates.empty:
        return None

    # 1. 決定抓取的最大範圍 (全域 Min ~ 全域 Max)
    # 這能減少對 API 的請求次數 (一次抓一大段)
    # [Fix] Expand Date Range: Convert "Trading Days expectation" to "Calendar Days buffer"
    # User might set offset_days=5 expecting 5 trading days.
    # To be safe against holidays/weekends, we multiply by 2 and add a buffer.
    safe_buffer = offset_days * 2 + 15
    
    # [Fix] Enforce datetime conversion to avoid 'int' vs 'Timestamp' confusion
    # Sometimes inputs might be mixed types if upstream processing wasn't strict
    start_dates = pd.to_datetime(start_dates, errors='coerce')
    end_dates = pd.to_datetime(end_dates, errors='coerce')
    
    # Drop invalid dates
    # Check if empty after dropna
    valid_mask = start_dates.notna() & end_dates.notna()
    if not valid_mask.all():
         start_dates = start_dates[valid_mask]
         end_dates = end_dates[valid_mask]
         
    if start_dates.empty or end_dates.empty:
        return None

    global_min = start_dates.min() - timedelta(days=safe_buffer)
    global_max = end_dates.max() + timedelta(days=safe_buffer)
    
    start_str = global_min.strftime('%Y-%m-%d')
    end_str = global_max.strftime('%Y-%m-%d')
    
    try:
        # Wrap API call to catch FinMind's KeyError: 'data'
        try:
            client.initialize_frame(stock_id=stock_id, start_time=start_str, end_time=end_str)
            price_df = client.get_stock()
        except KeyError:
            # FinMind returns KeyError 'data' if no data found or API error (Token/Limit)
            print(f"[Warning] FinMind API returned no data for {stock_id}. This usually indicates an INVALID TOKEN or RATE LIMIT reached.")
            return None
        except Exception:
            # Other API errors
            return None
        
        if price_df is None or price_df.empty:
            return None
            
        if 'Date' not in price_df.columns:
            price_df = price_df.reset_index() # Ensure Date is a column
        
        if 'Date' not in price_df.columns:
            # Should not happen if API is correct, but safe guard
            return None

        # [Fix] Strict conversion to datetime64[ns] to avoid int vs Timestamp comparison
        price_df['Date'] = pd.to_datetime(price_df['Date'], errors='coerce')
        price_df = price_df.dropna(subset=['Date'])
        
        # [Fix] Sort by Date to ensure index reflects time order
        price_df = price_df.sort_values('Date').reset_index(drop=True)
        
        # [Critical Fix] Assign trading_idx BEFORE filtering
        # This ensures that even if we filter out rows later, the remaining rows 
        # keep their original index relative to the continuous trading history.
        # [Fix] Use existing trading_idx if available to preserve continuity from fetch
        # Otherwise generate new one (backward compatibility)
        if 'trading_idx' not in price_df.columns:
            price_df['trading_idx'] = price_df.index
        
        if price_df.empty:
            return None

        # [Fix] Ensure numpy array is strictly datetime64[ns]
        dates_in_df = price_df['Date'].values.astype('datetime64[ns]')
        trading_indices = price_df['trading_idx'].values
        
        # 2. 建立精確篩選遮罩 (Mask)
        # 用戶要求：確定抓到的是「實際交易日」 (offset_days 代表交易日數)
        mask = pd.Series(False, index=price_df.index)
        
        # Ensure start/end are iterables
        if not isinstance(start_dates, (list, pd.Series, np.ndarray)):
            start_dates = [start_dates]
        if not isinstance(end_dates, (list, pd.Series, np.ndarray)):
            end_dates = [end_dates]

        for s_date, e_date in zip(start_dates, end_dates):
            # [Fix] Validate and convert s_date/e_date to Timestamp
            if pd.isna(s_date) or pd.isna(e_date):
                continue
                
            try:
                s_ts = pd.Timestamp(s_date)
                e_ts = pd.Timestamp(e_date)
                if pd.isna(s_ts) or pd.isna(e_ts):
                    continue
                # [Fix] Convert to numpy datetime64 to avoid TypeError: '<' not supported between instances of 'int' and 'Timestamp'
                s_val = s_ts.to_datetime64()
                e_val = e_ts.to_datetime64()
            except:
                continue

            # searchsorted works correctly with datetime64[ns] array and scalar
            idx_start = dates_in_df.searchsorted(s_val, side='left')
            idx_end = dates_in_df.searchsorted(e_val, side='right') - 1
            
            target_idx_start = max(0, idx_start - offset_days)
            target_idx_end = min(len(price_df) - 1, idx_end + offset_days)
            
            if target_idx_start <= target_idx_end:
                mask.iloc[target_idx_start : target_idx_end + 1] = True

        # [Fix] Restore this line which was accidentally deleted
        filtered_df = price_df[mask].copy()
        
        if filtered_df.empty:
            return None
            
        return filtered_df
        
    except Exception as e:
        import traceback
        print(f"Error fetching {stock_id}: {e}")
        print(traceback.format_exc()) 
        return None

def _fetch_worker_range(stock_id, start_dates, end_dates, offset_days, token=None):
    from module.get_info_FinMind import FinMindClient
    # Create a new local client for thread safety
    local_client = FinMindClient()
    if token:
        local_client.login_by_token(token)
    return fetch_and_filter_price_range(local_client, stock_id, start_dates, end_dates, offset_days)

def batch_fetch_prices(client, disposal_info, offset_days=3, max_workers=8):
    """
    使用多執行緒平行抓取所有股票的價格資料。
    支援 Finlab 資料格式 (自動偵測 '處置開始時間' 與 '處置結束時間')。
    """
    # Ensure it's a standard pandas DataFrame (dissociate from Finlab objects)
    disposal_events = pd.DataFrame(disposal_info).copy()
    
    # Normalize stock_id column name
    if 'stock_id' in disposal_events.columns and 'Stock_id' not in disposal_events.columns:
        disposal_events = disposal_events.rename(columns={'stock_id': 'Stock_id'})
    # Handle case where stock_id might be in index
    if 'Stock_id' not in disposal_events.columns and 'stock_id' not in disposal_events.columns:
        disposal_events = disposal_events.reset_index()
        if 'stock_id' in disposal_events.columns:
             disposal_events = disposal_events.rename(columns={'stock_id': 'Stock_id'})
             
    token = client.config.api_token if hasattr(client, 'config') else None

    # 欄位偵測
    # 欄位偵測 (優先使用標準化欄位)
    if 'event_start_date' in disposal_events.columns and 'event_end_date' in disposal_events.columns:
        unique_stocks = disposal_events['Stock_id'].unique()
        col_start = 'event_start_date'
        col_end = 'event_end_date'
        print(f"Using pre-processed columns '{col_start}' and '{col_end}'.")
    elif '處置開始時間' in disposal_events.columns and '處置結束時間' in disposal_events.columns:
        unique_stocks = disposal_events['Stock_id'].unique()
        col_start = '處置開始時間'
        col_end = '處置結束時間'
        print(f"Detected Finlab format. Using '{col_start}' and '{col_end}' for range filtering.")
    elif 'date' in disposal_events.columns:
        # Fallback for old format (assume single day event, start=end)
        unique_stocks = disposal_events['Stock_id'].unique()

        col_start = 'date'
        col_end = 'date'
        print(f"Using single date column '{col_start}' (Start=End).")
    else:
        print("Error: Required date columns not found.")
        return pd.DataFrame()
    
    print(f"Starting batch fetch for {len(unique_stocks)} stocks with {max_workers} workers...")
    
    all_prices = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_stock = {}
        for stock_id in unique_stocks:
            subset = disposal_events[disposal_events['Stock_id'] == stock_id]
            starts = pd.to_datetime(subset[col_start])
            ends = pd.to_datetime(subset[col_end])
            
            future = executor.submit(_fetch_worker_range, stock_id, starts, ends, offset_days, token)
            future_to_stock[future] = stock_id
            
        for future in tqdm(as_completed(future_to_stock), total=len(unique_stocks), desc="Fetching Prices"):
            stock_id = future_to_stock[future]
            try:
                result_df = future.result()
                if result_df is not None:
                    all_prices.append(result_df)
            except Exception as exc:
                print(f"{stock_id} generated an exception: {exc}")
    
    if all_prices:
        final_df = pd.concat(all_prices).drop_duplicates()
        print(f"Fetched total {len(final_df)} rows.")
        return final_df
    else:
        print("No data fetched.")
        return pd.DataFrame()

def run_event_study(price_df, disposal_info, offset_days=3):
    """
    執行事件研究分析：
    1. 判定處置區間 (Start to End)
    2. 合併並標記相對天數 (t-label)
    3. 保留 [Start-Offset, End+Offset] 的所有資料 (Based on Trading Index)
    """
    if price_df.empty:
        print("Price DataFrame is empty.")
        return pd.DataFrame(), pd.DataFrame()
            
    # 1. 準備股價表
    prices = price_df.sort_values(['Stock_id', 'Date']).copy()
    
    # [Added] Filter invalid prices (Open > 0) to avoid infinite returns
    prices = prices[prices['Open'] > 0].copy()
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    # 建立交易日索引
    # [Fix] Use existing trading_idx if available to keep original continuity (with gaps)
    if 'trading_idx' not in prices.columns:
        prices['trading_idx'] = prices.groupby('Stock_id').cumcount()
    
    # 計算 Gap Days
    prices['prev_trade_date'] = prices.groupby('Stock_id')['Date'].shift(1)
    prices['trade_date_diff'] = (prices['Date'] - prices['prev_trade_date']).dt.days
    prices['gap_days'] = prices['trade_date_diff'].fillna(1) - 1
    
    # 2. 準備事件表
    # Check if pre-processed
    if 'event_start_date' in disposal_info.columns and 'disposal_level' in disposal_info.columns:
        events = disposal_info.copy()
        # Keep necessary columns
        keep_cols = ['Stock_id', 'event_start_date', 'event_end_date', '分時交易', 'interval', 
                     '處置條件', 'condition', 'is_first_disposal', 'is_second_disposal', 'disposal_level']
        events = events[[c for c in keep_cols if c in events.columns]].copy()
             
    elif '處置開始時間' in disposal_info.columns:
        # Fallback to internal processing logic if raw data passed
        events = disposal_info[['stock_id', '處置開始時間', '處置結束時間', '分時交易', '處置條件']].copy()
        events = events.rename(columns={
            'stock_id': 'Stock_id', 
            '處置開始時間': 'event_start_date',
            '處置結束時間': 'event_end_date',
            '分時交易': 'interval',
            '處置條件': 'condition'
        })
        
        if 'condition' in events.columns:
            events['condition'] = events['condition'].astype(str).str.replace('因連續3個營業日達本中心作業要點第四條第一項第一款', '連續三次', regex=False)
            
        events['event_start_date'] = pd.to_datetime(events['event_start_date'])
        events['event_end_date'] = pd.to_datetime(events['event_end_date'])
        
        # Continuity Classification Logic (Internal Fallback)
        events = events.sort_values(['Stock_id', 'event_start_date'])
        events['prev_end_date'] = events.groupby('Stock_id')['event_end_date'].shift(1)
        events['gap_to_prev'] = (events['event_start_date'] - events['prev_end_date']).dt.days
        events['is_second_disposal'] = events['gap_to_prev'].fillna(9999) <= 3.5
        events['is_first_disposal'] = ~events['is_second_disposal']
        events = events.drop(columns=['prev_end_date', 'gap_to_prev'])
        
    else:
        # Fallback Date
        events = disposal_info[['stock_id', 'date']].rename(columns={'stock_id': 'Stock_id', 'date': 'event_start_date'})
        events['event_end_date'] = events['event_start_date']
        events['event_start_date'] = pd.to_datetime(events['event_start_date'])
        events['event_end_date'] = pd.to_datetime(events['event_end_date'])
        
        # Simple default
        events['is_first_disposal'] = True
        events['is_second_disposal'] = False
        
    
    # Remove duplicates
    events = events.drop_duplicates(subset=['Stock_id', 'event_start_date', 'event_end_date'])
    
    # Ensure dates are datetime objects
    events['event_start_date'] = pd.to_datetime(events['event_start_date'])
    events['event_end_date'] = pd.to_datetime(events['event_end_date'])
    
    # Ensure disposal_level exists
    if 'disposal_level' not in events.columns:
        if 'is_second_disposal' in events.columns:
            events['disposal_level'] = events.apply(lambda r: 2 if r['is_second_disposal'] else 1, axis=1)
        else:
            events['disposal_level'] = 1

    # 3. [Logic Change] Map event dates to Trading Index using Vectorized searchsorted
    # 此步驟將「每個事件的起始日/結束日」轉換為「該股票價格表中的索引位置 (Trading Index)」
    # 這樣後續可以直接用整數索引來篩選 [T-3, T+N]，完全避開假日問題
    
    # Pre-compute price dates map for fast lookup
    # prices is already sorted by Stock_id, Date
    # We create a dictionary: Stock_id -> (Dates Array, TradingIndices Array)
    price_dates_map = {}
    for stock_id, grp in prices.groupby('Stock_id'):
        price_dates_map[stock_id] = (grp['Date'].values, grp['trading_idx'].values)
        
    def get_event_indices(group):
        stock = group.name
        # Note: If grouping by column, name is the value of the key
        
        if stock not in price_dates_map:
            # No price data for this stock
            group['trading_idx_start'] = np.nan
            group['trading_idx_end'] = np.nan
            return group
            
        p_dates, p_idxs = price_dates_map[stock]
        
        # Start Index Search (Left side): Find insertion point for Start Date
        # If Start Date is a holiday, this gives the NEXT trading day index (exactly what we want)
        s_starts = np.searchsorted(p_dates, group['event_start_date'].values, side='left')
        
        # End Index Search (Right side - 1): Find insertion point for End Date
        # If End Date is a holiday, 'right' gives next trading day. 
        # But we want the period covering End Date. 
        # If End Date=Sunday, we want up to Friday. searchsorted(Sun, right)=Mon_idx. Mon_idx-1 = Fri_idx. Correct.
        s_ends = np.searchsorted(p_dates, group['event_end_date'].values, side='right') - 1
        
        # Handle Indices Out of Bounds
        # 1. Start is after all data (s_starts >= len) -> Event is in future or missing data
        # 2. End is before all data (s_ends < 0) -> Event is in past or missing data
        
        valid_start = s_starts < len(p_dates)
        valid_end = s_ends >= 0
        
        # Initialize result with NaN
        t_start = np.full(len(group), np.nan)
        t_end = np.full(len(group), np.nan)
        
        # Clip for safe indexing
        s_starts_clamped = np.clip(s_starts, 0, len(p_dates) - 1)
        s_ends_clamped = np.clip(s_ends, 0, len(p_dates) - 1)
        
        # Map array index -> Trading Index (p_idxs)
        # Note: p_idxs[i] is the trading_idx value at array position i
        
        # Fill valid values
        # We need to be careful with boolean indexing on arrays vs Series
        # Extract results first
        val_starts = p_idxs[s_starts_clamped].astype(float)
        val_ends = p_idxs[s_ends_clamped].astype(float)
        
        # Apply mask
        val_starts[~valid_start] = np.nan
        val_ends[~valid_end] = np.nan
        
        group['trading_idx_start'] = val_starts
        group['trading_idx_end'] = val_ends
        
        return group

    # Apply mapping
    # Note: group_keys=False prevents index expansion
    if not events.empty:
        events = events.groupby('Stock_id', group_keys=False).apply(get_event_indices)
    else:
        events['trading_idx_start'] = np.nan
        events['trading_idx_end'] = np.nan

    # 4. 合併與篩選 (Merge and Filter)
    merged = pd.merge(prices, events, on='Stock_id', how='inner')
    
    # 這裡的 offset_days 代表「前後 N 筆交易日」 (ex: 5 days means 5 candles)
    # 而不再是日曆天
    mask = (merged['trading_idx'] >= merged['trading_idx_start'] - offset_days) & \
           (merged['trading_idx'] <= merged['trading_idx_end'] + offset_days)
    
    event_study_df = merged[mask].copy()
    
    # 5. 計算相對天數
    # Calculate relative days based purely on Trading Index Diff
    event_study_df['relative_day'] = (event_study_df['trading_idx'] - event_study_df['trading_idx_start'])
    event_study_df['relative_day_end'] = (event_study_df['trading_idx'] - event_study_df['trading_idx_end'])
    
    # Drop rows where start index was invalid (NaN)
    event_study_df = event_study_df.dropna(subset=['relative_day'])
    event_study_df['relative_day'] = event_study_df['relative_day'].astype(int)
    
    # 計算日曆天 Gap (For reference)
    event_study_df['calendar_relative_day'] = (event_study_df['Date'] - event_study_df['event_start_date']).dt.days

    # 6. 產生 t_label (s+N / e+N format)
    def generate_label(row):
        # 優先判斷是否已經過了解除日 (e+N)
        # 邏輯：如果你在 End Date 之後 (trading_idx > trading_idx_end)，則算 e+
        # 或者 Date >= Event End Date
        
        # 使用 trading_idx 判斷最準
        # 如果當天 index >= end index，則屬於 e+系列
        # 但要注意：e+0 定義為「不處置的第一天」還是「處置最後一天」？
        # 通常處置期間是 [Start, End]。
        # 用戶指示：End Date should be e+0? 
        # 之前的邏輯：`if row['Date'] >= row['event_end_date']` return e+
        # 如果 End Date 是週五，週五當天符合 >= End Date。所以週五是 e+0?
        # 如果週五是最後一天處置日，通常 e+0 是指「恢復正常的第一天」or 「最後一天」?
        # 依照慣例，s+0 是開始第一天。
        # 如果 End Date 是處置最後一天。那 End Date 當天應該還是處置中 (s+N)。
        # 而 e+1 才是恢復正常。
        # 但這取決於使用者習慣。
        # (原代碼邏輯)：`if date >= end_date: e+...`
        # 這樣 End Date 當天會變成 e+0 (因為 diff=0)。
        # 我們維持原邏輯。
        
        idx_current = row['trading_idx']
        idx_end = row['trading_idx_end']
        
        if idx_current >= idx_end:
            # e+N
            val = int(idx_current - idx_end)
            return f'e+{val}'
            
        # s+N / s-N
        val_s = int(row['relative_day'])
        if val_s >= 0:
            return f's+{val_s}'
        else:
            return f's{val_s}' # s-3

    t_label_series = event_study_df.apply(generate_label, axis=1)
    
    # Split into columns dynamically based on level
    level_map = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
    unique_levels = sorted(event_study_df['disposal_level'].unique())
    print(f"Detected Disposal Levels: {unique_levels}")
    
    for lvl in unique_levels:
        suffix = level_map.get(lvl, f"level_{lvl}")
        col_name = f't_label_{suffix}'
        event_study_df[col_name] = event_study_df.apply(
            lambda row: t_label_series[row.name] if row['disposal_level'] == lvl else None, axis=1
        )
    
    # [Added] Capture Long Format Dataframe before pivoting
    long_df = event_study_df.copy()
    
    # 7. 轉置為寬表格 (Wide Format)
    print("Converting to Wide Format...")
    
    # A. 準備基礎價格表
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    base_cols = ['Date', 'Stock_id'] + price_cols
    df_base = event_study_df[base_cols].drop_duplicates(subset=['Date', 'Stock_id'])
    
    # B. 準備事件屬性欄位
    event_attrs = ['condition', 'interval', 'event_start_date', 'event_end_date', 
                   'relative_day', 'gap_days', 'calendar_relative_day', 'trading_idx_start', 'trading_idx_end', 'relative_day_end']
    existing_attrs = [col for col in event_attrs if col in event_study_df.columns]
    
    # C. Dynamic Processing
    final_df = df_base.copy()
    
    for lvl in unique_levels:
        suffix = level_map.get(lvl, f"level_{lvl}")
        df_lvl = event_study_df[event_study_df['disposal_level'] == lvl].copy()
        
        if not df_lvl.empty:
            df_lvl = df_lvl.sort_values('event_start_date').drop_duplicates(subset=['Date', 'Stock_id'], keep='last')
            rename_dict = {col: f"{col}_{suffix}" for col in existing_attrs}
            t_label_col = f't_label_{suffix}'
            cols_to_keep = ['Date', 'Stock_id', t_label_col] + existing_attrs
            df_lvl = df_lvl[cols_to_keep].rename(columns=rename_dict)
            final_df = final_df.merge(df_lvl, on=['Date', 'Stock_id'], how='left')
        
    final_df = final_df.sort_values(['Stock_id', 'Date'])
    
    # Calculate returns for Long format
    long_df['daily_ret'] = (long_df['Close']/long_df['Open']) - 1 
    
    # Wide format cleanup
    cols_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'daily_ret']
    valid_drop_cols = [c for c in cols_to_drop if c in final_df.columns]
    final_df = final_df.drop(columns=valid_drop_cols)
    
    print(f"Analysis completed. Wide shape: {final_df.shape}, Long shape: {long_df.shape}")
    return final_df, long_df

def fetch_and_merge_indexes_from_postgres(price_df, pg_client, taiex_id='TAIEX'):
    """
    從 PostgreSQL 抓取大盤 (TAIEX) 與產業指數並合併至 price_df。
    取代原本使用 FinMind API 的版本。
    """
    
    
    industry_map = {
        '水泥工業': 'Cement',
        '食品工業': 'Food',
        '塑膠工業': 'Plastics',
        '紡織纖維': 'Textiles',
        '電機機械': 'ElectricMachinery',
        '電器電纜': 'ElectricalCable',
        '化學工業': 'Chemical',
        '生技醫療業': 'BiotechnologyMedicalCare',
        '生技醫療': 'BiotechnologyMedicalCare',
        '玻璃陶瓷': 'GlassCeramic',
        '造紙工業': 'PaperPulp',
        '鋼鐵工業': 'IronSteel',
        '橡膠工業': 'Rubber',
        '汽車工業': 'Automobile',
        '半導體業': 'Semiconductor',
        '半導體': 'Semiconductor',
        '電腦及週邊': 'ComputerPeripheralEquipment',
        '電腦及週邊設備業': 'ComputerPeripheralEquipment',
        '光電業': 'Optoelectronic',
        '光電': 'Optoelectronic',
        '通信網路業': 'CommunicationsInternet',
        '通信網路': 'CommunicationsInternet',
        '電子零組件': 'ElectronicPartsComponents',
        '電子零組件業': 'ElectronicPartsComponents',
        '電子通路業': 'ElectronicProductsDistribution',
        '電子通路': 'ElectronicProductsDistribution',
        '資訊服務業': 'InformationService',
        '資訊服務': 'InformationService',
        '其他電子業': 'OtherElectronic',
        '其他電子': 'OtherElectronic',
        '其他電子類': 'OtherElectronic',
        '建材營造': 'BuildingMaterialConstruction',
        '建材營造業': 'BuildingMaterialConstruction',
        '航運業': 'ShippingTransportation',
        '航運': 'ShippingTransportation',
        '觀光餐旅': 'Tourism', 
        '觀光事業': 'Tourism',
        '觀光': 'Tourism',
        '金融保險': 'FinancialInsurance',
        '金融保險業': 'FinancialInsurance',
        '貿易百貨': 'TradingConsumersGoods',
        '貿易百貨業': 'TradingConsumersGoods',
        '油電燃氣': 'OilGasElectricity',
        '油電燃氣業': 'OilGasElectricity',
        '其他': 'Other',
        '電子工業': 'Electronic',
        '化學生技醫療': 'ChemicalBiotechnologyMedicalCare',
        '文創': 'CulturalCreative',
        '文化創意業': 'CulturalCreative',
        '農業科技': 'AgriculturalTechnology',
        '農業科技業': 'AgriculturalTechnology',
        '電子商務': 'ECommerce',
        '電子商務業': 'ECommerce',
        '運動休閒': 'SportLeisure',
        '運動休閒類': 'SportLeisure',
        '居家生活': 'HomeLife',
        '居家生活類': 'HomeLife',
        '綠能環保': 'GreenEnergyEnvironmental',
        '綠能環保類': 'GreenEnergyEnvironmental',
        '數位雲端': 'DigitalCloud',
        '數位雲端類': 'DigitalCloud',
        # Ignore these
        'ETF': None,
        '上櫃指數股票型基金(ETF)': None,
        '創新版股票': None,
        '創新板股票': None,
        '存託憑證': None,
    }

    # [Critical Fix] Ensure price_df['Date'] is datetime object for merging
    if 'Date' in price_df.columns:
        price_df = price_df.copy() # Avoid SettingWithCopy
        price_df['Date'] = pd.to_datetime(price_df['Date'])

    # 假設指數資料在 taiwan_stock_daily 表中，且 Stock_id 為 'TAIEX'
    query_taiex = f"""
        SELECT "Date", "Open", "High", "Low", "Close"
        FROM public.taiwan_stock_daily
        WHERE "Stock_id" = '{taiex_id}'
        AND "Date" >= '2018-01-01'
        ORDER BY "Date"
    """
    
    try:
        with pg_client._get_engine().connect() as conn:
            taiex = pd.read_sql(text(query_taiex), conn)
            
        if not taiex.empty:
            taiex = taiex.rename(columns={
                'Open': 'TAIEX_Open', 'High': 'TAIEX_High', 
                'Low': 'TAIEX_Low', 'Close': 'TAIEX_Close'
            })
            taiex['Date'] = pd.to_datetime(taiex['Date'])
            
            # 串接大盤
            merged_df = price_df.merge(taiex, on='Date', how='left')
        else:
             print("[Warning] TAIEX data not found in Postgres.")
             merged_df = price_df
             
    except Exception as e:
        print(f"[Error] Failed to fetch TAIEX: {e}")
        merged_df = price_df

    
    if 'industry' not in merged_df.columns:
        print("[Warning] No 'industry' column found in price_df. Skipping industry index merge.")
        return merged_df

    # Map industries
    merged_df['Related_Index_ID'] = merged_df['industry'].map(industry_map)
    
    unknown_mask = merged_df['Related_Index_ID'].isna()
    if unknown_mask.any():
        # Get industries that resulted in NaN
        failed_inds = merged_df.loc[unknown_mask, 'industry'].unique()
        # Filter out those that we INTENTIONALLY mapped to None (e.g. ETF)
        really_unknown = [ind for ind in failed_inds if ind not in industry_map]
        
        if really_unknown:
             print(f"[Warning] Some industries could not be mapped: {really_unknown}")
        
    required_indices = merged_df['Related_Index_ID'].dropna().unique().tolist()
    print(f"Required Industry Indices: {required_indices}")
    
    if required_indices:
        indices_list_str = "', '".join(str(s) for s in required_indices)
        query_indices = f"""
            SELECT "Date", "Stock_id" as "Related_Index_ID", "Open", "High", "Low", "Close"
            FROM public.taiwan_stock_daily
            WHERE "Stock_id" IN ('{indices_list_str}')
            AND "Date" >= '2018-01-01'
            ORDER BY "Stock_id", "Date"
        """
        
        try:
             with pg_client._get_engine().connect() as conn:
                big_index_df = pd.read_sql(text(query_indices), conn)
                
             if not big_index_df.empty:
                big_index_df = big_index_df.rename(columns={
                    'Open': 'Ind_Open', 'High': 'Ind_High', 
                    'Low': 'Ind_Low', 'Close': 'Ind_Close'
                })
                big_index_df['Date'] = pd.to_datetime(big_index_df['Date'])
                
                merged_df = merged_df.merge(
                    big_index_df, 
                    on=['Date', 'Related_Index_ID'], 
                    how='left'
                )
             else:
                print("[Warning] No industry index data found in Postgres.")

        except Exception as e:
            print(f"[Error] Failed to fetch industry indices: {e}")
    else:
        print("No industry indices to fetch based on current mapping.")

    return merged_df
