import pandas as pd
from datetime import timedelta
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def process_disposal_events(disposal_info):
    """
    前處理 Finlab 處置資料：
    1. 欄位標準化 (Stock_id, event_start_date, event_end_date)
    2. 排序並計算 Gap
    3. 標記 First/Second Disposal
    """
    events = disposal_info.copy()
    
    # 1. 欄位映射與標準化
    if '處置開始時間' in events.columns:
        events = events.rename(columns={
            'stock_id': 'Stock_id', 
            '處置開始時間': 'event_start_date',
            '處置結束時間': 'event_end_date',
            '分時交易': 'interval',
            '處置條件': 'condition'
        })
    elif 'date' in events.columns:
         events = events.rename(columns={'stock_id': 'Stock_id', 'date': 'event_start_date'})
         events['event_end_date'] = events['event_start_date']

    # 確保日期格式
    events['event_start_date'] = pd.to_datetime(events['event_start_date'])
    events['event_end_date'] = pd.to_datetime(events['event_end_date'])
    
    if 'condition' in events.columns:
        events['condition'] = events['condition'].astype(str).str.replace('因連續3個營業日達本中心作業要點第四條第一項第一款', '連續三次', regex=False)

    # 2. 排序
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
            # User requested strict overlap: Start <= Chain End
            limit_date = chain_end_date 
            
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
    
    # Generate Boolean Flags for backward compatibility (optional but good for debugging)
    events['is_first_disposal'] = events['disposal_level'] == 1
    events['is_second_disposal'] = events['disposal_level'] == 2
    
    print(f"Processed {len(events)} events.")
    print("Level Distribution:")
    print(events['disposal_level'].value_counts().sort_index())
          
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
    global_min = start_dates.min() - timedelta(days=offset_days)
    global_max = end_dates.max() + timedelta(days=offset_days)
    
    start_str = global_min.strftime('%Y-%m-%d')
    end_str = global_max.strftime('%Y-%m-%d')
    
    try:
        # 初始化並抓取資料
        client.initialize_frame(stock_id=stock_id, start_time=start_str, end_time=end_str)
        price_df = client.get_stock_price()
        
        if price_df.empty:
            return None
            
        price_df = price_df.reset_index() # Ensure Date is a column
        price_df['Date'] = pd.to_datetime(price_df['Date'])
        
        # 2. 建立精確篩選遮罩 (Mask)
        mask = pd.Series(False, index=price_df.index)
        
        # 針對每一筆處置事件，把該區間 [Start-3, End+3] 的日期都標記為 True
        # 使用 zip 同時遍歷開始與結束時間
        for s_date, e_date in zip(start_dates, end_dates):
            d_start = s_date - timedelta(days=offset_days)
            d_end = e_date + timedelta(days=offset_days)
            mask |= (price_df['Date'] >= d_start) & (price_df['Date'] <= d_end)
            
        filtered_df = price_df[mask].copy()
        if filtered_df.empty:
            return None
            
        return filtered_df
        
    except Exception as e:
        print(f"Error fetching {stock_id}: {e}")
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
    disposal_events = disposal_info.copy()
    token = client.config.api_token if hasattr(client, 'config') else None

    # 欄位偵測
    # 欄位偵測 (優先使用標準化欄位)
    if 'event_start_date' in disposal_events.columns and 'event_end_date' in disposal_events.columns:
        unique_stocks = disposal_events['Stock_id'].unique()
        col_start = 'event_start_date'
        col_end = 'event_end_date'
        print(f"Using pre-processed columns '{col_start}' and '{col_end}'.")
    elif '處置開始時間' in disposal_events.columns and '處置結束時間' in disposal_events.columns:
        unique_stocks = disposal_events['stock_id'].unique()
        col_start = '處置開始時間'
        col_end = '處置結束時間'
        print(f"Detected Finlab format. Using '{col_start}' and '{col_end}' for range filtering.")
    elif 'date' in disposal_events.columns:
        # Fallback for old format (assume single day event, start=end)
        unique_stocks = disposal_events['stock_id'].unique()
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
            subset = disposal_events[disposal_events['stock_id'] == stock_id]
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
    3. 保留 [Start-Offset, End+Offset] 的所有資料
    """
    if price_df.empty:
        print("Price DataFrame is empty.")
        return pd.DataFrame()
            
    # 1. 準備股價表
    prices = price_df.sort_values(['Stock_id', 'Date']).copy()
    
    # [Added] Filter invalid prices (Open > 0) to avoid infinite returns
    prices = prices[prices['Open'] > 0].copy()
    
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    # 建立交易日索引
    prices['trading_idx'] = prices.groupby('Stock_id').cumcount()
    
    # 計算 Gap Days
    prices['prev_trade_date'] = prices.groupby('Stock_id')['Date'].shift(1)
    prices['trade_date_diff'] = (prices['Date'] - prices['prev_trade_date']).dt.days
    prices['gap_days'] = prices['trade_date_diff'].fillna(1) - 1
    
    # 2. 準備事件表 (Start & End)
    # Check if pre-processed
    if 'event_start_date' in disposal_info.columns and 'is_first_disposal' in disposal_info.columns:
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
    
    # Ensure disposal_level exists (for fallback cases)
    if 'disposal_level' not in events.columns:
        # Fallback: simple mapping from boolean if available, else all 1
        if 'is_second_disposal' in events.columns:
            events['disposal_level'] = events.apply(lambda r: 2 if r['is_second_disposal'] else 1, axis=1)
        else:
            events['disposal_level'] = 1
    
    # 3. 合併 (Cross Join logic but filtered)
    # 由於一個股票可能有多個處置區間，且區間可能重疊，最好的方式是 merge 後過濾
    # 先只 merge Stock_id
    merged = pd.merge(prices, events, on='Stock_id', how='inner')
    
    # 4. 過濾有效資料：保留在 [Start - offset, End + offset] 範圍內的列
    # 注意：這裡 offset_days 必須與 fetch 時一致或更小
    mask = (merged['Date'] >= merged['event_start_date'] - timedelta(days=offset_days)) & \
           (merged['Date'] <= merged['event_end_date'] + timedelta(days=offset_days))
    
    event_study_df = merged[mask].copy()
    
    # 5. 計算相對天數
    # 這裡我們需要找到 event_start_date 對應的 trading_idx，才能算出 t+N
    # 為了效能，我們建立一個 lookup table
    # 找出每個 (Stock_id, Date) 的 trading_idx
    trade_idx_map = prices.set_index(['Stock_id', 'Date'])['trading_idx']
    
    # 為每筆事件找出 Start Date 的 Trading Index
    # 注意：Start Date 可能不是交易日，需用 searchsorted 或類似邏輯去找 "最近的交易日"
    # 簡單做法：將 Event Start Date 對應到 Price Table (Merge left on Stock, Start Date)
    # 但如果 Start Date 是假日，會對不起來。
    # 改進：使用 merge_asof if sorted? 
    # 或是由 event_study_df 中，針對每個 event_grp，找出 Date >= Start 的第一筆，設為 t=0
    
    # 讓我們用更簡單的邏輯：
    # 已經有 trading_idx。對於每一行，我們知道它的 Date 和 event_start_date。
    # 我們需要知道 event_start_date 當天的 trading_idx。
    # 如果 event_start_date 是假日，通常處置開始日會是交易日。如果不幸是假日，取之後的第一個交易日。
    
    # 方法：
    # 計算 `Date - StartDate` 的天數 (Calendar Days)
    event_study_df['calendar_relative_day'] = (event_study_df['Date'] - event_study_df['event_start_date']).dt.days
    
    # 計算 Relative Trading Days
    # 先找出每個 (Stock, EventStart) 的基準 Trading Index
    # Step A: 找出所有 Event Start Date 在該股票 trading_idx 的位置
    # 利用 prices 表
    start_indices = prices.reset_index().merge(
        events[['Stock_id', 'event_start_date']], 
        left_on=['Stock_id', 'Date'], 
        right_on=['Stock_id', 'event_start_date'],
        how='right' # Keep all events
    )
    # 如果 StartDate 沒對到 (假日)，trading_idx 會是 NaN。需 Backfill。
    # 但這裡我們只要有對到的就好，大部分處置起始日都是交易日。
    # 為了嚴謹，若 Start Date 沒對到，我們似乎無法定義 t=0。
    # 假設 Finlab 資料的處置起始日都是交易日。
    start_idx_map = start_indices.dropna(subset=['trading_idx']).set_index(['Stock_id', 'event_start_date'])['trading_idx']
    
    # Map back to main df
    event_study_df = event_study_df.join(start_idx_map, on=['Stock_id', 'event_start_date'], rsuffix='_start')
    
    # 計算 relative day
    event_study_df['relative_day'] = event_study_df['trading_idx'] - event_study_df['trading_idx_start']
    
    # 若有 NaN (表示起始日非交易日)，則無法計算精確 t，暫時填入 NaN 或移除
    event_study_df = event_study_df.dropna(subset=['relative_day'])
    event_study_df['relative_day'] = event_study_df['relative_day'].astype(int)

    event_study_df = event_study_df.dropna(subset=['relative_day'])
    event_study_df['relative_day'] = event_study_df['relative_day'].astype(int)

    # [Removed] Old Interval-based Classification Logic 
    # (Classification is now done in Step 2 based on continuity)

    # 6. 產生 t_label
    def format_t(x):
        if x > 0: return f't+{x}'
        elif x < 0: return f't{x}'
        else: return 't+0'
    # Temporary generic label
    t_label_series = event_study_df['relative_day'].apply(format_t)
    
    # Split into two columns (Now safe to use is_first_disposal)
    # Split into columns dynamically based on level
    # Map level number to suffix name
    level_map = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
    
    unique_levels = sorted(event_study_df['disposal_level'].unique())
    print(f"Detected Disposal Levels: {unique_levels}")
    
    for lvl in unique_levels:
        suffix = level_map.get(lvl, f"level_{lvl}") # Fallback to level_5, level_6...
        col_name = f't_label_{suffix}'
        event_study_df[col_name] = event_study_df.apply(
            lambda row: t_label_series[row.name] if row['disposal_level'] == lvl else None, axis=1
        )
    
    # 7. 轉置為寬表格 (Wide Format) 以消除重複日期
    print("Converting to Wide Format...")
    
    # A. 準備基礎價格表 (Base Price Data) - 確保唯一性
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    base_cols = ['Date', 'Stock_id'] + price_cols
    df_base = event_study_df[base_cols].drop_duplicates(subset=['Date', 'Stock_id'])
    
    # B. 準備事件屬性欄位
    # 這些欄位需要分流 (Split)
    event_attrs = ['condition', 'interval', 'event_start_date', 'event_end_date', 
                   'relative_day', 'gap_days', 'calendar_relative_day']
    # 僅保留實際存在的欄位
    existing_attrs = [col for col in event_attrs if col in event_study_df.columns]
    
    
    # C. Dynamic Processing for each level
    # 處理每一個 Level
    for lvl in unique_levels:
        suffix = level_map.get(lvl, f"level_{lvl}")
        
        # Filter data for this level
        df_lvl = event_study_df[event_study_df['disposal_level'] == lvl].copy()
        
        if not df_lvl.empty:
            # Deduplicate: Keep latest event start date
            df_lvl = df_lvl.sort_values('event_start_date').drop_duplicates(subset=['Date', 'Stock_id'], keep='last')
            
            # Rename columns
            rename_dict = {col: f"{col}_{suffix}" for col in existing_attrs}
            
            # Keep t_label column
            t_label_col = f't_label_{suffix}'
            
            cols_to_keep = ['Date', 'Stock_id', t_label_col] + existing_attrs
            df_lvl = df_lvl[cols_to_keep].rename(columns=rename_dict)
            
            # Merge to final
            final_df = final_df.merge(df_lvl, on=['Date', 'Stock_id'], how='left')
        
    # F. 最終整理
    final_df = final_df.sort_values(['Stock_id', 'Date'])
    final_df['daily_ret'] = (final_df['Close']/final_df['Open']) - 1
    
    print(f"Analysis completed. Result shape: {final_df.shape}")
    return final_df
