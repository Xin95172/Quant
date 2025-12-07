
import pandas as pd
from datetime import timedelta
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

def fetch_and_filter_price(client, stock_id, event_dates, offset_days=3):
    """
    針對單檔股票，抓取處置日前後指定天數的股價資料。
    """
    if event_dates.empty:
        return None

    # 決定抓取的時間範圍 (min - offset ~ max + offset)
    min_date = event_dates.min() - timedelta(days=offset_days)
    max_date = event_dates.max() + timedelta(days=offset_days)
    
    start_str = min_date.strftime('%Y-%m-%d')
    end_str = max_date.strftime('%Y-%m-%d')
    
    try:
        # 初始化並抓取資料
        # 注意：假設 client 有 initialize_frame 和 get_stock_price 方法
        client.initialize_frame(stock_id=stock_id, start_time=start_str, end_time=end_str)
        price_df = client.get_stock_price()
        
        if price_df.empty:
            return None
            
        price_df = price_df.reset_index()
        
        # 建立篩選遮罩 (Mask)
        # 只要日期落在任一處置事件的前後範圍內，即保留
        mask = pd.Series(False, index=price_df.index)
        
        # 優化：向量化處理日期篩選可能比較困難，因為區間多且不連續
        # 但我們可以先轉換為 datetime 以加速比較
        price_dates = pd.to_datetime(price_df['Date'])
        
        for event_date in event_dates:
            d_start = event_date - timedelta(days=offset_days)
            d_end = event_date + timedelta(days=offset_days)
            mask |= (price_dates >= d_start) & (price_dates <= d_end)
            
        filtered_df = price_df[mask].copy()
        if filtered_df.empty:
            return None
            
        return filtered_df
        
    except Exception as e:
        print(f"Error fetching {stock_id}: {e}")
        return None

# Add explicit import inside function or global import if possible
# But to avoid circular imports if this module is imported by others, better to import inside.
# However, standard practice is top-level. Let's add top-level import for FinMindClient via string or pass the class?
# Easier: Just import it inside the function or assume access.
# Let's add the import at the top of the file first in another step? 
# OR: We can use the existing client to get the token, and make new clients.

def _fetch_worker(stock_id, dates, offset_days, token=None):
    from module.get_info_FinMind import FinMindClient
    # Create a new local client for thread safety
    local_client = FinMindClient()
    if token:
        local_client.login_by_token(token)
    return fetch_and_filter_price(local_client, stock_id, dates, offset_days)

def batch_fetch_prices(client, disposal_info, offset_days=3, max_workers=8):
    """
    使用多執行緒平行抓取所有股票的價格資料。
    支援 Finlab 資料格式 (自動偵測 '處置開始時間' 欄位)。
    [Safety Fix] Create independent clients for each thread to avoid race conditions.
    """
    disposal_events = disposal_info.copy()
    
    # Extract token from the passed client to reuse
    token = client.config.api_token if hasattr(client, 'config') else None

    # 偵測是否為 Finlab 資料格式
    is_finlab = '處置開始時間' in disposal_events.columns
    
    if is_finlab:
        # 使用 'stock_id' 和 '處置開始時間'
        unique_stocks = disposal_events['stock_id'].unique()
        date_col = '處置開始時間'
    else:
        # 舊有邏輯
        unique_stocks = disposal_events['stock_id'].unique()
        date_col = 'date'
    
    print(f"Starting batch fetch for {len(unique_stocks)} stocks with {max_workers} workers...")
    print(f"Using date column: {date_col}")
    
    all_prices = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交任務
        future_to_stock = {}
        for stock_id in unique_stocks:
            stock_dates = disposal_events[disposal_events['stock_id'] == stock_id][date_col]
            # 確保日期格式正確
            stock_dates = pd.to_datetime(stock_dates)
            
            # [Fix] Use _fetch_worker to instantiate new client per thread
            future = executor.submit(_fetch_worker, stock_id, stock_dates, offset_days, token)
            future_to_stock[future] = stock_id
            
        # 處理結果 (顯示進度條)
        for future in tqdm(as_completed(future_to_stock), total=len(unique_stocks), desc="Fetching Prices"):
            stock_id = future_to_stock[future]
            try:
                result_df = future.result()
                if result_df is not None:
                    all_prices.append(result_df)
            except Exception as exc:
                print(f"{stock_id} generated an exception: {exc}")
    
    # 合併與整理
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
    1. 判定處置起始日 (Event Definition) - 支援 Finlab 直接指定
    2. 計算交易日索引 (Trading Index)
    3. 計算假日間隔 (Gap Days)
    4. 計算相對天數 (Relative Days)
    """
    if price_df.empty:
        print("Price DataFrame is empty.")
        return pd.DataFrame()
            
    # 1. 準備股價表
    prices = price_df.sort_values(['Stock_id', 'Date']).copy()
    prices['Date'] = pd.to_datetime(prices['Date'])
    
    # 建立交易日索引 (0, 1, 2...)
    prices['trading_idx'] = prices.groupby('Stock_id').cumcount()
    
    # 計算 Gap Days (前移至合併前計算)
    prices['prev_trade_date'] = prices.groupby('Stock_id')['Date'].shift(1)
    prices['trade_date_diff'] = (prices['Date'] - prices['prev_trade_date']).dt.days
    prices['gap_days'] = prices['trade_date_diff'].fillna(1) - 1
    
    # 2. 準備事件表 (Start Date Detection)
    # 判斷是否為 Finlab 資料
    if '處置開始時間' in disposal_info.columns:
        events = disposal_info[['stock_id', '處置開始時間', '分時交易', '處置條件']].copy()
        events = events.rename(columns={
            'stock_id': 'Stock_id', 
            '處置開始時間': 'event_date',
            '分時交易': 'interval',
            '處置條件': 'condition'
        })
        
        if 'condition' in events.columns:
            events['condition'] = events['condition'].astype(str).str.replace('因連續3個營業日達本中心作業要點第四條第一項第一款', '連續三次', regex=False)
            
        start_events = events.copy()
        start_events['event_date'] = pd.to_datetime(start_events['event_date'])
        start_events = start_events.drop_duplicates(subset=['Stock_id', 'event_date'])
        
    else:
        events = disposal_info[['stock_id', 'date']].sort_values(['stock_id', 'date'])
        events = events.rename(columns={'stock_id': 'Stock_id', 'date': 'event_date'})
        events['event_date'] = pd.to_datetime(events['event_date'])
        
        # 找出連續期間的起始日
        events['prev_date'] = events.groupby('Stock_id')['event_date'].shift(1)
        events['days_diff'] = (events['event_date'] - events['prev_date']).dt.days
        is_start = events['days_diff'].isna() | (events['days_diff'] > 1)
        start_events = events[is_start][['Stock_id', 'event_date']].copy()
    
    # 3. 找出事件日對應的交易日索引
    event_indices = pd.merge(
        start_events, 
        prices[['Stock_id', 'Date', 'trading_idx']], 
        left_on=['Stock_id', 'event_date'], 
        right_on=['Stock_id', 'Date'], 
        how='inner'
    )
    
    # [Modified] Dynamic column selection for retention
    cols_to_keep = ['Stock_id', 'event_date', 'trading_idx']
    if 'condition' in start_events.columns:
        cols_to_keep.append('condition')
    if 'interval' in start_events.columns:
        cols_to_keep.append('interval')
        
    event_indices = event_indices[cols_to_keep].rename(columns={'trading_idx': 'event_idx'})
    
    # 4. 合併股價與事件索引
    merged = pd.merge(prices, event_indices, on='Stock_id', how='inner')
    
    # 5. 計算相對天數 (Relative Trading Days)
    merged['relative_day'] = merged['trading_idx'] - merged['event_idx']
    
    # 6. 計算自然日相對天數
    merged['calendar_relative_day'] = (merged['Date'] - merged['event_date']).dt.days
    
    # 7. 篩選範圍
    mask = (merged['relative_day'] >= -offset_days) & (merged['relative_day'] <= offset_days)
    event_study_df = merged[mask].copy()
    
    # 8. 產生 t_label
    def format_t(x):
        if x > 0: return f't+{x}'
        elif x < 0: return f't{x}'
        else: return 't+0'
    event_study_df['t_label'] = event_study_df['relative_day'].apply(format_t)
    
    # 9. 正規化 (Normalization) - 已移除
    pass
    
    # 10. 整理欄位
    base_cols = ['Date', 'Stock_id', 't_label', 'relative_day', 'gap_days', 'calendar_relative_day']
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    cols = list(base_cols)
    # Add condition and interval before price columns
    if 'condition' in event_study_df.columns:
        cols.append('condition')
    if 'interval' in event_study_df.columns:
        cols.append('interval')
        
    cols.extend(price_cols)
    
    final_df = event_study_df[cols].sort_values(['Stock_id', 'Date'])
    final_df['daily_ret'] = (final_df['Close']/final_df['Open']) - 1
    
    print(f"Analysis completed. Result shape: {final_df.shape}")
    return final_df
