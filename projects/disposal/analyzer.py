import pandas as pd
import numpy as np
import plotly.graph_objects as go
from module.plot_func import plot
from typing import Optional, List
from IPython.display import display

class DisposalAnalyzer:
    """
    處置股效果分析器
    封裝了多種針對處置股事件的研究方法
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def display_dataframe(self):
        display(self.df)

    @staticmethod
    @staticmethod
    def _parse_t_val(t_str: str) -> float:
        """解析時間標籤 (e.g., 's+1', 'e-5') 為數值以利排序 - [Optimized]"""
        if not isinstance(t_str, str) or not t_str: return 999.0
        
        # Fast parsing assuming standard format s/e +/- N
        try:
            prefix = t_str[0] # 's' or 'e'
            # Check for sign
            if '+' in t_str:
                val = int(t_str.split('+')[1])
            elif '-' in t_str or ('s' in t_str and len(t_str) > 1 and t_str[1] == '-'):
                # s-3 or s3 (if s-3 is stored as s-3)
                # Assuming 's-3' format or 's3' (negative not explicitly allowed in split old logic?)
                # Old logic: if 's-' in t_str: val = -int(t_str.split('-')[-1])
                parts = t_str.split('-')
                val = -int(parts[1]) if len(parts) > 1 else 0
            else:
                # s3 (s+3) format or just number? Assuming sN is positive if no sign
                val = int(t_str[1:])
                
            if prefix == 's': return float(val)
            elif prefix == 'e': return 1000.0 + val
        except:
            return 999.0
        return 999.0

    def _compute_returns(self, df: pd.DataFrame, session: str) -> pd.DataFrame:
        """根據 session 計算個股與大盤報酬"""
        df = df.copy()
        
        # 動態建立分組欄位，避免 KeyError
        group_cols = ['Stock_id']
        for col in ['base_start_date', 'base_end_date']:
            if col in df.columns:
                group_cols.append(col)
        
        # 1. Stock Return
        if session == 'position':
            df['daily_ret'] = (df['Close'] / df['Open']) - 1
            if 'TAIEX_Close' in df.columns:
                df['market_ret'] = (df['TAIEX_Close'] / df['TAIEX_Open']) - 1
        elif session == 'after_market':
            df['prev_close'] = df.groupby(group_cols)['Close'].shift(1)
            df['daily_ret'] = (df['Open'] / df['prev_close']) - 1
            if 'TAIEX_Close' in df.columns:
                df['market_prev_close'] = df.groupby(group_cols)['TAIEX_Close'].shift(1)
                df['market_ret'] = (df['TAIEX_Open'] / df['market_prev_close']) - 1
        elif session == 'all':
            df['prev_close'] = df.groupby(group_cols)['Close'].shift(1)
            df['daily_ret'] = (df['Close'] / df['prev_close']) - 1
            if 'TAIEX_Close' in df.columns:
                 df['market_prev_close'] = df.groupby(group_cols)['TAIEX_Close'].shift(1)
                 df['market_ret'] = (df['TAIEX_Close'] / df['market_prev_close']) - 1

        # 2. Abnormal Return
        if 'market_ret' in df.columns:
            df['abnormal_ret'] = df['daily_ret'] - df['market_ret']
        else:
            df['abnormal_ret'] = np.nan
            
        return df

    def _plot_stats(self, stats: pd.DataFrame, target_col: str, note: str, return_marks: str | None = None, v_lines: list | None = None):
        """呼叫繪圖模組"""
        # 排序
        stats['sort_key'] = stats[target_col].apply(self._parse_t_val)
        stats = stats.sort_values('sort_key').drop(columns=['sort_key'])
        
        plot(
            df=stats,
            x=target_col,
            ly='mean',       # 上圖：平均超額報酬
            bar_col='count', # 下圖：樣本數
            ly_type='bar',   # 上圖強制畫長條
            mid_col='std',   # 中圖：標準差 (變更參數名稱)
            note=note,
            bar_kwargs={'width': 0.8},
            return_marks=return_marks,
            v_lines=['s+0', 'e+0']
        )

    def _display_dataframe(self, col: str, title: str):
        """以 Pandas DataFrame 顯示分佈統計 (Days vs Events)"""
        if col not in self.df.columns:
            return
            
        print(f"\n[{title}]")
        
        # 1. Days Count (Rows)
        days_counts = self.df[col].value_counts().rename('days_count')
        
        # 2. Event Count (Unique Events)
        # 只有當 'Stock_id' 和 'event_start_date' 存在時才能算
        if 'Stock_id' in self.df.columns and 'event_start_date' in self.df.columns:
            # Drop duplicates to get unique events
            unique_events = self.df.drop_duplicates(subset=['Stock_id', 'event_start_date'])
            event_counts = unique_events[col].value_counts().rename('event_count')

            extras = []
            if 'interval' in unique_events.columns:
                s_interval = unique_events['interval'].astype(str)

                # 抓5分盤
                c5 = unique_events[s_interval.str.contains('5')][col].value_counts().rename('5min_count')

                # 抓20分盤
                c20 = unique_events[s_interval.str.contains('20')][col].value_counts().rename('20min_count')
                
                extras = [c5, c20]

            # Combine
            stats = pd.concat([days_counts, event_counts] + extras, axis=1).fillna(0).astype(int)
        else:
            # Fallback if metadata missing
            stats = pd.DataFrame(days_counts)
            stats['event_count'] = 'N/A'
            
        stats.index.name = col
        stats = stats.reset_index()
        
        # Percentage (based on Days)
        total_days = len(self.df)
        stats['days_pct'] = (stats['days_count'] / total_days * 100).map('{:.2f}%'.format)
        
        # Sort
        if pd.api.types.is_numeric_dtype(stats[col]):
            stats = stats.sort_values(col)
        else:
            stats = stats.sort_values('days_count', ascending=False)
            
        # Display
        # Display
        display_cols = [col, 'days_count', 'event_count', 'days_pct']
        if '5min_count' in stats.columns:
            display_cols.extend(['5min_count', '20min_count'])
        
        if 'event_count' in stats.columns and stats['event_count'].dtype != 'O':
             # Format numbers
             fmt = {'days_count': '{:,}', 'event_count': '{:,}'}
             if '5min_count' in stats.columns:
                 fmt.update({'5min_count': '{:,}', '20min_count': '{:,}'})
                 
             display(stats[display_cols].style.format(fmt))
        else:
             display(stats[display_cols])

    def overall_analysis(self, min_samples: int = 50, prefix: str = 't_label_'):
        """
        [Method 1] 多層級處置分析
        計算各時間點 (t_label_*) 的平均日報酬率並繪圖。
        """
        if self.df.empty:
            print("Dataframe is empty. Please check data source.")
            return

        self._display_dataframe('condition', 'Disposal Condition Distribution')

        if 'disposal_level' in self.df.columns:

            # Correct Event Count Logic
            # Days Count: Number of rows (trading days)
            # Event Count: Number of distinct events (identified by 's+0' marker in corresponding t_label column)
            
            # 1. Calculate basic stats (based on Days)
            level_stats = self.df.groupby('disposal_level')['daily_ret'].agg(['mean', 'count', 'std']).reset_index()
            level_stats = level_stats.rename(columns={'count': 'days_count'})
            
            # 2. Calculate true Event Count
            event_counts = {}
            unique_levels = self.df['disposal_level'].dropna().unique()
            level_map = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
            
            for lvl in unique_levels:
                # Find the column that represents this level's timing
                suffix = level_map.get(lvl, f"level_{int(lvl)}")
                t_col = f"t_label_{suffix}"
                
                if t_col in self.df.columns:
                    # Count how many times 's+0' appears for this level
                    # This assumes 's+0' is present for every event. If data is truncated, this might undercount,
                    # but it's the most accurate proxy available in the Wide format.
                    # We filter by disposal_level to ensure we are looking at the right rows
                    count = self.df[
                        (self.df['disposal_level'] == lvl) & 
                        (self.df[t_col] == 's+0')
                    ].shape[0]
                    event_counts[lvl] = count
                else:
                    event_counts[lvl] = 0
            
            # Map event counts back to stats
            level_stats['event_count'] = level_stats['disposal_level'].map(event_counts).fillna(0).astype(int)
            
            # Sort
            level_stats = level_stats.sort_values('disposal_level')
            
            # Display readable table
            print("\n[Disposal Level Statistics]")
            display_cols = ['disposal_level', 'days_count', 'event_count', 'mean', 'std']
            display(level_stats[display_cols].style.format({
                'mean': '{:.2%}', 
                'std': '{:.2%}',
                'days_count': '{:,}',
                'event_count': '{:,}'
            }))

            plot(
                df=level_stats,
                x='disposal_level',
                ly='mean',       # Top: Mean Daily Return
                bar_col='days_count', # Bottom: Days Count (more relevant for significance)
                ly_type='bar',
                note="Disposal Effect by Level",
                bar_kwargs={'width': 0.8}
            )

    def seprate_by_trend(self):
        """
        比較處置原因：利用 s-3 至 s-1 的累積漲跌幅區分超漲 (Overbought) / 超跌 (Oversold)
        """
        if self.df.empty:
            print("Dataframe is empty. Please check data source.")
            return

        required_cols = ['Stock_id', 'event_start_date', 'daily_ret', 'disposal_level']
        if not all(col in self.df.columns for col in required_cols):
             print(f"缺少必要欄位 {required_cols}，無法進行趨勢分析。")
             return

        # 準備方向判斷
        event_trends = []

        unique_levels = self.df['disposal_level'].dropna().unique()
        level_map = {1: 'first', 2: 'second', 3: 'third', 4: 'fourth'}
        
        for lvl in unique_levels:
            suffix = level_map.get(lvl, f"level_{int(lvl)}")
            t_col = f"t_label_{suffix}"
            if t_col not in self.df.columns: continue
            # 分別取出 s-3 (Open) 與 s-1 (Close)
            mask_s3 = (self.df['disposal_level'] == lvl) & (self.df[t_col] == 's-3')
            df_s3 = self.df.loc[mask_s3, ['Stock_id', 'event_start_date', 'Open']].rename(columns={'Open': 'Open_s3'})
            
            mask_s1 = (self.df['disposal_level'] == lvl) & (self.df[t_col] == 's-1')
            df_s1 = self.df.loc[mask_s1, ['Stock_id', 'event_start_date', 'Close']].rename(columns={'Close': 'Close_s1'})
            
            if df_s3.empty or df_s1.empty: continue

            # 合併計算區間報酬: (s-1 Close / s-3 Open) - 1
            merged = pd.merge(df_s3, df_s1, on=['Stock_id', 'event_start_date'], how='inner')
            merged['cum_ret'] = (merged['Close_s1'] / merged['Open_s3']) - 1
            
            merged['direction'] = np.where(merged['cum_ret'] > 0, 'Overbought', 'Oversold')
            event_trends.append(merged[['Stock_id', 'event_start_date', 'direction']])
            
        if not event_trends:
            print("找不到 s-3 ~ s-1 的時間標籤資料")
            return
            
        # 合併所有事件的判斷結果
        all_trends = pd.concat(event_trends, ignore_index=True)
        
        # 將判斷結果 Merge 回主表
        if 'direction' in self.df.columns:
            self.df = self.df.drop(columns=['direction'])
            
        self.df = self.df.merge(all_trends, on=['Stock_id', 'event_start_date'], how='left')
        self.df['direction'] = self.df['direction'].fillna('Unknown')

        # 1. 顯示基本分佈
        self._display_dataframe('direction', '處置前趨勢分佈 (s-3 ~ s-1)')

        # 2. 交叉分析
        if 'disposal_level' in self.df.columns:
            print("\n[方向 vs 層級 交叉表 (交易天數)]")
            ct = pd.crosstab(
                self.df['direction'],
                self.df['disposal_level'],
                margins=True,
                margins_name='Total'
            )
            display(ct)
        
        
        # 重新定義 "t_label"
        # 邏輯：
        # 1. 識別連續處置事件
        # 2. 定義區間：
        #    - Start Date: 該組中 Level 最小 (通常是1) 的 event_start_date -> s+0
        #    - End Date:   該組中 Level 最大 (通常是處置最後層級) 的 event_end_date -> e+0
        # 3. 標籤規則：
        #    - 若日期 < End Date: 標記為 s 系列 (相對於 Start Date)
        #    - 若日期 >= End Date: 標記為 e 系列 (相對於 End Date, 當天為 e+0)
        
        df_process = self.df.copy()
        df_process.drop_duplicates(subset=['Date', 'Stock_id'], inplace=True)
        if 'Date' not in df_process.columns:
             df_process['Date'] = pd.to_datetime(df_process['Date'])

        # 排序：Stock -> Date
        df_process = df_process.sort_values(['Stock_id', 'Date'])
        
        # 生成分組 Group ID (同前次邏輯)
        df_process['prev_stock'] = df_process['Stock_id'].shift(1)
        df_process['prev_level'] = df_process['disposal_level'].shift(1)
        cond_stock_change = df_process['Stock_id'] != df_process['prev_stock']
        is_level_1 = df_process['disposal_level'] == 1
        prev_not_1 = df_process['prev_level'] != 1
        
        # Check for start date change to identify new event chains accurately
        df_process['prev_start_date'] = df_process['event_start_date'].shift(1)
        cond_start_date_change = df_process['event_start_date'] != df_process['prev_start_date']
        
        # New Group Logic:
        # 1. Stock Change (Handled by cond_stock_change)
        # 2. Level Reset: Level 1 AND Start Date Changed
        #    - If Previous was Level 2, 3... -> Level 1 (New Chain or Start Date change) -> Reset
        #    - If Previous was Level 1 -> Level 1 (Different Start Date) -> Reset
        cond_new_event = is_level_1 & cond_start_date_change
        
        df_process['new_group'] = cond_stock_change | cond_new_event
        df_process['group_id'] = df_process['new_group'].cumsum()
        
        # 取得每個 Group 的 Start (Min Level) 與 End (Max Level) 日期
        # 向量化取得每個 Group 的 Start (Min Level) 與 End (Max Level) 日期
        
        # 1. 計算該 Group 內的 Row Index (第幾筆交易) - Vectorized
        df_process['group_row_idx'] = df_process.groupby('group_id').cumcount()
        
        # 2. 取得每個 Group 的 Start (Min Level) 與 End (Max Level) 的基準資訊
        # [Optimize] Use direct indexing instead of expensive merge
        g = df_process.groupby('group_id')['disposal_level']
        
        # idxmin/idxmax returns the index of the first occurrence of min/max
        idx_min = g.idxmin()
        idx_max = g.idxmax()
        
        # 取得 Start 基準：Min Level 發生的那一天的 trading_idx (作為基準 0)
        # Note: trading_idx is global. group_row_idx is local (0..N).
        # We define t_label based on offset from Event Start/End.
        # Event Start is at the Min Level row.
        # Event End is at the Max Level row.
        
        # Extract trading_idx and trading_idx_end using direct mapping
        # map: group_id -> trading_idx_start (of start event)
        start_t_idx_map = df_process.loc[idx_min, ['group_id', 'trading_idx_start']].set_index('group_id')['trading_idx_start']
        # End event reference: we need the trading_idx_end of the LAST event (Max Level)
        end_t_idx_map = df_process.loc[idx_max, ['group_id', 'trading_idx_end']].set_index('group_id')['trading_idx_end']
        
        # Map back to main DF - vectorized O(N)
        df_process['base_start_t_idx'] = df_process['group_id'].map(start_t_idx_map)
        df_process['base_end_t_idx'] = df_process['group_id'].map(end_t_idx_map)

        # Retrieve base_start_date and base_end_date
        start_date_map = df_process.loc[idx_min, ['group_id', 'event_start_date']].set_index('group_id')['event_start_date']
        end_date_map = df_process.loc[idx_max, ['group_id', 'event_end_date']].set_index('group_id')['event_end_date']
        
        df_process['base_start_date'] = df_process['group_id'].map(start_date_map)
        df_process['base_end_date'] = df_process['group_id'].map(end_date_map)
        
        # 3. Vectorized Label Generation
        df_process['t_label'] = None
        
        curr_t_idx = df_process['trading_idx'].values
        base_start = df_process['base_start_t_idx'].values
        base_end = df_process['base_end_t_idx'].values
        
        # e+N: Current >= End Index
        mask_e = (curr_t_idx >= base_end)
        
        # Allocate result array
        labels = np.full(len(df_process), None, dtype=object)
        
        if mask_e.any():
            diff_e = (curr_t_idx[mask_e] - base_end[mask_e]).astype(int)
            labels[mask_e] = 'e+' + diff_e.astype(str)
            
        # s+N / s-N: Not e
        mask_s = (~mask_e) & (~np.isnan(base_start))
        if mask_s.any():
            diff_s = (curr_t_idx[mask_s] - base_start[mask_s]).astype(int)
            # vectorized string formatting
            s_vals = diff_s
            pos_mask = s_vals >= 0
            neg_mask = s_vals < 0
            
            # Fill logic
            # Sub-masking relative to original array involves boolean indexing carefully
            # Let's perform assignment on the subset directly? No, hard with mixed masks.
            # Use temporary arrays for the subset
            
            subset_labels = np.empty(len(diff_s), dtype=object)
            
            if pos_mask.any():
                subset_labels[pos_mask] = 's+' + s_vals[pos_mask].astype(str)
            if neg_mask.any():
                subset_labels[neg_mask] = 's' + s_vals[neg_mask].astype(str)
                
            labels[mask_s] = subset_labels

        df_process['t_label'] = labels

        # Debug: Keep intermediate columns
        # final_df = df_process.drop(columns=[
        #     'prev_stock', 'prev_level', 'new_group', 'group_id',
        #     'group_row_idx', 'base_start_t_idx', 'base_end_t_idx',
        #     'prev_trading_idx', 'prev_start_date'
        # ])
        final_df = df_process

        final_df.to_csv('test.csv', index=False)
        return final_df

    def plot_trend_return(self, df: pd.DataFrame, session: str = 'position'):
        """
        繪製不同趨勢下的每日平均報酬 (Daily Return)
        """
        # 決定分組欄位：優先使用合併後的 base_start_date，其次用 event_start_date
        if 'base_start_date' not in df.columns:
            print("少了 base_start_date")
        if 'base_end_date' not in df.columns:
            print("少了 base_end_date")
        
        # 計算報酬
        df = self._compute_returns(df, session)
        
        # 移除暫存欄位
        if 'prev_close' in df.columns:
            df.drop(columns=['prev_close'], inplace=True)

        target_col = 't_label'
        if target_col not in df.columns:
             target_col = 't_label_first'

        if target_col not in df.columns:
            print(f"缺少 {target_col}，無法繪圖")
            return

        trend_stats = df.groupby(['direction', target_col])['daily_ret'].agg(['mean', 'std', 'count']).reset_index()
        
        # 定義要繪製的子集配置: (標籤, 篩選函數)
        plot_configs = [('All', lambda d: pd.Series(True, index=d.index))]
        
        if 'interval' in df.columns:
            plot_configs.append(('5min', lambda d: d['interval'].astype(str).str.contains('5')))
            plot_configs.append(('20min', lambda d: d['interval'].astype(str).str.contains('20')))

        # 針對每個配置繪圖
        for config_name, filter_func in plot_configs:
            # 根據配置篩選資料
            try:
                mask = filter_func(df)
                sub_df = df[mask].copy()
            except Exception as e:
                print(f"Error filtering for {config_name}: {e}")
                continue
                
            if sub_df.empty:
                print(f"No data for {config_name} interval.")
                continue

            # 重新計算該子集的統計數據
            sub_stats = sub_df.groupby(['direction', target_col])['daily_ret'].agg(['mean', 'std', 'count']).reset_index()

            # 分別繪圖 (Overbought / Oversold)
            for direction in ['Overbought', 'Oversold']:
                # 篩選該方向數據
                stats = sub_stats[sub_stats['direction'] == direction].copy()
                if stats.empty:
                    print(f"No data for {direction} ({config_name})")
                    continue
                
                # 排序方便閱讀
                stats['sort_key'] = stats[target_col].apply(self._parse_t_val)
                stats = stats.sort_values('sort_key').drop(columns=['sort_key'])
                    
                # 繪圖
                self._plot_stats(
                    stats, 
                    target_col=target_col,
                    note=f'{direction} ({config_name}) - Daily Return - {session}'
                )
        
                stats.to_csv(f'test_{direction}.csv', index=False)


    def _prepare_surface_data(self, df: pd.DataFrame, session: str, bins: int):
        """
        準備 3D/2D 繪圖所需的聚合數據
        回傳:
            grid_df: 聚合後的 DataFrame, 包含 relative_day_idx, ind_factor_mid, mean, std, count
            x_tickvals: X軸刻度值 (連續索引)
            x_ticktext: X軸刻度標籤 (s+N / e+N)
        """
        # 1. 準備數據
        # 確保數據按時間排序，否則 shift 計算會錯誤
        if 'Date' in df.columns:
            df = df.sort_values(['Stock_id', 'Date'])
        
        df = self._compute_returns(df, session)
        
        # 動態建立分組欄位，避免 KeyError
        group_cols = ['Stock_id']
        for col in ['base_start_date', 'base_end_date']:
            if col in df.columns:
                group_cols.append(col)
        
        # 計算產業因子
        if session == 'position':
            df['close_-5'] = df.groupby(group_cols)['Ind_Close'].shift(5)
            df['close_-1'] = df.groupby(group_cols)['Ind_Close'].shift(1)
            df['ind_factor'] = (df['close_-1']/df['close_-5']) - 1
        elif session == 'after_market':
            df['close_-1'] = df.groupby(group_cols)['Ind_Close'].shift(1)
            df['ind_factor'] = (df['Ind_Close'] / df['close_-1']) - 1

        # 準備 X 軸: Relative Day
        # 優先嘗試從 t_label (或 t_label_first) 還原正確的 s/e 數值結構
        # 因為 CSV 中的 relative_day 可能只是連續天數，沒有區分 e 系列
        if 't_label' not in df.columns:
             # 嘗試尋找替代欄位
             target_col = next((c for c in ['t_label_first', 't_label_second', 't_label_third', 't_label_fourth'] if c in df.columns), None)
             if target_col:
                 df['t_label'] = df[target_col]
        
        if 't_label' in df.columns:
             # 強制重算，以確保 e 系列的 1000+ 特性被保留
             df['relative_day'] = df['t_label'].apply(self._parse_t_val)
        elif 'relative_day' not in df.columns:
             print("缺少 relative_day 或 t_label，無法定位時間軸。")
             return None, None, None
        
        # 移除無法解析的天數
        # 999.0 為 _parse_t_val 解析失敗的代碼，需排除
        df = df[(df['relative_day'] < 2000) & (df['relative_day'] != 999.0)]

        # 2. 準備 Y 軸: Industry Return Binning
        q_low = df['ind_factor'].quantile(0.01)
        q_high = df['ind_factor'].quantile(0.99)
        df_filtered = df[(df['ind_factor'] >= q_low) & (df['ind_factor'] <= q_high)].copy()
        
        try:
             df_filtered['ind_factor_bin'] = pd.cut(df_filtered['ind_factor'], bins=bins)
             df_filtered['ind_factor_mid'] = df_filtered['ind_factor_bin'].apply(lambda x: x.mid).astype(float)
        except Exception as e:
            print(f"分組失敗: {e}")
            return None, None, None

        # 3. 計算統計量 (Aggregation)
        grid_df = df_filtered.groupby(['relative_day', 'ind_factor_mid'])['daily_ret'].agg(['mean', 'std', 'count']).reset_index()
        
        if grid_df.empty:
            print("無數據可繪圖")
            return None, None, None

        # 4. X軸標籤映射 (連續索引)
        try:
             # Get unique relative days sorted
             unique_days = np.sort(df_filtered['relative_day'].unique())
             
             # Create a mapping from relative_day to Ordinal Index (0, 1, 2...)
             day_map = {val: i for i, val in enumerate(unique_days)}
             
             # Apply mapping to grid_df
             grid_df['relative_day_idx'] = grid_df['relative_day'].map(day_map)
             
             # Generate Ticks
             def format_label(val):
                 val = int(val)
                 if val >= 1000:
                     return f"e{val - 1000:+d}"
                 else:
                     return f"s{val:+d}"
             
             x_tickvals = np.arange(len(unique_days))
             x_ticktext = [format_label(val) for val in unique_days]
             
             return grid_df, x_tickvals, x_ticktext

        except Exception as e:
            print(f"X軸標籤映射失敗: {e}")
            return None, None, None

    def plot_3d_return_surface(self, df: pd.DataFrame, session: str = 'position', bins: int = 20, split_by_direction: bool = True, use_browser: bool = True, show_metrics: list = None):
        """
        繪製 3D 表面圖 (Mean, Std, Count)
        :param split_by_direction: 若 True 且 df 中包含 'direction' 欄位，則自動分開繪圖
        :param use_browser: 若 True，使用外部瀏覽器開啟圖表 (避免 IDE WebGL 崩潰)
        :param show_metrics: 指定要顯示的指標列表，預設為 ['mean']。可選值: 'mean', 'std', 'count'
        """
        # 預設僅顯示平均值，避免一次彈出太多視窗
        if show_metrics is None:
            show_metrics = ['mean']
        elif isinstance(show_metrics, str):
            show_metrics = [show_metrics]

        # 自動分組邏輯
        if split_by_direction and 'direction' in df.columns:
            directions = df['direction'].unique()
            if len(directions) > 1:
                print(f"\n[Auto Split] 檢測到多種 Direction: {directions}，將分開繪圖...")
                df = df.loc[df['direction'] != 'unknown']
                for d in directions:
                    sub_df = df[df['direction'] == d].copy()
                    print(f"\n>>> Plotting for Direction: {d}")
                    # 遞迴呼叫，傳遞所有參數
                    self.plot_3d_return_surface(sub_df, session, bins, split_by_direction=False, use_browser=use_browser, show_metrics=show_metrics)
                return

        print(f"\n[3D Surface Analysis] Session: {session}")
        
        # 嘗試取得目前的 direction 名稱以標註在標題
        current_direction = ""
        if 'direction' in df.columns and len(df['direction'].unique()) == 1:
            current_direction = f"[{df['direction'].iloc[0]}] "

        grid_df, x_tickvals, x_ticktext = self._prepare_surface_data(df, session, bins)
        
        if grid_df is None:
            return

        metrics_map = {'mean': 'Mean Return', 'std': 'Standard Deviation', 'count': 'Sample Count'}
        colorscales = {'mean': 'Viridis', 'std': 'Plasma', 'count': 'Blues'}

        for metric in show_metrics:
            if metric not in metrics_map:
                print(f"Unknown metric: {metric}, skipping.")
                continue
                
            title = metrics_map[metric]
            
            # Pivot using the Ordinal Index instead of numerical relative_day
            pivot_matrix = grid_df.pivot(index='ind_factor_mid', columns='relative_day_idx', values=metric)
            # Interpolate for smoother surface
            pivot_matrix = pivot_matrix.interpolate(axis=0).interpolate(axis=1)
            
            x_data = pivot_matrix.columns.values
            y_data = pivot_matrix.index.values
            z_data = pivot_matrix.values
            
            fig = go.Figure(data=[go.Surface(
                z=z_data, 
                x=x_data, 
                y=y_data,
                colorscale=colorscales.get(metric, 'Viridis'),
                colorbar=dict(title=title),
                connectgaps=True 
            )])


            scene_dict=dict(
                xaxis_title='t_label',
                yaxis_title='Industry Return',
                zaxis_title=title,
                aspectratio=dict(x=1, y=1, z=0.6) 
            )

            # Apply custom axis labels if available
            if x_tickvals is not None:
                scene_dict['xaxis'] = dict(
                    tickvals=x_tickvals,
                    ticktext=x_ticktext,
                    title='Relative Day'
                )

            fig.update_layout(
                title=f'{current_direction}3D Surface (Plateau): {title} ({session})',
                scene=scene_dict,
                width=1000,
                height=800,
                annotations=[
                    dict(
                        text="X: t_label<br>Y: Industry Return<br>Z: " + title,
                        x=0, y=1, 
                        xref="paper", yref="paper",
                        showarrow=False,
                        align="left",
                        bgcolor="white",
                        opacity=0.8,
                        bordercolor="black",
                        borderwidth=1
                    )
                ]
            )
            
            if use_browser:
                try:
                    fig.show(renderer='browser')
                except:
                    print("無法呼叫瀏覽器，改用預設顯示。")
                    fig.show()
            else:
                fig.show()

    def plot_2d_slice(self, df: pd.DataFrame, slice_by: str, target, session: str = 'position', bins: int = 20, split_by_direction: bool = True):
        """
        繪製 2D 切片圖
        :param slice_by: 'ind_factor' (固定產業報酬，看時間走勢) 或 'time' (固定時間，看產業報酬影響)
        :param target: 切片目標值。若 slice_by='ind_factor' 則為 float (e.g. 0.05); 若 slice_by='time' 則為 str (e.g. 's+5') 或數值
        :param split_by_direction: 若 True 且 df 中包含 'direction' 欄位，則自動分開繪圖
        """
        # 自動分組邏輯
        if split_by_direction and 'direction' in df.columns:
            directions = df['direction'].unique()
            if len(directions) > 1:
                print(f"\n[Auto Split] 檢測到多種 Direction: {directions}，將分開繪圖...")
                for d in directions:
                    sub_df = df[df['direction'] == d].copy()
                    print(f"\n>>> Plotting for Direction: {d}")
                    self.plot_2d_slice(sub_df, slice_by, target, session, bins, split_by_direction=False)
                return

        print(f"\n[2D Slice Analysis] Session: {session}, Slice By: {slice_by}, Target: {target}")
        
        # 嘗試取得目前的 direction 名稱
        current_direction = ""
        if 'direction' in df.columns and len(df['direction'].unique()) == 1:
            current_direction = f"[{df['direction'].iloc[0]}] "
        
        grid_df, x_tickvals, x_ticktext = self._prepare_surface_data(df, session, bins)
        
        if grid_df is None:
            return

        fig = go.Figure()
        
        if slice_by == 'ind_factor':
            # Case A: 固定產業報酬，觀察時間走勢 (X=Time, Y=Return)
            
            # 確保 target 是 float
            try:
                target_val = float(target)
            except:
                print(f"Invalid target for ind_factor: {target}")
                return

            # 找到最近的 Bin
            unique_bins = np.sort(grid_df['ind_factor_mid'].unique())
            closest_idx = np.abs(unique_bins - target_val).argmin()
            closest_bin_val = unique_bins[closest_idx]
            
            print(f"Selected closest industry return bin: {closest_bin_val:.2%}")
            
            # 篩選
            slice_df = grid_df[grid_df['ind_factor_mid'] == closest_bin_val].sort_values('relative_day_idx')
            
            # 繪圖
            fig.add_trace(go.Scatter(
                x=slice_df['relative_day_idx'],
                y=slice_df['mean'],
                mode='lines+markers',
                name=f'Ind Ret ~ {closest_bin_val:.2%}',
                error_y=dict(type='data', array=slice_df['std'], visible=True)
            ))
            
            fig.update_layout(
                title=f'{current_direction}Daily Return Slice @ Industry Return ~ {closest_bin_val:.2%} ({session})',
                xaxis=dict(
                    tickvals=x_tickvals,
                    ticktext=x_ticktext,
                    title='Relative Day'
                ),
                yaxis=dict(title='Daily Return')
            )

        elif slice_by == 'time':
            # Case B: 固定時間，觀察產業報酬影響 (X=Ind Ret, Y=Return)
            
            # 解析 target 時間
            if isinstance(target, str):
                target_val = self._parse_t_val(target)
            else:
                target_val = float(target)
                
            if target_val == 999.0:
                 print(f"Invalid time target: {target}")
                 return

            # 找到最近的時間點
            # 注意 grid_df['relative_day'] 是數值 (包含 1000+ 的 e 系列)
            unique_days = np.sort(grid_df['relative_day'].unique())
            closest_idx = np.abs(unique_days - target_val).argmin()
            closest_day_val = unique_days[closest_idx]
            
            # 還原顯示標籤 (for title)
            if closest_day_val >= 1000:
                day_str = f"e{int(closest_day_val)-1000:+d}"
            else:
                day_str = f"s{int(closest_day_val):+d}"
                
            print(f"Selected closest relative day: {day_str} (val={closest_day_val})")
            
            # 篩選
            slice_df = grid_df[grid_df['relative_day'] == closest_day_val].sort_values('ind_factor_mid')
            
            # 繪圖
            fig.add_trace(go.Scatter(
                x=slice_df['ind_factor_mid'],
                y=slice_df['mean'],
                mode='lines+markers',
                name=f'Time @ {day_str}',
                error_y=dict(type='data', array=slice_df['std'], visible=True)
            ))
            
            fig.update_layout(
                title=f'{current_direction}Daily Return Slice @ Time {day_str} ({session})',
                xaxis=dict(
                    title='Industry Return',
                    tickformat='.1%' 
                ),
                yaxis=dict(title='Daily Return')
            )
            
        else:
            print(f"Unknown slice_by mode: {slice_by}")
            return

        fig.update_layout(width=1000, height=600)
        fig.show()


# Backward compatibility
def run_multi_level_analysis(df: pd.DataFrame):
    analyzer = DisposalAnalyzer(df)
    analyzer.overall_analysis()