import pandas as pd
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
    def _parse_t_val(t_str: str) -> int:
        """解析時間標籤 (e.g., 's+1', 'e-5') 為數值以利排序"""
        if not isinstance(t_str, str): return 999
        prefix = t_str[0]
        try:
            val = int(t_str.split('+')[-1]) if '+' in t_str else int(t_str.split('s')[-1]) 
            if 's-' in t_str: val = -int(t_str.split('-')[-1])
            if prefix == 's': return val
            elif prefix == 'e': return 1000 + val
        except:
            return 999
        return 999

    def _plot_stats(self, stats: pd.DataFrame, target_col: str, note: str):
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
            note=note,
            bar_kwargs={'width': 0.8}
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
        比較處置原因：利用 s-2 至 s+0 的累積漲跌幅區分超漲 (Overbought) / 超跌 (Oversold)
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
            
            # 取出該層級 s-2 到 s+0 的資料
            mask = (self.df['disposal_level'] == lvl) & (self.df[t_col].isin(['s-2', 's-1', 's+0']))
            subset = self.df.loc[mask, ['Stock_id', 'event_start_date', 'daily_ret']].copy()
            
            if subset.empty: continue

            # 計算這段期間的累積報酬 (簡單加總 daily_ret)
            grouped = subset.groupby(['Stock_id', 'event_start_date'])['daily_ret'].sum().reset_index()
            grouped['direction'] = grouped['daily_ret'].apply(
                lambda x: 'Overbought' if x > 0 else 'Oversold'
            )
            event_trends.append(grouped[['Stock_id', 'event_start_date', 'direction']])
            
        if not event_trends:
            print("找不到 s-2 ~ s+0 的時間標籤資料")
            return
            
        # 合併所有事件的判斷結果
        all_trends = pd.concat(event_trends, ignore_index=True)
        
        # 將判斷結果 Merge 回主表
        if 'direction' in self.df.columns:
            self.df = self.df.drop(columns=['direction'])
            
        self.df = self.df.merge(all_trends, on=['Stock_id', 'event_start_date'], how='left')
        self.df['direction'] = self.df['direction'].fillna('Unknown')
        
        # 1. 顯示基本分佈
        self._display_dataframe('direction', '處置前趨勢分佈 (s-2 ~ s+0)')

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
        df_process['prev_end'] = df_process['event_end_date'].shift(1)
        
        cond_stock_change = df_process['Stock_id'] != df_process['prev_stock']
        is_level_1 = df_process['disposal_level'] == 1
        prev_not_1 = df_process['prev_level'] != 1
        end_date_diff = df_process['event_end_date'] != df_process['prev_end']
        cond_new_event = is_level_1 & (prev_not_1 | end_date_diff)
        
        df_process['new_group'] = cond_stock_change | cond_new_event
        df_process['group_id'] = df_process['new_group'].cumsum()
        
        # 取得每個 Group 的 Start (Min Level) 與 End (Max Level) 日期
        def get_dates(x):
            start_date = x.loc[x['disposal_level'].idxmin(), 'event_start_date']
            end_date = x.loc[x['disposal_level'].idxmax(), 'event_end_date']
            return pd.Series({'base_start_date': start_date, 'base_end_date': end_date})

        group_dates = df_process.groupby('group_id').apply(get_dates, include_groups=False)
        df_process = df_process.merge(group_dates, on='group_id', how='left')
                
        # 1. 計算該 Group 內的 Row Index (第幾筆交易)
        df_process['group_row_idx'] = df_process.groupby('group_id').cumcount()
        
        # 2. 找出 Base Start Date 與 Base End Date 在該 Group 對應的 Index
        df_process['base_start_date'] = pd.to_datetime(df_process['base_start_date'])
        df_process['base_end_date'] = pd.to_datetime(df_process['base_end_date'])
        
        # 找出 Start Index
        start_indices = df_process[df_process['Date'] == df_process['base_start_date']][['group_id', 'group_row_idx']]
        start_indices = start_indices.rename(columns={'group_row_idx': 'base_start_idx'})
        # 避免重複 (理論上同 Group 同 Date 已經由 drop_duplicates 濾除，但保險起見)
        start_indices = start_indices.drop_duplicates(subset=['group_id'])
        
        # 找出 End Index
        end_indices = df_process[df_process['Date'] == df_process['base_end_date']][['group_id', 'group_row_idx']]
        end_indices = end_indices.rename(columns={'group_row_idx': 'base_end_idx'})
        end_indices = end_indices.drop_duplicates(subset=['group_id'])
        
        # Merge 指標回主表
        df_process = df_process.merge(start_indices, on='group_id', how='left')
        df_process = df_process.merge(end_indices, on='group_id', how='left')
        
        # 3. 計算 Diff 並產生 Label
        def format_t_label_trading_days(row):
            # 'e' 系列 (>= End Index)
            if pd.notna(row['base_end_idx']):
                if row['group_row_idx'] >= row['base_end_idx']:
                    diff = int(row['group_row_idx'] - row['base_end_idx'])
                    return f"e+{diff}"
            
            # 計算 's' 系列
            if pd.notna(row['base_start_idx']):
                diff = int(row['group_row_idx'] - row['base_start_idx'])
                return f"s{'+' if diff >= 0 else ''}{diff}"
                
            return None

        df_process['t_label'] = df_process.apply(format_t_label_trading_days, axis=1)
        
        final_df = df_process.drop(columns=[
            'prev_stock', 'prev_level', 'prev_end', 'new_group', 'group_id',
            'group_row_idx', 'base_start_idx', 'base_end_idx'
        ])
        final_df.to_csv('test.csv')
        
        return final_df

    def plot_trend_return(self, df: pd.DataFrame, session: str = 'position'):
        """
        繪製不同趨勢下的每日平均報酬 (Daily Return)
        """
        if session == 'position':
            pass
        elif session == 'after_market':
            print("跟台指期一樣，夜盤在日盤前面")
            df['daily_ret'] = (df['Open'] / df['Close'].shift(1)) - 1
        elif session == 'all':
            print("跟台指期一樣，夜盤在日盤前面")
            df['daily_ret'] = (df['Close'] / df['Close'].shift(1)) - 1

        target_col = 't_label'
        if target_col not in df.columns:
             target_col = 't_label_first'

        if target_col not in df.columns:
            print(f"缺少 {target_col}，無法繪圖")
            return

        trend_stats = df.groupby(['direction', target_col])['daily_ret'].agg(['mean', 'count']).reset_index()
        
        # 分別繪圖
        for direction in ['Overbought', 'Oversold']:
            # 篩選該方向數據
            stats = trend_stats[trend_stats['direction'] == direction].copy()
            if stats.empty:
                print(f"No data for {direction}")
                continue
                
            # 繪圖
            self._plot_stats(
                stats, 
                target_col=target_col, 
                note=f'{direction} - Daily Return - {session}'
            )


# Backward compatibility
def run_multi_level_analysis(df: pd.DataFrame):
    analyzer = DisposalAnalyzer(df)
    analyzer.overall_analysis()