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

        # 3. 針對不同方向計算每日平均報酬並繪圖 (同標的連續處置合併)
        print("\n[每日報酬率分析 (Daily Return Analysis) - Consecutive Merged]")

        if 't_label_first' not in self.df.columns:
            print("缺少 t_label_first 欄位，無法定位處置起點。")
            return
            
        # 建立臨時 Index 用於計算相對天數
        self.df['__tmpidx'] = range(len(self.df))
        
        # 1. 識別連續處置區塊 (Chains)
        # [修正] 使用日期區間來判定是否真正處於「處置中」
        # disposal_level 只是事件編號，不能用來過濾日期
        
        # 確保日期格式正確 (與 pandas 相容)
        for col in ['Date', 'event_start_date', 'event_end_date']:
            if col in self.df.columns and not pd.api.types.is_datetime64_any_dtype(self.df[col]):
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')

        # 條件 A: 當天日期落在處置起始與結束日期之間 (Active Disposal)
        if 'event_start_date' in self.df.columns and 'event_end_date' in self.df.columns and 'Date' in self.df.columns:
            cond_in_disposal = (self.df['Date'] >= self.df['event_start_date']) & (self.df['Date'] <= self.df['event_end_date'])
        else:
            # Fallback (不太可能發生，除非 csv 欄位不對)
            print("Warning: Missing date columns for active disposal check. using disposal_level fallback.")
            cond_in_disposal = (self.df['disposal_level'].notna()) & (self.df['disposal_level'] > 0)

        # 條件 B: 是第一次處置的 s-2, s-1 (前置觀察期)
        cond_pre_days = self.df['t_label_first'].isin(['s-2', 's-1'])
            
        is_event = cond_in_disposal | cond_pre_days
        
        # 偵測斷點：Stock 不同 OR 是否為處置日的狀態改變
        group_change = (is_event != is_event.shift(1)) | (self.df['Stock_id'] != self.df['Stock_id'].shift(1))
        self.df['__group_id'] = group_change.cumsum()
        
        # 只取處置區塊的資料
        valid_mask = is_event
        # 複製出來處理，避免影響原始 df
        cols_to_keep = ['Stock_id', 'Date', 'disposal_level', 'direction', 'daily_ret', 
                        't_label_first', '__group_id', 'event_start_date']
        valid_rows = self.df.loc[valid_mask, cols_to_keep].copy()
        
        # [新增] 去除重複日期 (Deduplicate)
        # 因為 L1 和 L2 的時間可能重疊，導致同一天有兩筆資料 (一筆 L1, 一筆 L2)
        # 我們只保留一筆 (keep='first' 隨著原始排序通常保留 L1 或較早開始的事件)
        valid_rows = valid_rows.sort_values(by=['Stock_id', 'Date', 'disposal_level']) # 確保排序穩定
        valid_rows = valid_rows.drop_duplicates(subset=['Stock_id', 'Date'], keep='first')

        # 準備欄位
        valid_rows['unified_t_label'] = None
        valid_rows['unified_direction'] = None
        
        # 以 Group 為單位進行處理
        results = []
        
        for gid, group_df in valid_rows.groupby('__group_id'):
            # 必須重新因為 Date 排序，確保時間軸正確
            group_df = group_df.sort_values('Date')
            
            # 定位 Anchor (L1 s+0)
            # 優先找 t_label_first == 's+0'
            anchor_rows = group_df[group_df['t_label_first'] == 's+0']
            
            if not anchor_rows.empty:
                anchor_row = anchor_rows.iloc[0]
                # 取得該 Group 統一的趨勢方向
                main_direction = anchor_row['direction']
                
                # 計算時間軸
                anchor_idx = group_df.index.get_loc(anchor_row.name) # get integer location
                
                # Assign distinct values
                group_df['unified_direction'] = main_direction
                
                # 建立相對天數 (t_int)
                n_rows = len(group_df)
                t_ints = range(n_rows)
                rel_t = [t - anchor_idx for t in t_ints]
                
                # 轉成 t_label 字串
                def to_label(val):
                    if val == 0: return 's+0'
                    if val < 0: return f"s{val}"
                    return f"s+{val}"
                
                group_df['unified_t_label'] = [to_label(x) for x in rel_t]
                
                results.append(group_df)
            else:
                continue
                
        if results:
            final_df = pd.concat(results)
            
            # 繪圖
            _plot_cols = ['daily_ret']
            direction_order = ['Overbought', 'Oversold']
            
            # 聚合計算 Mean & Count
            # Group by (Time, Direction)
            stats = final_df.groupby(['unified_t_label', 'unified_direction'])['daily_ret'].agg(['mean', 'count', 'std']).reset_index()
            
            # 8. 繪圖
            # 由於 _plot_stats 只支援單一序列繪圖 (Hardcoded ly='mean')，
            # 我們這裡透過迴圈分別畫出 Overbought 和 Oversold 的圖表，而不是畫在同一張圖上
            print(f"\n[繪圖] 產出 {len(direction_order)}張圖表 (Overbought/Oversold)...")
            
            for direction in direction_order:
                # 篩選對應方向的數據
                subset = stats[stats['unified_direction'] == direction].copy()
                
                if subset.empty:
                    print(f"Skipping {direction}: No data.")
                    continue
                
                print(f" -> Plotting {direction} (Data points: {len(subset)})")
                
                # 呼叫既有的 _plot_stats (它會自動排序並畫 mean/count)
                self._plot_stats(
                    stats=subset, 
                    target_col='unified_t_label', 
                    note=f"Daily Return Trajectory - {direction} (Consecutive Merged)"
                )
            
        else:
            print("No valid consecutive disposal groups found (with s+0 anchor).")
            
        # 清理暫存
        if '__tmpidx' in self.df.columns:
            del self.df['__tmpidx']
        if '__group_id' in self.df.columns:
            del self.df['__group_id']
        
        results.to_csv('test.csv', index=False)

# Backward compatibility
def run_multi_level_analysis(df: pd.DataFrame):
    analyzer = DisposalAnalyzer(df)
    analyzer.overall_analysis()