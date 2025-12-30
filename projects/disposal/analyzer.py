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
            
            # Combine
            stats = pd.concat([days_counts, event_counts], axis=1).fillna(0).astype(int)
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
        display_cols = [col, 'days_count', 'event_count', 'days_pct']
        if 'event_count' in stats.columns and stats['event_count'].dtype != 'O':
             # Format numbers
             display(stats[display_cols].style.format({
                 'days_count': '{:,}',
                 'event_count': '{:,}'
             }))
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

# Backward compatibility
def run_multi_level_analysis(df: pd.DataFrame):
    analyzer = DisposalAnalyzer(df)
    analyzer.overall_analysis()