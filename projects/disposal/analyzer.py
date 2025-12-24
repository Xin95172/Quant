import pandas as pd
from module.plot_func import plot
from typing import Optional, List

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

    def overall_analysis(self, min_samples: int = 50, prefix: str = 't_label_'):
        """
        [Method 1] 多層級處置分析
        計算各時間點 (t_label_*) 的平均日報酬率並繪圖。
        """
        if self.df.empty:
            print("Dataframe is empty. Please check data source.")
            return

        print("Condition Counts:")
        print(self.df['condition'].value_counts())
        print("-" * 30)

        t_cols = [c for c in self.df.columns if c.startswith(prefix)]
        print(f"Found {len(t_cols)} levels with prefix '{prefix}'")

        for target_col in t_cols:
            # Check sample size
            valid_count = self.df[target_col].notna().sum()
            if valid_count < min_samples:
                continue

            # Statistics
            stats = self.df.groupby(target_col)['daily_ret'].agg(['mean', 'count', 'std']).reset_index()
            
            # Plot
            self._plot_stats(stats, target_col, note=f"Disposal Effect: {target_col}")

# Backward compatibility
def run_multi_level_analysis(df: pd.DataFrame):
    analyzer = DisposalAnalyzer(df)
    analyzer.overall_analysis()
