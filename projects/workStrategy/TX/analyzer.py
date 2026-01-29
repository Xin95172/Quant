import pandas as pd
import numpy as np
import module.plot_func as plot
import plotly.graph_objects as go
from IPython.display import display

class TXAnalyzer:
    def __init__(self, df: pd.DataFrame):
        df_day = df[df['trading_session'] == 'position'].copy()
        df_night = df[df['trading_session'] == 'after_market'].copy()
            
        df_night = df_night.add_suffix('_a')
            
        self.df = pd.concat([df_day, df_night], axis=1)
    
        self.df['daily_ret'] = (self.df['Close'] / self.df['Open']) - 1
        self.df['cum_daily_ret'] = self.df['daily_ret'].cumsum()
        self.df['daily_pnl'] = self.df['Close'] - self.df['Open']
        self.df['cum_daily_pnl'] = self.df['daily_pnl'].cumsum()
        
        self.df['daily_ret_a'] = (self.df['Close_a'] / self.df['Open_a']) - 1
        self.df['cum_daily_ret_a'] = self.df['daily_ret_a'].cumsum()
        self.df['daily_pnl_a'] = self.df['Close_a'] - self.df['Open_a']
        self.df['cum_daily_pnl_a'] = self.df['daily_pnl_a'].cumsum()
    
    def _get_statistics(self, ret_col: pd.Series):
        return ret_col.describe()
    
    def display_df(self):
        return self.df
    
    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to calculate all strategy factors."""
        # --- A. 宏觀債券 (Macro - The Risk Radar) ---
        # 1. 估值殺手 (Valuation): 20天長債變動
        df['yield_shock'] = df['US_bond_5y'] - df['US_bond_5y'].shift(20)

        # 防未來數據 (Macro指標通常落後或當天晚上才看得到，保守 shift 2)
        macro_cols = ['yield_shock']
        df[macro_cols] = df[macro_cols].shift(3)

        # --- B. 日曆效應 (Calendar - The Time Decay) ---
        # 計算 Gap (T日 - T-1日的天數)
        df.index = pd.to_datetime(df.index)
        df['prev_date'] = df.index.to_series().shift(1)
        df['holiday'] = (df.index.to_series() - df['prev_date']).dt.days
        df['holiday'] = df['holiday'].shift(-1)

        # --- C. 技術乖離 (Technical - The Entry Trigger) ---
        # 15MA 乖離率
        df['15_ma'] = df['Close'].rolling(window=15).mean()
        df['divergence'] = (df['Close'] / df['15_ma']) - 1
        df['divergence'] = df['divergence'].shift(1)
        
        return df

    def _apply_signals_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        df['pos_night'] = 1.0
        df['pos_day'] = 1.0

        # --- 1. 定義輔助 Mask (條件) ---
        # A. 宏觀條件
        severe_shock = df['yield_shock'] > 0.3    # 重大衝擊
        mild_shock = df['yield_shock'] > 0.15   # 輕微衝擊
        safe_zone = df['yield_shock'] < 0.18   # 宏觀安全

        # B. 技術條件
        strong_tech = df['divergence'] > 0.004    # 強勢

        # C. 節日條件 (Gap)
        # holiday 用 yield_shock 排序
        is_abnormal = (df['yield_shock'] < -0.2) | (df['yield_shock'] > 0.35)

        # --- 2. 執行風控層 (Layers) ---
        
        # Layer 2: 宏觀風控 (Macro)
        df.loc[mild_shock, ['pos_night', 'pos_day']] = 0.0

        # Layer 3: 節日風控 (Holiday)
        # 3.1 放假前一天：
        df.loc[(df['holiday'].shift(1) > 2) & is_abnormal, 'pos_night'] = 0.0
        # 3.2 放假後第一天：
        df.loc[(df['holiday'].shift(1) > 2) & ~safe_zone, 'pos_day'] = 0.0

        # --- 3. 執行進場層 (Entries) ---

        # Layer 4: 技術進場 (Technical)
        # 4.1 夜盤進場 (無重大衝擊 + 技術強)
        df.loc[~severe_shock & strong_tech, 'pos_night'] = 1.0
        
        # Layer 5: 特別例外 (Overrides)
        # 5.1 日盤離場：原本乖離不夠要砍單...
        df.loc[~strong_tech, 'pos_day'] = 0.0
        
        # 5.2 聰明接刀 (Smart Re-entry):
        # 但如果是「剛收假」且「宏觀很安全」，強制買回來！(覆蓋掉上面的離場訊號)
        df.loc[(df['holiday'].shift(1) > 2) & safe_zone, 'pos_day'] = 1.0

        return df

    def _calculate_metrics(self, returns: pd.Series, point_version: bool = False) -> dict:
        """
        Calculate strategy performance metrics including advanced stats.
        returns: Series of daily returns (percentage if point_version=False, points if True)
        """
        if returns.empty:
            return {}

        # 1. Basic Distributions
        total_pnl = returns.sum()
        
        if point_version:
            # Points Mode: Calculation is additive
            cum_ret = returns.cumsum()
            total_ret_val = total_pnl
        else:
            # Percentage Mode: Calculation is compounded
            cum_ret = (1 + returns).cumprod()
            total_ret_val = (cum_ret.iloc[-1] - 1) if not cum_ret.empty else 0.0
        
        # 2. Time Statistics
        trading_days = len(returns)
        ann_factor = 252 
        
        # CAGR (Only for percentage mode)
        if not point_version and trading_days > 0:
            cagr = (1 + total_ret_val) ** (ann_factor / trading_days) - 1
        else:
            cagr = 0.0

        # Volatility (Annualized)
        vol_ann = returns.std() * np.sqrt(ann_factor)

        # Sharpe Ratio (Rf=0)
        sharpe = 0.0
        if vol_ann > 0:
            if point_version:
                # For point version, we use simplified annualized profit over volatility
                ann_pnl = (total_pnl / trading_days) * ann_factor if trading_days > 0 else 0
                sharpe = ann_pnl / vol_ann
            else:
                sharpe = cagr / vol_ann
        
        # 3. Drawdown Statistics
        running_max = cum_ret.cummax()
        if point_version:
            drawdown = cum_ret - running_max # Points down
        else:
            drawdown = (cum_ret / running_max) - 1 # Pct down
            
        max_dd = drawdown.min()
        
        # Max DD Duration (Days)
        is_dd = drawdown < 0
        dd_duration = is_dd.astype(int).groupby((is_dd != is_dd.shift()).cumsum()).cumsum()
        max_dd_duration = dd_duration.max() if not dd_duration.empty else 0

        # 4. Trade Statistics (Win Rate, Odds, etc.)
        # Filter non-zero returns to count "Active Trading Days"
        active_rets = returns[returns != 0]
        n_trades = len(active_rets)
        
        if n_trades > 0:
            wins = active_rets[active_rets > 0]
            losses = active_rets[active_rets < 0]
            
            n_win = len(wins)
            win_rate = n_win / n_trades
            
            avg_win = wins.mean() if not wins.empty else 0.0
            avg_loss = losses.mean() if not losses.empty else 0.0
            
            # Profit Factor
            gross_profit = wins.sum()
            gross_loss = abs(losses.sum())
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            
            # Odds (Avg Win / |Avg Loss|)
            odds = avg_win / abs(avg_loss) if abs(avg_loss) > 0 else 0.0
            
            # Expectancy (Avg Return per trade)
            avg_return = active_rets.mean() # 筆均
            
            # Kelly Criterion
            # W - (1-W)/R  (W: Win Rate, R: Odds)
            if odds > 0:
                kelly = win_rate - (1 - win_rate) / odds
            else:
                kelly = 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
            odds = 0.0
            avg_return = 0.0
            kelly = 0.0

        return {
            'Total Return' if not point_version else 'Total PnL': total_ret_val,
            'CAGR': cagr,
            'Volatility': vol_ann,
            'Sharpe': sharpe,
            'Max Drawdown' if not point_version else 'Max Points DD': max_dd,
            'Max DD Duration': max_dd_duration,
            'Profit Factor': profit_factor,
            'Win Rate': win_rate,
            'Odds': odds,
            'Avg Win': avg_win,
            'Avg Loss': avg_loss,
            'Avg Return (Exp)': avg_return,
            'Kelly': kelly
        }

    def update_df(self, df):
        self.df = df
        return self.df

    def daily_ret(self):
        return plot.plot(self.df, ly=['cum_daily_ret', 'cum_daily_ret_a'])

    def monthly_ret(self, mode: str = 'strategy', point_version: bool = False):
        """
        mode: 'strategy' (default), 'benchmark', 'night', 'day'
        """
        df = self.df.copy()
        
        target_col = 'strat_ret'
        label = "Strategy"
        
        if mode == 'strategy':
            # Ensure strategy returns are calculated
            df = self._calculate_factors(df)
            df = self._apply_signals_logic(df)
            if point_version:
                df['strat_ret'] = (df['daily_pnl_a'] * df['pos_night']) + (df['daily_pnl'] * df['pos_day'])
            else:
                df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
            target_col = 'strat_ret'
            label = "Strategy"
            
        elif mode == 'benchmark':
            if point_version:
                df['benchmark'] = df['daily_pnl_a'] + df['daily_pnl']
            else:
                df['benchmark'] = df['daily_ret_a'] + df['daily_ret']
            target_col = 'benchmark'
            label = "Buy & Hold (Benchmark)"
            
        elif mode == 'night':
            target_col = 'daily_pnl_a' if point_version else 'daily_ret_a'
            label = "Night Session (Raw)"
            
        elif mode == 'day':
            target_col = 'daily_pnl' if point_version else 'daily_ret'
            label = "Day Session (Raw)"
        
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        # Calculate monthly returns (sum for points, compounded for percentages)
        if point_version:
            monthly_df = df.groupby(['year', 'month'])[target_col].sum().reset_index()
            monthly_df['display_val'] = monthly_df[target_col]
            unit = "pts"
        else:
            monthly_df = df.groupby(['year', 'month'])[target_col].apply(lambda x: (1 + x).prod() - 1).reset_index()
            monthly_df['display_val'] = monthly_df[target_col] * 100
            unit = "%"
        
        # Create Pivot Table
        pivot_df = monthly_df.pivot(index='year', columns='month', values='display_val')
        
        # Prepare annotations
        annotations = []
        for y_idx, row in pivot_df.iterrows():
            for x_idx, val in row.items():
                if pd.notna(val):
                    annotations.append(
                        dict(
                            x=x_idx, 
                            y=y_idx, 
                            text=f"{val:.1f}" if point_version else f"{val:.2f}", 
                            showarrow=False, 
                            font=dict(color='black' if abs(val) < (100 if point_version else 5) else 'white')
                        )
                    )
                    
        # Plot Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            colorbar=dict(title=f'PnL ({unit})'),
            zmid=0
        ))
        
        fig.update_layout(
            title=f"Monthly P&L Heatmap ({'Points' if point_version else 'Percentage'}) - {label}",
            xaxis_title='Month',
            yaxis_title='Year',
            yaxis=dict(autorange='reversed'), # Make recent years at bottom
            xaxis=dict(tickmode='linear', tick0=1, dtick=1),
            annotations=annotations
        )
        
        fig.show()
    def indicator_position_ret(self):
        temp_df = self.df.copy()
        temp_df['daily_ret'] = temp_df['daily_ret'].shift(1)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df = temp_df.sort_values(by='daily_ret').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        return plot.plot(temp_df, ly='cum_demeaned_daily_ret_a', x='index', ry = 'daily_ret', sub_ly=['cum_daily_ret_a'])

    def indicator_gap_days(self, after_holiday: bool = False, *, sub_analysis: bool = False):
        temp_df = self.df.copy()
        
        # 1. 先計算所有指標 (在時間序列還沒被打亂前)
        # Macro
        temp_df['yield_shock'] = temp_df['US_bond_5y'] - temp_df['US_bond_5y'].shift(20)
        temp_df['yield_shock'] = temp_df['yield_shock'].shift(3) # Lag for safety
        
        # Technical
        temp_df['15_ma'] = temp_df['Close'].rolling(window=15).mean()
        temp_df['divergence'] = (temp_df['Close'] / temp_df['15_ma']) - 1
        temp_df['divergence'] = temp_df['divergence'].shift(1) # Yesterday's divergence for today's trade

        # Calendar
        temp_df.index = pd.to_datetime(temp_df.index)
        temp_df['prev_date'] = temp_df.index.to_series().shift(1)
        temp_df['gap'] = (temp_df.index.to_series() - temp_df['prev_date']).dt.days
        
        if after_holiday:
            # 想看週一 (Post-holiday)
            # Day: 週一日盤 = Current (No shift)
            # Night: 週一夜盤 = 記在週二 (Next Row) -> shift(-1)
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'].shift(-1)
        else:
            # 想看週五 (Pre-holiday)
            # Day: 週五日盤 = 記在週五 (Prev Row) -> shift(1)
            temp_df['daily_ret'] = temp_df['daily_ret'].shift(1)
            # Night: 週五夜盤 = 記在週一 (Current Row) -> No shift
            pass

        temp_df = temp_df.sort_values(by='gap').reset_index(drop=True)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()

        if sub_analysis:
            # 2. 進行篩選
            # 只看剛放完假的 (例如週一)
            temp_df = temp_df.loc[temp_df['gap'] > 2]
            
            # 過濾掉宏觀高風險 (Yield Shock > 0.3)
            # is_shock = temp_df['yield_shock'] > 0.3
            # temp_df = temp_df.loc[~is_shock]
            
            # 確保有技術指標 (前面 shift 造成前幾筆是 NaN)
            temp_df = temp_df.dropna(subset=['divergence'])
            
            # 依技術面強弱排序，觀察績效
            temp_df = temp_df.sort_values(by='yield_shock').reset_index(drop=True)
            
            # 重算累積報酬 (因為 filter 過了)
            temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
            temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
            temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            
            return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='yield_shock', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'])

        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='gap', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'])

    def indicator_maintenance_rate(self, point_version: bool = False):
        if 'TotalExchangeMarginMaintenance' not in self.df.columns:
            raise ValueError("TotalExchangeMarginMaintenance is not in the DataFrame.")
        temp_df = self.df.copy()
        temp_df['TotalExchangeMarginMaintenance'] = temp_df['TotalExchangeMarginMaintenance'].shift(1)
        if point_version:
            temp_df['daily_ret_a'] = (temp_df['Close_a'] - temp_df['Open_a'])
            temp_df['daily_ret'] = (temp_df['Close'] - temp_df['Open'])
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df = temp_df.sort_values(by='TotalExchangeMarginMaintenance').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()

        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='TotalExchangeMarginMaintenance', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'])

    def indicator_margin_delta(self):
        temp_df = self.df.copy()
        temp_df['margin_delta'] = (temp_df['MarginPurchaseMoney']/temp_df['MarginPurchaseMoney'].shift(1)) - 1
        temp_df['avg_margin_delta'] = temp_df['margin_delta'].shift(1).rolling(window=20).mean()
        temp_df = temp_df.sort_values(by='avg_margin_delta').reset_index(drop=True)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        temp_df['cum_ret'] = (temp_df['daily_ret_a'] + temp_df['daily_ret']).cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='avg_margin_delta', sub_ly=['cum_daily_ret_a', 'cum_daily_ret', 'cum_ret'])

    def indicator_institutional_flow(self):
        temp_df = self.df.copy()
        temp_df['foreign_inflow'] = (temp_df['Net_Foreign_Investor'] - temp_df['Net_Foreign_Investor'].rolling(window=20).mean()) / temp_df['Net_Foreign_Investor'].rolling(window=20).std()
        temp_df['foreign_inflow'] = temp_df['foreign_inflow'].shift(1)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df = temp_df.sort_values(by='foreign_inflow').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        temp_df['cum_ret'] = (temp_df['daily_ret_a'] + temp_df['daily_ret']).cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='foreign_inflow', sub_ly=['cum_daily_ret_a', 'cum_daily_ret', 'cum_ret'])

    def indicator_bull_or_bear(self):
        temp_df = self.df.copy()
        temp_df['15_ma'] = temp_df['Close'].rolling(window=15).mean()
        temp_df['divergence'] = (temp_df['Close'] / temp_df['15_ma']) - 1
        temp_df['divergence'] = temp_df['divergence'].shift(1)
        temp_df = temp_df.dropna(subset=['divergence'])
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()

        temp_df = temp_df.sort_values(by='divergence').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'])

    def indicator_weekday_stats(self):
        """
        統計並繪製每週各交易日 (Mon-Fri) 的平均報酬率
        """
        import plotly.express as px
        
        df = self.df.copy()
        df['weekday'] = df.index.weekday
        
        # Group by weekday
        weekday_stats = df.groupby('weekday')[['daily_ret', 'daily_ret_a']].mean()
        
        # Rename index for better readability
        weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
        weekday_stats.index = weekday_stats.index.map(weekday_map)
        
        print("=== Weekday Average Returns ===")
        display(weekday_stats)
        
        # Plot
        fig = px.bar(
            weekday_stats, 
            barmode='group',
            title='Average Return by Weekday',
            labels={'value': 'Avg Return', 'index': 'Weekday', 'variable': 'Session'}
        )
        fig.show()

    def indicator_US_bond(self, indicator: str, sub_analysis: bool = False):
        import numpy as np
        temp_df = self.df.copy()
        ind_list = [
            'yield_shock', 'yield_divergence', 'yield_presure',
            'cash_crunch', 'near_inversion', 'near_yield_vol'
            ]

        ffill_col = ['US_bond_5y', 'US_bond_3m']
        for col in ffill_col:
            if temp_df[col].isna().sum() > 0:
                temp_df[col] = temp_df[col].ffill()

        # 長債
        temp_df['yield_shock'] = temp_df['US_bond_5y'] - temp_df['US_bond_5y'].shift(20)
        temp_df['30ma_5y'] = temp_df['US_bond_5y'].rolling(window=30).mean()
        temp_df['yield_divergence'] = (temp_df['US_bond_5y'] / temp_df['30ma_5y']) - 1
        temp_df['yield_spread'] = temp_df['US_bond_5y'] - temp_df['US_bond_3m']
        temp_df['spread_delta'] = temp_df['yield_spread'].diff(15)
        temp_df['yield_presure'] = np.where(
            temp_df['yield_spread'] < 0,
            temp_df['spread_delta'],
            0
        )

        # 短債
            ## 資金成本成長多少，越大資金越緊縮
        temp_df['cash_crunch'] = temp_df['US_bond_3m'].diff(5)
            ## 「現在」比「未來」更缺錢
        temp_df['near_inversion'] = temp_df['US_bond_6m'] - temp_df['US_bond_3m']
            ## 市場對資金水位感到恐慌
        temp_df['near_yield_vol'] = temp_df['US_bond_3m'].rolling(10).std()

        temp_df[ind_list] = temp_df[ind_list].shift(2)

        temp_df['demean_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demean_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['daily_ret_a'] = temp_df['daily_ret_a']
        temp_df['daily_ret'] = temp_df['daily_ret']

        if indicator in ind_list:
            temp_df = temp_df.sort_values(by=indicator).reset_index(drop=True)
            temp_df['cum_demean_daily_ret_a'] = temp_df['demean_daily_ret_a'].cumsum()
            temp_df['cum_demean_daily_ret'] = temp_df['demean_daily_ret'].cumsum()
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_demean_daily_ret_a', 'cum_demean_daily_ret'], ry=indicator, sub_ly=['cum_daily_ret_a', 'cum_daily_ret'])

    def check_risk_events(self, filter_tech_signal: bool = False) -> pd.DataFrame:
        """
        檢查並列出觸發風控的所有日期與原因 (Risk Event Log)
        
        Args:
            factors: 因子列表
            filter_tech_signal: 是否只顯示 "技術面有訊號 (Strong Buy/Dip Buy) 但被風控擋掉" 的事件
        """
        df = self.df.copy()
        
        # 1. 計算指標
        df = self._calculate_factors(df)
        
        events = []
        
        # Helper function
        def add_event(date, factor_name, value, action, tech_val):
            # 判斷技術面是否有訊號 (Layer 4 想要進場的條件)
            # 1. Day Momentum: > 0.0045
            # 2. Night Safe Trend: > 0.0035
            # 只要 > 0.0035 就視為有技術面訊號
            is_tech_signal = (tech_val > 0.0035)
            
            # 如果開啟過濾，且沒有技術訊號，則不加入
            if filter_tech_signal and not is_tech_signal:
                return

            events.append({
                'Date': date,
                'Factor': factor_name,
                'Value': value,
                'Action': action,
                'Tech_Signal': 'Buy' if is_tech_signal else 'Neutral',
                'Divergence': tech_val
            })

        # --- Check Logic (Must match _apply_signals_logic) ---
        
        # A. 估值衝擊 (Yield Shock)
        mask = df['yield_shock'] > 0.15
        for date, val in df.loc[mask, 'yield_shock'].items():
            tech_val = df.loc[date, 'divergence']
            add_event(date, 'Yield Shock', val, 'Flat All (Shock > 0.15)', tech_val)

        # E. Holiday
        mask = df['holiday'] > 2
        for date, val in df.loc[mask, 'holiday'].items():
            tech_val = df.loc[date, 'divergence']
            add_event(date, 'Holiday', val, 'Flat All (Holiday > 2)', tech_val)

        # F. Technical
        mask = df['divergence'] < 0.0045 # Technical Exit
        for date, val in df.loc[mask, 'divergence'].items():
            add_event(date, 'Technical', val, 'Flat Day (Div < 0.0045)', val)

        if not events:
            return pd.DataFrame(columns=['Date', 'Factor', 'Value', 'Action', 'Tech_Signal', 'Divergence'])

        res_df = pd.DataFrame(events)
        res_df = res_df.sort_values(by='Date')
        return res_df

    def backtest(self, risk_log: bool = False, point_version: bool = False):
        """
        回測策略 (Backtest Strategy)
        """
        df = self.df.copy()

        # 1. 指標計算 (Indicators Calculation)
        df = self._calculate_factors(df)

        # 2. 策略邏輯 (Strategy Logic)
        df = self._apply_signals_logic(df)

        # 3. 計算回測結果 (Backtest PnL)
        if point_version:
            df['strat_ret'] = (df['daily_pnl_a'] * df['pos_night']) + (df['daily_pnl'] * df['pos_day'])
            df['benchmark_ret'] = (df['daily_pnl_a'] + df['daily_pnl'])
            y_label = "Points"
        else:
            df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
            df['benchmark_ret'] = (df['daily_ret_a'] + df['daily_ret'])
            y_label = "Returns (%)"

        df['cum_strat'] = df['strat_ret'].cumsum()
        df['cum_bnh'] = df['benchmark_ret'].cumsum()

        # ===============================================================
        # 4. 顯示績效統計 (Performance Metrics)
        # ===============================================================
        print(f"=== Performance Metrics ({'Points' if point_version else 'Percentage'}) ===")
        # Strategy Metrics
        strat_metrics = self._calculate_metrics(df['strat_ret'], point_version=point_version)
        # Benchmark Metrics
        bnh_metrics = self._calculate_metrics(df['benchmark_ret'], point_version=point_version)
        
        # Combine into DataFrame
        metrics_df = pd.DataFrame([strat_metrics, bnh_metrics], index=['Strategy', 'Benchmark']).T
        
        # Formatting function
        def format_metrics(val, name):
            if isinstance(val, (int, float)):
                if not point_version and ('Return' in name or 'CAGR' in name or 'Volatility' in name or 'Drawdown' in name or 'Win Rate' in name or 'Avg' in name):
                     return f"{val*100:.2f}%"
                elif point_version:
                    if 'Win Rate' in name:
                         return f"{val*100:.2f}%"
                    elif 'Total PnL' in name or 'Points DD' in name or 'Avg' in name:
                        return f"{val:.2f} pts"
                    else:
                        return f"{val:.2f}"
                elif not point_version:
                    if 'Duration' in name:
                        return f"{int(val)} days"
                    else:
                        return f"{val:.2f}"
            return val

        # Apply formatting
        formatted_df = metrics_df.copy().astype(object)
        for idx in formatted_df.index:
            formatted_df.loc[idx] = formatted_df.loc[idx].apply(lambda x: format_metrics(x, idx))

        display(formatted_df.T)

        # ===============================================================
        # 5. 顯示風控事件 (Risk Events)
        # ===============================================================
        if risk_log:
            print("=== Risk Events Log (Top 20) ===")
            risk_events = self.check_risk_events()
            if not risk_events.empty:
                display(risk_events.tail(20)) # 顯示最近 20 筆，避免洗版
            else:
                print("No risk events triggered.")

        return plot.plot(df, ly=['cum_strat', 'cum_bnh'])

    def show_factor_distributions(self):
        """
        顯示策略使用之各項指標因子分佈情形 (Histograms & Statistics)
        """
        from plotly.subplots import make_subplots
        
        df = self.df.copy()
        # 確保因子已計算
        df = self._calculate_factors(df)
        
        # 定義主要因子 (對應 _apply_signals_logic 使用的)
        factors = ['yield_shock', 'near_yield_vol', 'divergence', 'holiday']
        
        # 移除沒有該 flag 的情形 (例如 holiday 需要計算)
        valid_factors = [f for f in factors if f in df.columns]

        if not valid_factors:
            print("No valid factors found to plot.")
            return

        # 1. 統計數據
        print("=== Factor Statistics ===")
        display(df[valid_factors].describe())
        
        # 2. 繪圖 (2x2 Grid)
        rows = 2
        cols = 2
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=valid_factors)
        
        for i, col in enumerate(valid_factors):
            row = (i // cols) + 1
            c = (i % cols) + 1
            
            # 過濾 NaN
            series = df[col].dropna()
            
            fig.add_trace(
                go.Histogram(x=series, name=col, nbinsx=100, histnorm='probability'),
                row=row, col=c
            )
            
        fig.update_layout(
            title_text="Strategy Factors Distribution", 
            showlegend=False,
            height=700,
        )
        fig.show()

    def show_performance_distributions(self, rolling_window: int = 126):
        """
        顯示策略績效指標的分佈 (Rolling Metrics & Returns Distribution)
        rolling_window: 滾動窗口天數 (預設 126 天約半年)
        包含所有指標: CAGR, Volatility, Sharpe, Max Drawdown, Max DD Duration, 
        Profit Factor, Win Rate, Odds, Avg Win, Avg Loss, Avg Return (Exp), Kelly
        """
        from plotly.subplots import make_subplots
        from tqdm import tqdm
        
        df = self.df.copy()
        
        # 自動執行回測邏輯以取得報酬率 (若尚未計算)
        if 'strat_ret' not in df.columns:
            df = self._calculate_factors(df)
            df = self._apply_signals_logic(df)
            df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])

        # 1. 準備滾動數據
        daily_rets = df['strat_ret'].dropna()
        n_samples = len(daily_rets)
        
        if n_samples < rolling_window:
            print(f"Not enough data for rolling window {rolling_window}. Samples: {n_samples}")
            return

        rolling_metrics = []
        
        # 使用迴圈計算 (雖然較慢但最準確，且能重用 _calculate_metrics 邏輯)
        # 為了效率，先取得 numpy array
        dates = daily_rets.index.tolist()
        
        print(f"Calculating rolling metrics for {n_samples - rolling_window + 1} windows...")
        for i in tqdm(range(n_samples - rolling_window + 1), desc="Rolling Metrics"):
            window_slice = daily_rets.iloc[i : i+rolling_window]
            # 呼叫既有的計算函數 (回傳的是 raw float dict)
            metrics = self._calculate_metrics(window_slice)
            # 加上日期標籤 (用窗口最後一天)
            metrics['Date'] = dates[i+rolling_window-1]
            rolling_metrics.append(metrics)
            
        r_df = pd.DataFrame(rolling_metrics).set_index('Date')
        
        # 移除 Total Return (因為是 Rolling 的，Total Return 只是該期間的報酬，用 CAGR 或 Avg Return 可能較好，但這裡還是會有)
        # 這裡根據需求顯示 12 個指標
        target_metrics = [
            'CAGR', 'Volatility', 'Sharpe', 
            'Max Drawdown', 'Max DD Duration', 'Profit Factor', 
            'Win Rate', 'Odds', 'Kelly',
            'Avg Win', 'Avg Loss', 'Avg Return (Exp)'
        ]
        
        # 2. 統計摘要
        print(f"=== Rolling Performance Statistics (Window: {rolling_window} days) ===")
        display(r_df[target_metrics].describe())

        # 3. 繪圖 (4x3 Grid)
        rows = 4
        cols = 3
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=target_metrics)
        
        for i, metric in enumerate(target_metrics):
            if metric not in r_df.columns:
                continue
                
            row = (i // cols) + 1
            c = (i % cols) + 1
            
            # 過濾 inf / nan
            series = r_df[metric].replace([np.inf, -np.inf], np.nan).dropna()
            
            # 根據指標設定不同顏色 (綠色好/紅色壞 的概念，或統一)
            color = '#1f77b4' # default blue
            if metric in ['Max Drawdown', 'Volatility', 'Avg Loss', 'Max DD Duration']:
                color = '#d62728' # red for risk/less is better
            elif metric in ['Sharpe', 'CAGR', 'Profit Factor', 'Win Rate', 'Kelly']:
                color = '#2ca02c' # green for good
                
            fig.add_trace(
                go.Histogram(x=series, name=metric, nbinsx=50, histnorm='probability', marker_color=color),
                row=row, col=c
            )
            
            # Add mean line annotation (optional/cluttered)
        
        fig.update_layout(
            title_text=f"Rolling Strategy Performance Distributions (Window: {rolling_window} days)", 
            showlegend=False,
            height=1000,
        )
        fig.show()