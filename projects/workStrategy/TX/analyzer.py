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
        
        self.df['daily_ret_a'] = (self.df['Close_a'] / self.df['Open_a']) - 1
        self.df['cum_daily_ret_a'] = self.df['daily_ret_a'].cumsum()
    
    def _get_statistics(self, ret_col: pd.Series):
        return ret_col.describe()
    
    def display_df(self):
        return self.df
    
    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper to calculate all strategy factors."""
        # --- A. 宏觀債券 (Macro - The Risk Radar) ---
        # 1. 估值殺手 (Valuation): 20天長債變動
        df['yield_shock'] = df['US_bond_5y'] - df['US_bond_5y'].shift(20)
        # 2. 救命訊號 (Liquidity Crisis): 倒掛急陡 (6m - 3m)
        df['near_inversion'] = df['US_bond_6m'] - df['US_bond_3m']
        # 3. 恐慌指數 (Panic/ATM): 短債波動率 (提款機效應)
        df['near_yield_vol'] = df['US_bond_3m'].rolling(20).std() 
        
        # 4. 趨勢過熱 (Overheated): 5年債乖離
        df['30ma_5y'] = df['US_bond_5y'].rolling(window=30).mean()
        df['yield_divergence'] = (df['US_bond_5y'] / df['30ma_5y']) - 1

        # 防未來數據 (Macro指標通常落後或當天晚上才看得到，保守 shift 2)
        macro_cols = ['yield_shock', 'near_inversion', 'near_yield_vol', 'yield_divergence']
        df[macro_cols] = df[macro_cols].shift(2)

        # --- B. 日曆效應 (Calendar - The Time Decay) ---
        # 計算 Gap (T日 - T-1日的天數)
        df.index = pd.to_datetime(df.index)
        df['prev_date'] = df.index.to_series().shift(1)
        df['holiday'] = (df.index.to_series() - df['prev_date']).dt.days

        # --- C. 技術乖離 (Technical - The Entry Trigger) ---
        # 15MA 乖離率
        df['15_ma'] = df['Close'].rolling(window=15).mean()
        df['divergence'] = (df['Close'] / df['15_ma']) - 1
        df['divergence'] = df['divergence'].shift(1)
        
        return df

    def _apply_signals_logic(self, df: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
        """Helper to apply position sizing logic based on calculated factors."""
        # 初始化部位
        df['pos_night'] = 1.0
        df['pos_day'] = 1.0

        # Layer 2: 宏觀濾網 (Macro Overlays) - 權重高於籌碼
        
        # A. 估值衝擊 (Yield Shock) -> 禁止做多
        if 'yield_shock' in factors:
            mask_shock = df['yield_shock'] > 0.15
            df.loc[mask_shock, ['pos_night', 'pos_day']] = 0.0

        # B. 提款機效應 (Yield Volatility) -> 觀望
            ## 可以考慮 < 0.013 加碼
            ## 在考慮要不要 > 0.015 夜盤也不做，減少 overfitting
        if 'near_yield_vol' in factors:
            mask_vol = df['near_yield_vol'] > 0.015
            df.loc[mask_vol, ['pos_day']] = 0.0

        # C. 核彈警報 (Inversion Crisis) -> 全面平倉
        if 'near_inversion' in factors:
            mask_crash = df['near_inversion'] > 0.3
            df.loc[mask_crash, ['pos_night', 'pos_day']] = 0.0

        # D. 債券乖離 (Yield Divergence) -> 觀望/休息
        # 這裡假設如果開啟 yield_shock 也一併檢查 yield_divergence，或是預設檢查
        if 'yield_shock' in factors:
            # 乖離過大休息
            df.loc[df['yield_divergence'] > 0.06, ['pos_night', 'pos_day']] = 0.0
            df.loc[df['yield_divergence'] < -0.04, ['pos_night', 'pos_day']] = 0.0

        # Layer 3: 日曆風控 (Holiday Risk)
        if 'holiday' in factors:
            # Gap > 1 代表這行是 "長假後的第一行" (如週一)，也就是包含 "長假前夜盤" (週五夜) 的那一行
            df.loc[df['holiday'] > 2, 'pos_night'] = 0.0

        # Layer 4: 技術微調 (Technical Entry) - 僅在無重大宏觀風險時啟用
        if 'technical' in factors:
            # 只有在沒有 Macro 警報 (Level 2 & 3 未觸發) 時，才用乖離率做逆勢/順勢加碼
            # 這裡我們簡單定義 "Macro Safe"，如果沒開 Macro 因子就預設 True
            is_shock = (df['yield_shock'] > 0.15) if 'yield_shock' in factors else False
            is_vol = (df['near_yield_vol'] > 0.015) if 'near_yield_vol' in factors else False
            is_crash = (df['near_inversion'] > 0.3) if 'near_inversion' in factors else False
            
            safe_zone = (~is_crash) & (~is_vol)
            
            # 夜盤喜歡均值回歸 (跌深買)
            # 如果乖離 < -0.05 且 宏觀安全 -> 夜盤嘗試抄底
            df.loc[safe_zone & (df['divergence'] < -0.05), 'pos_night'] = 1.0
            
            # 日盤/夜盤 順勢做多 (正乖離強勢)
            df.loc[df['divergence'] > 0.0045, 'pos_day'] = 1.0
            df.loc[safe_zone & (df['divergence'] > 0.0045), 'pos_night'] = 1.0

            # 日盤原本有做空邏輯，現移除或改觀望
            # 如果乖離 < -0.05 -> 觀望
            df.loc[df['divergence'] < -0.05, 'pos_day'] = 0.0
            
        return df

    def _calculate_metrics(self, returns: pd.Series) -> dict:
        """
        Calculate strategy performance metrics including advanced stats.
        returns: Series of daily returns (or PnL percentages)
        """
        if returns.empty:
            return {}

        # 1. Basic Distributions
        total_return = returns.sum() # Simple sum for daily percentage or log returns approximation
        # Cumulative Compounded Return (for Total Return % accuracy)
        # Assuming returns are simple returns:
        cum_ret = (1 + returns).cumprod()
        total_ret_pct = (cum_ret.iloc[-1] - 1) if not cum_ret.empty else 0.0
        
        # 2. Time Statistics
        trading_days = len(returns)
        # Annualized Factor (Crypto=365, Stock=252. TW Futures ~ 250)
        ann_factor = 252 
        
        # CAGR
        if trading_days > 0:
            cagr = (1 + total_ret_pct) ** (ann_factor / trading_days) - 1
        else:
            cagr = 0.0

        # Volatility (Annualized)
        vol_ann = returns.std() * np.sqrt(ann_factor)

        # Sharpe Ratio (Rf=0)
        if vol_ann > 0:
            sharpe = (cagr / vol_ann) 
        else:
            sharpe = 0.0

        # 3. Drawdown Statistics
        running_max = cum_ret.cummax()
        drawdown = (cum_ret / running_max) - 1
        max_dd = drawdown.min()
        
        # Max DD Duration (Days)
        # Calculate streaks of drawdown < 0
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
            'Total Return': f"{total_ret_pct*100:.2f}%",
            'CAGR': f"{cagr*100:.2f}%",
            'Volatility': f"{vol_ann*100:.2f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd*100:.2f}%",
            'Max DD Duration': f"{max_dd_duration} days",
            'Profit Factor': f"{profit_factor:.2f}",
            'Win Rate': f"{win_rate*100:.2f}%",
            'Odds': f"{odds:.2f}",
            'Avg Win': f"{avg_win*100:.2f}%",
            'Avg Loss': f"{avg_loss*100:.2f}%",
            'Avg Return (Exp)': f"{avg_return*100:.2f}%",
            'Kelly': f"{kelly:.2f}"
        }

    def update_df(self, df):
        self.df = df
        return self.df

    def daily_ret(self):
        return plot.plot(self.df, ly=['cum_daily_ret', 'cum_daily_ret_a'])

    def monthly_ret(self, session: str = 'night'):
        """
        session: 'night' (default) or 'day'
        """
        target_col = 'daily_ret_a' if session == 'night' else 'daily_ret'
        label = "Night Session" if session == 'night' else "Day Session"
        
        df = self.df.copy()
        df['year'] = df.index.year
        df['month'] = df.index.month
        
        # Calculate monthly returns (compounded)
        monthly_df = df.groupby(['year', 'month'])[target_col].apply(lambda x: (1 + x).prod() - 1).reset_index()
        monthly_df['return_pct'] = monthly_df[target_col] * 100
        
        # Create Pivot Table
        pivot_df = monthly_df.pivot(index='year', columns='month', values='return_pct')
        
        # Prepare annotations
        annotations = []
        for y_idx, row in pivot_df.iterrows():
            for x_idx, val in row.items():
                if pd.notna(val):
                    annotations.append(
                        dict(
                            x=x_idx, 
                            y=y_idx, 
                            text=f"{val:.2f}", 
                            showarrow=False, 
                            font=dict(color='black' if abs(val) < 5 else 'white')
                        )
                    )
                    
        # Plot Heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdYlGn',
            colorbar=dict(title='Return %'),
            zmid=0
        ))
        
        fig.update_layout(
            title=f'Monthly P&L Heatmap - {label}',
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

    def indicator_gap_days(self, after_holiday: bool = False):
        temp_df = self.df.copy()
        temp_df.index = pd.to_datetime(temp_df.index)
        temp_df['prev_date'] = temp_df.index.to_series().shift(1)
        temp_df['gap'] = (temp_df.index.to_series() - temp_df['prev_date']).dt.days
        if after_holiday:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'].shift(-1)
        elif after_holiday is False:
            temp_df['daily_ret'] = temp_df['daily_ret'].shift(1)

        temp_df = temp_df.sort_values(by='gap').reset_index(drop=True)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()

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

    def indicator_US_bond(self, indicator: str):
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

    def backtest(self, factors: list[str] = ['foreign_inflow', 'yield_shock', 'near_yield_vol', 'near_inversion', 'holiday', 'technical']):
        """
        回測策略 (Backtest Strategy)
        
        Args:
            factors (list[str]): 選擇要啟用的因子列表。
                - 'foreign_inflow': 籌碼因子 (外資 Z-score)
                - 'yield_shock': 宏觀因子 (估值衝擊)
                - 'near_yield_vol': 宏觀因子 (提款機效應/短債波動)
                - 'near_inversion': 宏觀因子 (核彈警報/倒掛)
                - 'holiday': 日曆因子 (長假風險)
                - 'technical': 技術因子 (15MA 乖離)
        """
        df = self.df.copy()

        # 1. 指標計算 (Indicators Calculation)
        df = self._calculate_factors(df)

        # 2. 策略邏輯 (Strategy Logic)
        df = self._apply_signals_logic(df, factors)

        # 3. 計算回測結果 (Backtest PnL)

        # ===============================================================
        # 3. 計算回測結果 (Backtest PnL)
        # ===============================================================
        df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
        df['cum_strat'] = df['strat_ret'].cumsum()
        
        # 基準：Buy & Hold
        df['cum_bnh'] = (df['daily_ret_a'] + df['daily_ret']).cumsum()

        # ===============================================================
        # 4. 顯示績效統計 (Performance Metrics)
        # ===============================================================
        print("=== Performance Metrics ===")
        # Strategy Metrics
        strat_metrics = self._calculate_metrics(df['strat_ret'])
        # Benchmark Metrics
        benchmark_ret = df['daily_ret_a'] + df['daily_ret']
        bnh_metrics = self._calculate_metrics(benchmark_ret)
        
        # Combine into DataFrame
        metrics_df = pd.DataFrame([strat_metrics, bnh_metrics], index=['Strategy', 'Benchmark']).T
        display(metrics_df.T)

        return plot.plot(df, ly=['cum_strat', 'cum_bnh'])