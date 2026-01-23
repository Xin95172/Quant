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
        temp_df['near_yield_vol'] = temp_df['US_bond_3m'].rolling(20).std()

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

    def backtest(self):
        df = self.df.copy()

        # ===============================================================
        # 1. 指標計算 (Indicators Calculation)
        # ===============================================================

        # --- A. 宏觀債券 (Macro - The Risk Radar) ---
        # 1. 估值殺手 (Valuation): 20天長債變動
        df['yield_shock'] = df['US_bond_5y'] - df['US_bond_5y'].shift(20)
        # 2. 救命訊號 (Liquidity Crisis): 倒掛急陡 (6m - 3m)
        df['near_inversion'] = df['US_bond_6m'] - df['US_bond_3m']
        # 3. 恐慌指數 (Panic/ATM): 短債波動率 (提款機效應)
        df['near_yield_vol'] = df['US_bond_3m'].rolling(20).std() # 改成20天更穩
        
        # 防未來數據 (Macro指標通常落後或當天晚上才看得到，保守 shift 2)
        macro_cols = ['yield_shock', 'near_inversion', 'near_yield_vol']
        df[macro_cols] = df[macro_cols].shift(2)

        # --- B. 籌碼流向 (Flow - The Trend Setter) ---
        # 外資 Z-score
        df['foreign_inflow'] = (df['Net_Foreign_Investor'] - df['Net_Foreign_Investor'].rolling(20).mean()) / df['Net_Foreign_Investor'].rolling(20).std()
        df['foreign_inflow'] = df['foreign_inflow'].shift(1) # T-1 下午出來，T日可用

        # --- C. 日曆效應 (Calendar - The Time Decay) ---
        # 計算 Gap (T日 - T-1日的天數)
        df.index = pd.to_datetime(df.index)
        df['prev_date'] = df.index.to_series().shift(1)
        df['gap'] = (df.index.to_series() - df['prev_date']).dt.days
        # 注意：Gap是在 T日 開盤才知道，但策略是在 T-1 收盤決定要不要留倉
        # 這裡我們用簡單規則：如果今天是週五(weekday=4)，視為潛在 Gap 風險
        df['is_friday'] = np.where(df.index.dayofweek == 4, 1, 0)

        # --- D. 技術乖離 (Technical - The Entry Trigger) ---
        # 15MA 乖離率
        df['15_ma'] = df['Close'].rolling(window=15).mean()
        df['divergence'] = (df['Close'] / df['15_ma']) - 1
        
        # 訊號對齊：
        # 預測夜盤 (T日晚) -> 用 T日 13:30 收盤的乖離 (需 shift 1 對齊到 T+1 的夜盤 row)
        df['signal_div_night'] = df['divergence'].shift(1)
        # 預測日盤 (T日早) -> 用 T-1日 13:30 收盤的乖離 (Shift 1)
        df['signal_div_day'] = df['divergence'].shift(1)

        # ===============================================================
        # 2. 策略邏輯 (Strategy Logic)
        # ===============================================================
        
        # 初始化部位
        df['pos_night'] = 0.0
        df['pos_day'] = 0.0

        # -----------------------------------------------------------
        # Layer 1: 基礎籌碼趨勢 (Base Trend)
        # -----------------------------------------------------------
        # 邏輯：跟隨外資，但過濾掉中間的雜訊(內資吃豆腐區)
        # 大買 (>1.5) -> 做多
        # 大賣 (<-1.5) -> 做空
        # 中間 -> 空手 (或逆勢，這裡先保守空手)
        
        BUY_THRESHOLD = 1.5
        SELL_THRESHOLD = -1.5
        
        df.loc[df['foreign_inflow'] > BUY_THRESHOLD, ['pos_night', 'pos_day']] = 1.0
        df.loc[df['foreign_inflow'] < SELL_THRESHOLD, ['pos_night', 'pos_day']] = -1.0

        # -----------------------------------------------------------
        # Layer 2: 宏觀濾網 (Macro Overlays) - 權重高於籌碼
        # -----------------------------------------------------------
        
        # A. 估值衝擊 (Yield Shock) -> 禁止做多
        # 如果長債升太快，成長股估值修正，多單全撤
        mask_shock = df['yield_shock'] > 0.15
        df.loc[mask_shock & (df['pos_night'] > 0), 'pos_night'] = 0.0
        df.loc[mask_shock & (df['pos_day'] > 0), 'pos_day'] = 0.0
        
        # B. 提款機效應 (Yield Volatility) -> 日盤轉空，夜盤觀望
        # 邏輯：波動大，外資白天提款，晚上美股震盪
        mask_vol = df['near_yield_vol'] > 0.015
        df.loc[mask_vol, 'pos_night'] = 0.0  # 夜盤避險不玩
        df.loc[mask_vol, 'pos_day'] = -1.0   # 日盤強制做空 (勝率高)

        # C. 核彈警報 (Inversion Crisis) -> 全面做空
        # 邏輯：流動性枯竭，不管是誰都在逃
        mask_crash = df['near_inversion'] > 0.3
        df.loc[mask_crash, ['pos_night', 'pos_day']] = -1.0

        # -----------------------------------------------------------
        # Layer 3: 日曆風控 (Holiday Risk)
        # -----------------------------------------------------------
        # 邏輯：長假前夕(週五)，夜盤不留倉 (Gap風險期望值為負)
        # Gap > 1 代表這行是 "長假後的第一行" (如週一)，也就是包含 "長假前夜盤" (週五夜) 的那一行
        df.loc[df['gap'] > 1, 'pos_night'] = 0.0

        # -----------------------------------------------------------
        # Layer 4: 技術微調 (Technical Entry) - 僅在無重大宏觀風險時啟用
        # -----------------------------------------------------------
        # 只有在沒有 Macro 警報 (Level 2 & 3 未觸發) 時，才用乖離率做逆勢/順勢加碼
        # 這裡簡化邏輯：如果乖離過大，進行修正
        
        safe_zone = (~mask_crash) & (~mask_vol)
        
        # 夜盤喜歡均值回歸 (跌深買)
        # 如果乖離 < -0.05 且 宏觀安全 -> 夜盤嘗試抄底
        df.loc[safe_zone & (df['signal_div_night'] < -0.05), 'pos_night'] = 1.0
        
        # 日盤喜歡動能 (跌深殺)
        # 如果乖離 < -0.05 -> 日盤順勢空 (不要抄底)
        df.loc[df['signal_div_day'] < -0.05, 'pos_day'] = -1.0

        # ===============================================================
        # 3. 計算回測結果 (Backtest PnL)
        # ===============================================================
        df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
        df['cum_strat'] = df['strat_ret'].cumsum()
        
        # 基準：Buy & Hold
        df['cum_bnh'] = (df['daily_ret_a'] + df['daily_ret']).cumsum()

        return plot.plot(df, ly=['cum_strat', 'cum_bnh'])