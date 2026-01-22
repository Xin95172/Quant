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
        temp_df['cum_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        return plot.plot(temp_df, ly='cum_daily_ret_a', x='index', ry = 'daily_ret', sub_ly='daily_ret_a')

    def indicator_gap_days(self,demean: bool = False, point_version: bool = False):
        temp_df = self.df.copy()
        temp_df['daily_ret'] = temp_df['daily_ret'].shift(1)
        temp_df.index = pd.to_datetime(temp_df.index)
        temp_df['prev_date'] = temp_df.index.to_series().shift(1)
        temp_df['gap'] = (temp_df.index.to_series() - temp_df['prev_date']).dt.days
        temp_df = temp_df.sort_values(by='gap').reset_index(drop=True)
        if point_version:
            temp_df['daily_ret_a'] = (temp_df['Close_a'] - temp_df['Open_a'])
            temp_df['daily_ret'] = (temp_df['Close'] - temp_df['Open'])
        if demean:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        else:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a']
            temp_df['daily_ret'] = temp_df['daily_ret']
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()

        return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='gap', sub_ly='daily_ret_a')

    def indicator_maintenance_rate(self, demean: bool = False, point_version: bool = False):
        if 'TotalExchangeMarginMaintenance' not in self.df.columns:
            raise ValueError("TotalExchangeMarginMaintenance is not in the DataFrame.")
        temp_df = self.df.copy()
        temp_df['TotalExchangeMarginMaintenance'] = temp_df['TotalExchangeMarginMaintenance'].shift(1)
        if point_version:
            temp_df['daily_ret_a'] = (temp_df['Close_a'] - temp_df['Open_a'])
            temp_df['daily_ret'] = (temp_df['Close'] - temp_df['Open'])
        if demean:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        else:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a']
            temp_df['daily_ret'] = temp_df['daily_ret']
        temp_df = temp_df.sort_values(by='TotalExchangeMarginMaintenance').reset_index(drop=True)
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()

        return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='TotalExchangeMarginMaintenance', sub_ly='daily_ret_a')

    def indicator_margin_delta(self, demean: bool = False):
        temp_df = self.df.copy()
        temp_df['margin_delta'] = (temp_df['MarginPurchaseMoney']/temp_df['MarginPurchaseMoney'].shift(1)) - 1
        temp_df['avg_margin_delta'] = temp_df['margin_delta'].shift(1).rolling(window=20).mean()
        temp_df = temp_df.sort_values(by='avg_margin_delta').reset_index(drop=True)
        if demean:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        else:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a']
            temp_df['daily_ret'] = temp_df['daily_ret']
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='avg_margin_delta', sub_ly=['daily_ret_a'])

    def indicator_institutional_flow(self, demean: bool = False):
        temp_df = self.df.copy()
        temp_df['foreign_inflow'] = (temp_df['Net_Foreign_Investor'] - temp_df['Net_Foreign_Investor'].rolling(window=20).mean()) / temp_df['Net_Foreign_Investor'].rolling(window=20).std()
        temp_df['foreign_inflow'] = temp_df['foreign_inflow'].shift(1)
        if demean:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        else:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a']
            temp_df['daily_ret'] = temp_df['daily_ret']
        temp_df = temp_df.sort_values(by='foreign_inflow').reset_index(drop=True)
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='foreign_inflow', sub_ly='daily_ret_a')

    def indicator_bull_or_bear(self, demean: bool = False):
        temp_df = self.df.copy()
        temp_df['15_ma'] = temp_df['Close'].rolling(window=15).mean()
        temp_df['divergence'] = (((temp_df['Close'] + temp_df['Open_a']) / 2) / temp_df['15_ma']) - 1
        temp_df['divergence'] = temp_df['divergence'].shift(1)
        
        # Drop NaN before sorting to prevent noise at the end of the plot
        temp_df = temp_df.dropna(subset=['divergence'])
        
        if demean:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        else:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a']
            temp_df['daily_ret'] = temp_df['daily_ret']
        temp_df = temp_df.sort_values(by='divergence').reset_index(drop=True)
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        display(self._get_statistics(ret_col=temp_df.loc[temp_df['divergence'] > 0, 'daily_ret_a']))
        display(self._get_statistics(ret_col=temp_df.loc[temp_df['divergence'] > 0, 'daily_ret']))
        return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='divergence', sub_ly='daily_ret_a', point_ry=0)

    def indicator_US_bond(self, demean: bool = False, indicator: [str] = ['yield_shock', 'yield_divergence', 'yield_presure']):
        import numpy as np
        temp_df = self.df.copy()

        # 長債
        if temp_df['US_bond_5y'].isna().sum() > 0:
            temp_df['US_bond_5y'] = temp_df['US_bond_5y'].ffill()
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
        if temp_df['US_bond_3m'].isna().sum() > 0:
            temp_df['US_bond_3m'] = temp_df['US_bond_3m'].ffill()
            ## 資金成本成長多少，越大資金越緊縮
        temp_df['cash_crunch'] = temp_df['US_bond_3m'].diff(5)
            ## 「現在」比「未來」更缺錢，大家寧願鎖定未來的低利率，也不願意現在借錢給別人。這是極度且立即的流動性壓力
        temp_df['near_inversion'] = temp_df['US_bond_6m'] - temp_df['US_bond_3m']
            ## 市場對資金水位感到恐慌
        temp_df['near_yield_vol'] = temp_df['US_bond_3m'].rolling(10).std()

        temp_df[['yield_shock', 'yield_divergence', 'yield_presure', 'cash_crunch', 'near_inversion', 'near_yield_vol']] = temp_df[['yield_shock', 'yield_divergence', 'yield_presure', 'cash_crunch', 'near_inversion', 'near_yield_vol']].shift(2)

        if demean:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
            temp_df['daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()
        else:
            temp_df['daily_ret_a'] = temp_df['daily_ret_a']
            temp_df['daily_ret'] = temp_df['daily_ret']

        if 'yield_shock' in indicator:
            temp_df = temp_df.sort_values(by='yield_shock').reset_index(drop=True)
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='yield_shock', sub_ly='daily_ret_a')
        if 'yield_divergence' in indicator:
            temp_df = temp_df.sort_values(by='yield_divergence').reset_index(drop=True)
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='yield_divergence', sub_ly='daily_ret_a')
        if 'yield_presure' in indicator:
            temp_df = temp_df.sort_values(by='yield_presure').reset_index(drop=True)
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='yield_presure', sub_ly='daily_ret_a')
        if 'cash_crunch' in indicator:
            temp_df = temp_df.sort_values(by='cash_crunch').reset_index(drop=True)
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='cash_crunch', sub_ly='daily_ret_a')
        if 'near_inversion' in indicator:
            temp_df = temp_df.sort_values(by='near_inversion').reset_index(drop=True)
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='near_inversion', sub_ly='daily_ret_a')
        if 'near_yield_vol' in indicator:
            temp_df = temp_df.sort_values(by='near_yield_vol').reset_index(drop=True)
            temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
            temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
            return plot.plot(temp_df, ly=['cum_daily_ret_a', 'cum_daily_ret'], ry='near_yield_vol', sub_ly='daily_ret_a')