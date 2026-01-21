import pandas as pd
import module.plot_func as plot
import plotly.graph_objects as go
import IPython.display as display

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
    
    def display_df(self):
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

    def indicator_vwap(self):
        temp_df = self.df.copy()
        temp_df['CumPV'] = (temp_df['Close'] * temp_df['Volume']).rolling(1).sum()
        temp_df['CumV'] = temp_df['Volume'].rolling(1).sum()
        temp_df['vwap'] = temp_df['CumPV'] / temp_df['CumV']
        
        temp_df['vwap'] = temp_df['vwap'].shift(1)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df = temp_df.sort_values(by='vwap').reset_index(drop=True)
        temp_df['cum_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()

        return plot.plot(temp_df, ly='cum_daily_ret_a', ry='vwap', sub_ly='daily_ret_a')

    def indicator_gap_days(self):
        temp_df = self.df.copy()
        cols_a = temp_df.columns[temp_df.columns.str.endswith('_a')]
        
        temp_df[cols_a] = temp_df[cols_a].shift(-1)
        
        temp_df['prev_date'] = temp_df.index.to_series().shift(1)
        temp_df['gap'] = (temp_df.index.to_series() - temp_df['prev_date']).dt.days
                
        temp_df = temp_df.sort_values(by='gap').reset_index(drop=True)
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['cum_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()

        return plot.plot(temp_df, ly='cum_daily_ret_a', ry='gap', sub_ly='daily_ret_a')

    