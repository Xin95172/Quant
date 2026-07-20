from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
import module.plot_func as plot
import plotly.graph_objects as go
from IPython.display import display


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the active TX day-session strategy."""

    backtest_start_date: str = '2026-03-01'
    move_threshold: float = 0.0001
    sox_threshold: float = 0.0075
    gap_threshold: float = 0.001
    foreign_option_threshold: float = -0.0035
    divergence_threshold: float = -0.0015
    day_long_position: float = 1.0
    day_short_position: float = -1.0
    night_position: float = 0.0


class StrategyEngine:
    """Stateless factor and position logic for the active TX strategy."""

    REQUIRED_COLUMNS = {
        'futures_id', 'Open', 'Close', 'Close_a', 'MOVE_open', 'MOVE_high', 'MOVE_low', 'MOVE_close',
        'SOX_open', 'SOX_close', 'Foreign_Opt_Signal_a',
    }

    @classmethod
    def calculate_factors(cls, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of the frame with all strategy-derived factor columns."""
        missing_columns = cls.REQUIRED_COLUMNS - set(frame.columns)
        if missing_columns:
            raise KeyError(f"strategy data missing columns: {sorted(missing_columns)}")

        df = frame.copy()
        df['SOX_ind'] = ((df['SOX_close'] / df['SOX_open']) - 1).shift(1).ffill()
        df['MOVE_ind'] = (df['MOVE_close'] / df['MOVE_open']) - 1
        df['MOVE_vol'] = ((df['MOVE_high'] / df['MOVE_low']) - 1).shift(1)

        df['3_ma'] = df['Close_a'].rolling(window=3).mean()
        df['divergence'] = (df['Close_a'] / df['3_ma']) - 1
        df['3_ma_v2'] = df['Close'].rolling(window=3).mean()
        df['divergence_v2'] = ((df['Close'] / df['3_ma_v2']) - 1).shift(1)
        df['gap'] = (df['Close_a'] / df['Close'].shift(1)) - 1
        df['gap_v2'] = (df['Open'] / df['Close'].shift(1)) - 1
        return df

    @staticmethod
    def apply_positions(frame: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
        """Return active day and night positions for a factor-ready frame."""
        df = frame.loc[frame.index > config.backtest_start_date].copy()
        df.dropna(subset='futures_id', inplace=True)
        df['pos_night'] = config.night_position
        df['pos_day'] = 0.0

        move_below_threshold = df['MOVE_ind'] < config.move_threshold
        sox_below_threshold = df['SOX_ind'] < config.sox_threshold
        foreign_option_bearish = df['Foreign_Opt_Signal_a'] < config.foreign_option_threshold

        base_gap_supports_long = df['gap'] < config.gap_threshold
        base_divergence_supports_long = df['divergence'] < config.divergence_threshold
        df.loc[move_below_threshold & sox_below_threshold & ~base_gap_supports_long, 'pos_day'] = config.day_long_position
        df.loc[move_below_threshold & ~sox_below_threshold, 'pos_day'] = config.day_long_position
        df.loc[~move_below_threshold & foreign_option_bearish, 'pos_day'] = config.day_short_position
        df.loc[~move_below_threshold & ~foreign_option_bearish & ~base_divergence_supports_long, 'pos_day'] = config.day_long_position

        final_divergence_supports_long = df['divergence_v2'] < config.divergence_threshold
        df.loc[move_below_threshold, 'pos_day'] = config.day_long_position
        df.loc[~move_below_threshold & foreign_option_bearish, 'pos_day'] = config.day_short_position
        df.loc[~move_below_threshold & ~foreign_option_bearish & ~final_divergence_supports_long, 'pos_day'] = config.day_long_position
        return df


class TXAnalyzer:
    """Notebook-facing facade for TX features, diagnostics, and backtesting."""

    # Construction and core state -------------------------------------------------
    def __init__(self, df: pd.DataFrame, config: StrategyConfig | None = None):
        self.config = config or StrategyConfig()
        df = df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df_day = df[df['trading_session'] == 'position'].copy()
        df_night = df[df['trading_session'] == 'after_market'].copy()

        if df_day.index.duplicated().any() or df_night.index.duplicated().any():
            raise ValueError('each trading session must contain at most one row per close date')
            
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
    
    def _get_statistics(self, ret_col: pd.Series) -> pd.Series:
        return ret_col.describe()

    def set_config(self, config: StrategyConfig) -> None:
        """Replace the active strategy configuration for subsequent analysis and backtests."""
        self.config = config

    def session_alignment_report(self) -> pd.Series:
        """Check day and night session pairing under the close-date convention.

        The night session labeled ``D`` is expected to close before the day session
        labeled ``D`` opens, so both must share the same normalized date.
        """
        day_available = self.df[['Open', 'Close']].notna().all(axis=1)
        night_available = self.df[['Open_a', 'Close_a']].notna().all(axis=1)
        paired = day_available & night_available
        return pd.Series({
            'total_dates': len(self.df),
            'paired_day_night_dates': int(paired.sum()),
            'day_only_dates': int((day_available & ~night_available).sum()),
            'night_only_dates': int((~day_available & night_available).sum()),
            'missing_both_dates': int((~day_available & ~night_available).sum()),
            'duplicate_dates': int(self.df.index.duplicated().sum()),
            'close_date_alignment_ok': bool(paired.all() and not self.df.index.duplicated().any()),
        }, name='session_alignment')

    def for_period(
        self,
        *,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> "TXAnalyzer":
        """Return an independent analyzer view limited to an inclusive date range."""
        frame = self.df
        if start is not None:
            frame = frame.loc[frame.index >= pd.Timestamp(start)]
        if end is not None:
            frame = frame.loc[frame.index <= pd.Timestamp(end)]

        view = object.__new__(TXAnalyzer)
        view.config = self.config
        view.df = frame.copy()
        return view

    @staticmethod
    def split_periods(
        index: pd.Index,
        *,
        start: str | pd.Timestamp | None = None,
        train_ratio: float = 0.6,
        validation_ratio: float = 0.2,
    ) -> dict[str, pd.Timestamp]:
        """Split ordered trading dates into train, validation, and test periods."""
        if train_ratio <= 0 or validation_ratio <= 0 or train_ratio + validation_ratio >= 1:
            raise ValueError('train_ratio and validation_ratio must be positive and sum to less than 1')

        dates = pd.DatetimeIndex(pd.to_datetime(index)).normalize().unique().sort_values()
        if start is not None:
            dates = dates[dates > pd.Timestamp(start)]
        if len(dates) < 3:
            raise ValueError('at least three trading dates are required to create train, validation, and test periods')

        train_count = max(1, int(len(dates) * train_ratio))
        validation_count = max(1, int(len(dates) * validation_ratio))
        if train_count + validation_count >= len(dates):
            raise ValueError('split ratios leave no trading dates for the test period')

        return {
            'train_end': dates[train_count - 1],
            'validation_start': dates[train_count],
            'validation_end': dates[train_count + validation_count - 1],
            'test_start': dates[train_count + validation_count],
        }
    
    def display_df(self) -> pd.DataFrame:
        return self.df
    
    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        return StrategyEngine.calculate_factors(df)

    def _apply_signals_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        return StrategyEngine.apply_positions(df, self.config)

    def _calculate_metrics(self, returns: pd.Series, point_version: bool = False) -> dict:
        """
        Calculate strategy performance metrics including advanced stats.
        returns: Series of daily returns (percentage if point_version=False, points if True)
        """
        if returns.empty:
            return {}
        returns.dropna(inplace=True)
        
        # 1. Basic Distributions
        total_pnl = returns.sum()
        
        if point_version:
            # Points Mode: Calculation is additive
            cum_ret = returns.cumsum()
            total_ret_val = total_pnl
        else:
            # Percentage Mode: Calculation is compounded
            cum_ret = returns.cumsum()
            total_ret_val = cum_ret.iloc[-1] if not cum_ret.empty else 0.0
        
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
            drawdown = cum_ret - running_max # Pct down
            
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

    def update_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace the working frame after an explicit research transformation."""
        self.df = df
        return self.df

    # Feature pipeline ------------------------------------------------------------
    def merge_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Left-join date-indexed features onto the strategy frame."""
        features = features.copy()
        features.index = pd.to_datetime(features.index).normalize()
        features = features.loc[~features.index.duplicated(keep='last')]

        df = self.df.copy()
        df.index = pd.to_datetime(df.index).normalize()
        df = df.drop(columns=features.columns.intersection(df.columns))
        self.df = df.join(features, how='left')
        return self.df

    def add_market_ohlc(self, market_df: pd.DataFrame, prefix: str, *, shift: int = 1) -> pd.DataFrame:
        """Add a daily market OHLC series with a consistent factor prefix."""
        required_columns = ['open', 'high', 'low', 'close']
        missing_columns = set(required_columns) - set(market_df.columns)
        if missing_columns:
            raise KeyError(f"{prefix} missing OHLC columns: {sorted(missing_columns)}")

        features = market_df[required_columns].copy()
        features.index = pd.to_datetime(features.index).normalize()
        if shift:
            features = features.shift(shift)
        features = features.rename(columns={column: f'{prefix}_{column}' for column in required_columns})
        return self.merge_features(features)

    @staticmethod
    def build_option_signals(
        day_df: pd.DataFrame,
        night_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert day and night institutional option data into strategy signals."""
        if day_df.empty:
            day_signal = pd.DataFrame(columns=['Foreign_Opt_Signal', 'Dealer_Opt_Signal'])
        else:
            required_columns = {'date', 'call_put', 'institutional_investors', 'long_deal_amount', 'short_deal_amount'}
            missing_columns = required_columns - set(day_df.columns)
            if missing_columns:
                raise KeyError(f"day option data missing columns: {sorted(missing_columns)}")

            day_data = day_df.copy()
            day_data['date'] = pd.to_datetime(day_data['date']).dt.normalize()
            day_data['call_put'] = day_data['call_put'].replace({
                '買權': 'CALL', '賣權': 'PUT', 'Call': 'CALL', 'Put': 'PUT',
            })
            day_data['net_amount'] = day_data['long_deal_amount'] - day_data['short_deal_amount']
            day_data['turnover'] = day_data['long_deal_amount'] + day_data['short_deal_amount']
            pivot = day_data.pivot_table(
                index='date',
                columns=['institutional_investors', 'call_put'],
                values=['net_amount', 'turnover'],
                aggfunc='sum',
                fill_value=0,
            )

            def signal_for(institution: str) -> pd.Series:
                net_call = pivot.get(('net_amount', institution, 'CALL'), pd.Series(0, index=pivot.index))
                net_put = pivot.get(('net_amount', institution, 'PUT'), pd.Series(0, index=pivot.index))
                call_turnover = pivot.get(('turnover', institution, 'CALL'), pd.Series(0, index=pivot.index))
                put_turnover = pivot.get(('turnover', institution, 'PUT'), pd.Series(0, index=pivot.index))
                return (net_call - net_put) / (call_turnover + put_turnover).replace(0, np.nan)

            day_signal = pd.DataFrame(index=pivot.index)
            day_signal['Foreign_Opt_Signal'] = signal_for('外資')
            day_signal['Dealer_Opt_Signal'] = signal_for('自營商')

        if night_df.empty:
            night_signal = pd.DataFrame(columns=['Foreign_Opt_Signal_a'])
        else:
            required_columns = {
                'foreign_long_call_amount', 'foreign_short_call_amount',
                'foreign_long_put_amount', 'foreign_short_put_amount',
            }
            missing_columns = required_columns - set(night_df.columns)
            if missing_columns:
                raise KeyError(f"night option data missing columns: {sorted(missing_columns)}")

            night_data = night_df.copy()
            night_data.index = pd.to_datetime(night_data.index).normalize()
            net_call = night_data['foreign_long_call_amount'] - night_data['foreign_short_call_amount']
            net_put = night_data['foreign_long_put_amount'] - night_data['foreign_short_put_amount']
            turnover = night_data[list(required_columns)].sum(axis=1).replace(0, np.nan)
            night_signal = pd.DataFrame({'Foreign_Opt_Signal_a': (net_call - net_put) / turnover})

        return day_signal.join(night_signal, how='outer')

    def add_option_signals(self, day_df: pd.DataFrame, night_df: pd.DataFrame) -> pd.DataFrame:
        """Build and merge institutional option signals."""
        return self.merge_features(self.build_option_signals(day_df, night_df))

    @staticmethod
    def fetch_option_daily(
        client,
        option_id: str,
        start_date: str,
        end_date: str,
        *,
        pause_seconds: float = 0.5,
    ) -> pd.DataFrame:
        """Download option daily data in yearly batches to avoid API request limits."""
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        period_starts = pd.date_range(start=start, end=end, freq='YS')
        if start not in period_starts:
            period_starts = period_starts.insert(0, start).sort_values().unique()

        chunks = []
        for period_start in period_starts:
            period_end = min(period_start + pd.offsets.YearEnd(0), end)
            if period_start > period_end:
                continue

            option_part = client.get_option_daily(
                option_id=option_id,
                start_date=period_start,
                end_date=period_end,
                trading_session='all',
            )
            if not option_part.empty:
                chunks.append(option_part)
            if pause_seconds:
                time.sleep(pause_seconds)

        if not chunks:
            raise RuntimeError(f'未下載到 {option_id} 選擇權資料')
        return pd.concat(chunks, ignore_index=True)

    def add_option_iv_skew(
        self,
        option_df: pd.DataFrame,
        settlement_df: pd.DataFrame,
        *,
        iv_calculator,
        risk_free_rate: float = 0.015,
    ) -> pd.DataFrame:
        """Calculate option IV skew by session and merge it into the strategy frame."""
        required_option_columns = {'date', 'trading_session', 'contract_date'}
        missing_option_columns = required_option_columns - set(option_df.columns)
        if missing_option_columns:
            raise KeyError(f"option data missing columns: {sorted(missing_option_columns)}")
        required_settlement_columns = {'contract', 'settle_date'}
        missing_settlement_columns = required_settlement_columns - set(settlement_df.columns)
        if missing_settlement_columns:
            raise KeyError(f"settlement data missing columns: {sorted(missing_settlement_columns)}")

        options = option_df.copy()
        options['date'] = pd.to_datetime(options['date']).dt.normalize()
        spot = self.df[['Close', 'Close_a']].copy()
        spot.index = pd.to_datetime(spot.index).normalize()
        options = options.merge(spot, left_on='date', right_index=True, how='left')
        options['underlying_price'] = np.where(
            options['trading_session'].eq('after_market'), options['Close_a'], options['Close']
        )

        settlements = settlement_df.copy()
        settlements['settle_date'] = pd.to_datetime(settlements['settle_date'])
        iv_df = iv_calculator(
            options,
            model='bs',
            underlying_col='underlying_price',
            risk_free_rate=risk_free_rate,
            shape_options={'group_cols': ['date', 'contract_date', 'trading_session']},
            settlement_df=settlements,
            settlement_contract_col='contract',
            settlement_date_col='settle_date',
        )
        required_iv_columns = {'date', 'trading_session', 'SkewSlope', 'SkewSlope3'}
        missing_iv_columns = required_iv_columns - set(iv_df.columns)
        if missing_iv_columns:
            raise KeyError(f"IV calculation missing columns: {sorted(missing_iv_columns)}")

        skew = iv_df.groupby(['date', 'trading_session'])[['SkewSlope', 'SkewSlope3']].first().unstack('trading_session')
        skew.columns = [
            factor if session == 'position' else f'{factor}_a'
            for factor, session in skew.columns
        ]
        return self.merge_features(skew)

    @staticmethod
    def fetch_fear_greed(timeout: int = 15) -> pd.DataFrame:
        """Fetch the current CNN Fear & Greed history in a stable tabular shape."""
        import requests

        response = requests.get(
            'https://production.dataviz.cnn.io/index/fearandgreed/graphdata',
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/122.0 Safari/537.36'
                )
            },
            timeout=timeout,
        )
        response.raise_for_status()
        history = response.json().get('fear_and_greed_historical', {}).get('data', [])
        if not history:
            return pd.DataFrame(columns=['date', 'score', 'rating'])

        fear_greed = pd.DataFrame(history)
        fear_greed['date'] = pd.to_datetime(fear_greed['x'], unit='ms')
        return (
            fear_greed.rename(columns={'y': 'score'})[['date', 'score', 'rating']]
            .sort_values('date')
            .reset_index(drop=True)
        )

    def add_fear_greed(self, historical_df: pd.DataFrame, latest_df: pd.DataFrame) -> pd.DataFrame:
        """Merge historical and current CNN Fear & Greed observations, then lag one day."""
        historical = historical_df.copy()
        latest = latest_df.copy()
        latest = latest.rename(columns={'score': 'fear_greed', 'rating': 'fear_greed_emotion'})
        columns = ['fear_greed', 'fear_greed_emotion']

        def normalize(frame: pd.DataFrame) -> pd.DataFrame:
            if 'date' in frame.columns:
                frame['date'] = pd.to_datetime(frame['date']).dt.normalize()
                frame = frame.set_index('date')
            else:
                frame.index = pd.to_datetime(frame.index).normalize()
            for column in columns:
                if column not in frame:
                    frame[column] = pd.NA
            return frame[columns].loc[~frame.index.duplicated(keep='last')]

        historical = normalize(historical)
        latest = normalize(latest)
        combined = historical.reindex(historical.index.union(latest.index))
        combined.update(latest)
        combined['fear_greed'] = pd.to_numeric(combined['fear_greed'], errors='coerce')
        return self.merge_features(combined.sort_index().shift(1))

    def feature_status(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Summarize feature availability before analysis or backtesting."""
        if columns is None:
            columns = [
                'MOVE_open', 'SOX_open', 'Foreign_Opt_Signal_a',
                'SkewSlope', 'fear_greed', 'US_bond_5y',
            ]

        status = []
        for column in columns:
            if column not in self.df:
                status.append({'feature': column, 'available': False, 'missing': len(self.df), 'last_value_date': pd.NaT})
                continue

            values = self.df[column]
            valid_dates = values.dropna().index
            status.append({
                'feature': column,
                'available': True,
                'missing': int(values.isna().sum()),
                'last_value_date': valid_dates.max() if len(valid_dates) else pd.NaT,
            })
        return pd.DataFrame(status).set_index('feature')

    # Performance summaries -------------------------------------------------------
    def daily_ret(self):
        return plot.plot(self.df, ly=['cum_daily_ret', 'cum_daily_ret_a'], title='daily_return')

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

    # Factor diagnostics ----------------------------------------------------------
    def indicator_position_ret(self):
        df = self.df.copy()
        df['ind'] = df['daily_ret'].shift(1) + df['daily_ret_a'].shift(1) + df['daily_ret'].shift(2)
        df['ind'] = df['ind'].rolling(window=3).sum()
        df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df = df.sort_values(by='daily_ret').reset_index(drop=True)
        df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
        df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
        return plot.plot(df, ly='cum_demeaned_daily_ret_a', x='index', ry = 'daily_ret', sub_ly=['cum_daily_ret_a'])

    def indicator_gap_days(self, after_holiday: bool = False, *, sub_analysis: bool = False):
        df = self.df.copy()

        # Calendar
        df.index = pd.to_datetime(df.index)
        df['prev_date'] = df.index.to_series().shift(1)
        df['gap'] = (df.index.to_series() - df['prev_date']).dt.days
        
        if after_holiday:
            # 想看週一 (Post-holiday)
            # Day: 週一日盤 = Current (No shift)
            # Night: 週一夜盤 = 記在週二 (Next Row) -> shift(-1)
            df['daily_ret_a'] = df['daily_ret_a'].shift(-1)
        else:
            # 想看週五 (Pre-holiday)
            # Day: 週五日盤 = 記在週五 (Prev Row) -> shift(1)
            df['daily_ret'] = df['daily_ret'].shift(1)
            # Night: 週五夜盤 = 記在週一 (Current Row) -> No shift
            pass

        df = df.sort_values(by='gap').reset_index(drop=True)
        df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()

        if sub_analysis:
            df['pos_day'] = 0
            df = df.loc[df['gap'] > 2].copy()
            df['3_ma'] = df['Close_a'].rolling(3).mean()
            df['divergence'] = (df['Close_a'] / df['3_ma']) - 1
            foreign_opt_short = df['Foreign_Opt_Signal'] < -0.0035 # option 偏空
            condition_day = ~foreign_opt_short & (df['divergence'] > -0.05)
            df.loc[condition_day, 'pos_day'] = 1.0
            
            df = df.sort_values(by='pos_day').reset_index(drop=True)
            
            # 重算累積報酬 (因為 filter 過了)
            df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
            df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
            df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
            df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            
            return plot.plot(df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='pos_day', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title='gap_days_after_holiday')

        period = 'after_holiday' if after_holiday else 'before_holiday'
        return plot.plot(df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='gap', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'gap_days_{period}')

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

        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='TotalExchangeMarginMaintenance', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title='margin_maintenance_rate')

    def indicator_option_iv(self, sub_analysis: bool = False, trading_session: str = ['day', 'night']):
        df = self.df.copy()
        if trading_session == 'day':
            df = df.sort_values(by='SkewSlope_a').reset_index(drop=True)
            df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
            df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='SkewSlope_a', sub_ly=['cum_daily_ret'], title='option_iv_night')
        elif trading_session == 'night':
            df['SkewSlope'] = df['SkewSlope'].shift(1)
            df = df.sort_values(by='SkewSlope').reset_index(drop=True)
            df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
            df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret_a'], ry='SkewSlope', sub_ly=['cum_daily_ret_a'], title='option_iv_day')
    
    def indicator_opt_position(self, indicator: str = 'Foreign_Opt_Signal', trading_session: str = 'day' or 'night', sub_analysis: bool = False, time_series_analysis: bool = False):
        # Foreign_Opt_Signal, Dealer_Opt_Signal
        # signal 代表，每一塊錢中，有多少做多(> 0) / 做空(< 0)
        df = self.df.copy()
        if trading_session == 'day':
            df = df.sort_values(by=f'{indicator}_a').reset_index(drop=True)
            df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
            df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            
            if sub_analysis:
                df_l = df.loc[df[f'{indicator}_a'] < -0.0035]
                df_r = df.loc[df[f'{indicator}_a'] > -0.0035]
                for df in [df_l, df_r]:
                    df['3ma'] = df['Close_a'].rolling(window=3).mean()
                    df['divergence'] = (df['Close_a'] / df['3ma'] - 1)

                    df['MOVE_ind'] = (df['MOVE_close'] / df['MOVE_open']) - 1
                    
                    df['SOX_ind'] = (df['SOX_close'] / df['SOX_open']) - 1
                    df['SOX_ind'] = df['SOX_ind'].shift(1)

                    df = df.sort_values(by='MOVE_ind').reset_index(drop=True)
                    df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
                    df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
                    df['cum_daily_ret'] = df['daily_ret'].cumsum()
                    plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='MOVE_ind', sub_ly=['cum_daily_ret'], title=f'option_position_day_{indicator}_move_split')
            else:
                return plot.plot(df, ly=['cum_demeaned_daily_ret'], ry=f'{indicator}_a', sub_ly=['cum_daily_ret'], title=f'option_position_day_{indicator}')

        elif trading_session == 'night':
            df['pos_continue'] = df[indicator] + df[f'{indicator}_a'] + df[f'{indicator}'].shift(1)
            df['pos_continue'] = df['pos_continue'].shift(1)
            if time_series_analysis:
                df['signal'] = df['pos_continue'] > 0.012
                df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
                return plot.plot(df, ly=['cum_daily_ret_a'], ry='signal', ry_dashed=False, title=f'option_position_night_{indicator}_time_series')
            df = df.sort_values(by='pos_continue').reset_index(drop=True)
            df['demeaned_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
            df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            return plot.plot(df, ly=['cum_demeaned_daily_ret_a'], ry='pos_continue', sub_ly=['cum_daily_ret_a'], title=f'option_position_night_{indicator}')

    def check_volatility(self, window: int = 20):
        """
        事件變數 + 事件前 window 天波動度分佈（PDF）
        pos_continue_t = Foreign_Opt_Signal_t + Foreign_Opt_Signal_a_t + Foreign_Opt_Signal_{t-1}
        sig_lag_t      = pos_continue_{t-1}
        event_t        = (sig_lag_t >= 0.012)
        """
        df = self.df.copy()
        ret_col = 'daily_ret_a'

        # 訊號與事件
        df['pos_continue'] = df['Foreign_Opt_Signal'] + df['Foreign_Opt_Signal_a'] + df['Foreign_Opt_Signal'].shift(1)
        df['sig_lag'] = df['pos_continue'].shift(1)
        th = 0.012
        df['event'] = (df['sig_lag'] >= th)

        # 波動：事件日前 window 天（不偷看）
        df['vol'] = df[ret_col].rolling(window).std().shift(1)

        # 事件統計（頻率與 run-length）
        event_rate = df['event'].mean()
        runs = (df['event'] != df['event'].shift()).cumsum()
        run_lengths = df.groupby(runs)['event'].agg(['first', 'size'])
        event_runs = run_lengths[run_lengths['first'] == True]['size']
        avg_run = event_runs.mean() if not event_runs.empty else 0
        max_run = event_runs.max() if not event_runs.empty else 0
        first_20 = df.index[df['event']].to_series().head(20)
        print(f"[event] rate: {event_rate:.4f}, avg_run: {avg_run}, max_run: {max_run}")
        if not first_20.empty:
            print("[event] first 20 dates:")
            print(first_20)

        # Sanity check：確認 lag 與報酬無前視
        sample = df[['Foreign_Opt_Signal', 'Foreign_Opt_Signal_a', 'pos_continue', 'sig_lag', 'event', ret_col]].head(5)
        print("[sanity] sample (check shifts):")
        print(sample)

        sig_vol = df.loc[df['event'] == True, 'vol'].dropna()
        nonsig_vol = df.loc[df['event'] == False, 'vol'].dropna()

        if sig_vol.empty or nonsig_vol.empty:
            print("[warn] empty group")
            return None

        # PDF
        plot.plot_pdf(sig_vol.to_frame('vol'), col="vol", title=f"event vol (prev {window}d)")
        plot.plot_pdf(nonsig_vol.to_frame('vol'), col="vol", title=f"non-event vol (prev {window}d)")

        # 上漲/下跌機率
        res_rows = []
        for label, mask in [('event', df['event'] == True), ('non_event', df['event'] == False)]:
            sub = df.loc[mask, [ret_col]].dropna()
            if sub.empty:
                p_pos = p_neg = np.nan
                n = 0
                mean_neg = mean_pos = mean_abs_neg = mean_abs_pos = np.nan
            else:
                n = len(sub)
                p_pos = (sub[ret_col] > 0).mean()
                p_neg = (sub[ret_col] < 0).mean()
                neg = sub.loc[sub[ret_col] < 0, ret_col]
                pos = sub.loc[sub[ret_col] > 0, ret_col]
                mean_neg = neg.mean() if not neg.empty else np.nan
                mean_pos = pos.mean() if not pos.empty else np.nan
                mean_abs_neg = neg.abs().mean() if not neg.empty else np.nan
                mean_abs_pos = pos.abs().mean() if not pos.empty else np.nan
            res_rows.append({
                'group': label,
                'p_pos': p_pos, 'p_neg': p_neg, 'n': n,
                'mean_neg': mean_neg, 'mean_pos': mean_pos,
                'mean_abs_neg': mean_abs_neg, 'mean_abs_pos': mean_abs_pos,
            })
        freq_df = pd.DataFrame(res_rows)
        print("=== Up/Down Probability & Magnitude by Event Group ===")
        print(freq_df)

        # 左尾風險：多個門檻的 tail probability
        thresholds = [0.005, 0.01, 0.015]  # 0.5%, 1%, 1.5%（可依需要調整）
        tail_rows = []
        for x in thresholds:
            p_evt = (df.loc[df['event'] == True, ret_col] <= -x).mean()
            p_none = (df.loc[df['event'] == False, ret_col] <= -x).mean()
            tail_rows.append({
                'threshold_x': x,
                'p_event': p_evt,
                'p_nonevent': p_none,
                'diff': p_evt - p_none,
                'ratio': (p_evt / p_none) if p_none not in [0, np.nan] else np.nan,
            })
        tail_df = pd.DataFrame(tail_rows)
        print("=== Tail Probability (ret <= -x) by Event Group ===")
        print(tail_df)

        # 分位數比較（左尾更穩健）
        quants = [0.01, 0.05, 0.10]
        q_rows = []
        for qv in quants:
            evt_val = df.loc[df['event'] == True, ret_col].quantile(qv)
            none_val = df.loc[df['event'] == False, ret_col].quantile(qv)
            q_rows.append({
                'quantile': qv,
                'event_value': evt_val,
                'non_event_value': none_val,
                'diff': evt_val - none_val,
            })
        q_df = pd.DataFrame(q_rows)
        print("=== Quantile Comparison (event vs non_event) ===")
        print(q_df)

        # 連跌（二連跌機率），條件用 event_t
        res_pairs = []
        evt_mask = df['event'] == True
        none_mask = df['event'] == False
        for label, mask in [('event', evt_mask), ('non_event', none_mask)]:
            sub = df.loc[mask, [ret_col]].dropna()
            # 將當期與下一期配對
            pair_down = (sub[ret_col] < 0) & (sub[ret_col].shift(-1) < 0)
            n_pairs = pair_down.notna().sum() - 1  # 有效配對數
            p_2down = pair_down.mean()
            res_pairs.append({'group': label, 'p_2down': p_2down, 'n_pairs': n_pairs})
        pairs_df = pd.DataFrame(res_pairs)
        print("=== Two-day Down Probability (condition on event_t) ===")
        print(pairs_df)

        # conditional downside/upside vol within rolling window
        res_cond = []
        for label, mask in [('event', df['event'] == True), ('non_event', df['event'] == False)]:
            sub = df.loc[mask, ret_col]
            if sub.empty:
                res_cond.append({'group': label, 'downside_std': np.nan, 'upside_std': np.nan, 'ratio': np.nan})
                continue
            down_series = sub.rolling(window).apply(lambda x: x[x < 0].std() if (x < 0).any() else np.nan, raw=False)
            up_series = sub.rolling(window).apply(lambda x: x[x > 0].std() if (x > 0).any() else np.nan, raw=False)
            down_mean = down_series.dropna().mean()
            up_mean = up_series.dropna().mean()
            ratio = (down_mean / up_mean) if up_mean not in [0, np.nan] else np.nan
            res_cond.append({
                'group': label,
                'downside_std': down_mean,
                'upside_std': up_mean,
                'ratio': ratio,
            })
        cond_df = pd.DataFrame(res_cond)
        print("=== Conditional Downside/Upstate Vol (rolling window) ===")
        print(cond_df)

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
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='avg_margin_delta', sub_ly=['cum_daily_ret_a', 'cum_daily_ret', 'cum_ret'], title='margin_delta')

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

    def indicator_15ma_divergence(self, window: int = 15):
        temp_df = self.df.copy()
        temp_df[f'{window}_ma'] = temp_df['Close'].rolling(window=window).mean()
        temp_df['divergence'] = (temp_df['Close'] / temp_df[f'{window}_ma']) - 1
        temp_df['divergence'] = temp_df['divergence'].shift(1)
        temp_df = temp_df.dropna(subset=['divergence'])
        temp_df['demeaned_daily_ret_a'] = temp_df['daily_ret_a'] - temp_df['daily_ret_a'].mean()
        temp_df['demeaned_daily_ret'] = temp_df['daily_ret'] - temp_df['daily_ret'].mean()

        temp_df = temp_df.sort_values(by='divergence').reset_index(drop=True)
        temp_df['cum_demeaned_daily_ret_a'] = temp_df['demeaned_daily_ret_a'].cumsum()
        temp_df['cum_demeaned_daily_ret'] = temp_df['demeaned_daily_ret'].cumsum()
        temp_df['cum_daily_ret_a'] = temp_df['daily_ret_a'].cumsum()
        temp_df['cum_daily_ret'] = temp_df['daily_ret'].cumsum()
        return plot.plot(temp_df, ly=['cum_demeaned_daily_ret_a', 'cum_demeaned_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'{window}ma_divergence')

    def compare_ma_divergence_windows(
        self,
        windows: range | list[int] | tuple[int, ...],
        *,
        return_column: str = 'daily_ret',
        demean_return: bool = True,
        volatility_window: int | None = None,
        volatility_bins: int = 5,
        threshold_map: dict[int, float] | None = None,
        bin_percentile: float = 5,
    ) -> pd.DataFrame:
        """Show divergence-sorted daily returns in one heatmap.

        Each row is one MA window. Each cell is the mean return in a
        divergence-percentile bin. Set ``demean_return=False`` for raw returns.
        With ``volatility_window``, returns are first neutralized within
        quantiles of the preceding realized volatility, without look-ahead.
        Hover to read the raw divergence at any cell. Pass manually observed
        thresholds in ``threshold_map`` to mark them in red.
        """

        windows = list(windows)
        if not windows or any(window < 2 for window in windows):
            raise ValueError('windows must contain moving-average lengths of at least 2')
        if return_column not in self.df:
            raise KeyError(f"missing return column: {return_column}")
        if not 0 < bin_percentile <= 25:
            raise ValueError('bin_percentile must be greater than 0 and at most 25')
        if volatility_window is not None and volatility_window < 2:
            raise ValueError('volatility_window must be at least 2')
        if volatility_bins < 2:
            raise ValueError('volatility_bins must be at least 2')

        threshold_map = threshold_map or {}
        analysis_return = self.df[return_column].copy()
        if volatility_window is not None:
            realized_volatility = analysis_return.rolling(volatility_window).std().shift(1)
            volatility_frame = pd.DataFrame({'return': analysis_return, 'volatility': realized_volatility}).dropna()
            volatility_bucket = pd.qcut(volatility_frame['volatility'], q=volatility_bins, duplicates='drop')
            if volatility_bucket.nunique() < 2:
                raise ValueError('not enough volatility variation to create volatility bins')
            volatility_frame['return'] = volatility_frame['return'] - volatility_frame.groupby(volatility_bucket, observed=True)['return'].transform('mean')
            analysis_return = volatility_frame['return'].reindex(self.df.index)

        value_label = f'Demeaned {return_column}' if demean_return else return_column
        if volatility_window is not None:
            value_label = f'Volatility-controlled {value_label} ({volatility_window}D, {volatility_bins} bins)'
        value_title = value_label.replace(' ', '_')
        summary = []
        heatmap_values = []
        heatmap_divergence = []
        bin_edges = np.append(np.arange(0, 100, bin_percentile), 100.0)
        percentiles = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_labels = [f'{lower:g}% to {upper:g}%' for lower, upper in zip(bin_edges[:-1], bin_edges[1:])]
        fig = go.Figure()

        for window in windows:
            divergence = ((self.df['Close'] / self.df['Close'].rolling(window=window).mean()) - 1).shift(1)
            frame = pd.DataFrame({'divergence': divergence, 'return': analysis_return}).dropna()
            if frame.empty:
                summary.append({'window': window, 'observations': 0, 'minimum_return': np.nan, 'divergence_at_minimum': np.nan})
                heatmap_values.append(np.full(len(percentiles), np.nan))
                heatmap_divergence.append(np.full(len(percentiles), np.nan))
                continue

            frame = frame.sort_values('divergence').reset_index(drop=True)
            frame['display_return'] = frame['return'] - frame['return'].mean() if demean_return else frame['return']
            frame['divergence_percentile'] = (np.arange(len(frame)) + 1) / len(frame) * 100
            bin_means = []
            bin_divergence = []
            for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
                values = frame.loc[(frame['divergence_percentile'] > lower) & (frame['divergence_percentile'] <= upper)]
                bin_means.append(values['display_return'].mean())
                bin_divergence.append(values['divergence'].mean())
            heatmap_values.append(bin_means)
            heatmap_divergence.append(bin_divergence)
            minimum = frame.loc[frame['display_return'].idxmin()]
            summary.append({
                'window': window,
                'observations': len(frame),
                'minimum_return': minimum['display_return'],
                'divergence_at_minimum': minimum['divergence'],
            })

        fig.add_trace(
            go.Heatmap(
                x=percentiles,
                y=windows,
                z=np.asarray(heatmap_values),
                customdata=np.asarray(heatmap_divergence),
                text=np.tile(bin_labels, (len(windows), 1)),
                colorscale='RdBu',
                zmid=0,
                colorbar=dict(title=value_title.replace('_', '<br>', 1), tickformat='.1%'),
                hovertemplate=f'MA window: %{{y}}<br>Divergence percentile bin: %{{text}}<br>Raw divergence: %{{customdata:.4%}}<br>{value_title}: %{{z:.3%}}<extra></extra>',
            )
        )
        marker_x = []
        marker_y = []
        marker_text = []
        for row, window in enumerate(windows):
            if window not in threshold_map or not np.isfinite(heatmap_divergence[row]).any():
                continue
            threshold = threshold_map[window]
            marker_x.append(np.interp(threshold, heatmap_divergence[row], percentiles))
            marker_y.append(window)
            marker_text.append(f'{window} MA threshold: {threshold:.4%}')
        if marker_x:
            fig.add_trace(
                go.Scatter(
                    x=marker_x,
                    y=marker_y,
                    mode='markers',
                    marker=dict(color='#111111', size=10, symbol='x'),
                    text=marker_text,
                    hovertemplate='%{text}<extra></extra>',
                    name='observed threshold',
                )
            )
        fig.update_layout(
            title=f'MA divergence: {value_title} heatmap',
            template='plotly_white',
            height=620,
            hovermode='closest',
            showlegend=False,
        )
        fig.update_xaxes(title_text='Divergence percentile within each MA window', range=[0, 100])
        fig.update_yaxes(title_text='MA window')
        fig.show()

    def compare_ma_divergence_by_volatility(
        self,
        windows: range | list[int] | tuple[int, ...],
        *,
        return_column: str = 'daily_ret',
        volatility_window: int = 20,
        volatility_bins: int = 5,
        divergence_bin_percentile: float = 20,
    ) -> pd.DataFrame:
        """Compare every MA window, volatility regime, and divergence bin in one heatmap.

        Rows are MA windows. Columns are volatility regimes, each split into
        divergence-percentile bins ranked within that regime. Volatility uses
        only preceding returns, avoiding look-ahead.
        """
        windows = list(windows)
        if not windows or any(window < 2 for window in windows):
            raise ValueError('windows must contain moving-average lengths of at least 2')
        if return_column not in self.df:
            raise KeyError(f"missing return column: {return_column}")
        if volatility_window < 2:
            raise ValueError('volatility_window must be at least 2')
        if volatility_bins < 2:
            raise ValueError('volatility_bins must be at least 2')
        if not 0 < divergence_bin_percentile <= 50:
            raise ValueError('divergence_bin_percentile must be greater than 0 and at most 50')

        returns = self.df[return_column]
        realized_volatility = returns.rolling(volatility_window).std().shift(1)
        volatility_frame = pd.DataFrame({'return': returns, 'volatility': realized_volatility}).dropna()
        volatility_frame['volatility_group'] = pd.qcut(volatility_frame['volatility'], q=volatility_bins, labels=False, duplicates='drop')
        group_count = int(volatility_frame['volatility_group'].nunique())
        if group_count < 2:
            raise ValueError('not enough volatility variation to create volatility bins')

        bin_edges = np.append(np.arange(0, 100, divergence_bin_percentile), 100.0)
        bin_labels = [f'{lower:g}% to {upper:g}%' for lower, upper in zip(bin_edges[:-1], bin_edges[1:])]
        regime_labels = [f'Q{group + 1}' for group in range(group_count)]
        regime_labels[0] += ' low vol'
        regime_labels[-1] += ' high vol'
        column_labels = [f'{regime}\n{bin_label}' for regime in regime_labels for bin_label in bin_labels]
        mean_returns = np.full((len(windows), len(column_labels)), np.nan)
        mean_divergence = np.full((len(windows), len(column_labels)), np.nan)
        observations = np.zeros((len(windows), len(column_labels)), dtype=int)

        for row, window in enumerate(windows):
            divergence = ((self.df['Close'] / self.df['Close'].rolling(window=window).mean()) - 1).shift(1)
            frame = pd.DataFrame({'divergence': divergence}).join(volatility_frame).dropna()
            for group in range(group_count):
                regime = frame.loc[frame['volatility_group'] == group].copy()
                if regime.empty:
                    continue
                regime['divergence_percentile'] = regime['divergence'].rank(method='first') / len(regime) * 100
                for bin_index, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
                    column = group * len(bin_labels) + bin_index
                    values = regime.loc[(regime['divergence_percentile'] > lower) & (regime['divergence_percentile'] <= upper)]
                    mean_returns[row, column] = values['return'].mean()
                    mean_divergence[row, column] = values['divergence'].mean()
                    observations[row, column] = len(values)

        customdata = np.stack((mean_divergence, observations), axis=-1)
        column_positions = np.arange(len(column_labels))
        fig = go.Figure(
            go.Heatmap(
                x=column_positions,
                y=windows,
                z=mean_returns,
                customdata=customdata,
                text=np.tile(column_labels, (len(windows), 1)),
                colorscale='RdBu',
                zmid=0,
                xgap=1,
                ygap=1,
                colorbar=dict(title=f'Mean<br>{return_column}', tickformat='.1%'),
                hovertemplate='MA window: %{y}<br>%{text}<br>Mean raw divergence: %{customdata[0]:.4%}<br>Mean daily return: %{z:.3%}<br>Observations: %{customdata[1]}<extra></extra>',
            )
        )
        for group in range(1, group_count):
            fig.add_vline(x=group * len(bin_labels) - 0.5, line_color='#555555', line_width=1)
        fig.update_layout(
            title='MA divergence: daily_ret by volatility regime',
            template='plotly_white',
            height=max(650, len(windows) * 18 + 220),
        )
        fig.update_xaxes(
            title_text='Volatility regime / divergence percentile bin',
            tickmode='array',
            tickvals=column_positions,
            ticktext=column_labels,
            tickangle=-45,
        )
        fig.update_yaxes(title_text='MA window')
        columns = pd.MultiIndex.from_product([regime_labels, bin_labels], names=['volatility_regime', 'divergence_percentile_bin'])
        fig.show()
        return pd.DataFrame(mean_returns, index=windows, columns=columns)

    def indicator_night_price(self, sub_analysis=False):
        df = self.df.copy()
        df['3_ma'] = df['Close_a'].rolling(3).mean()
        df['divergence'] = (df['Close_a'] / df['3_ma']) - 1
        df = df.dropna(subset=['divergence'])
        if sub_analysis:
            df_l = df.loc[df['divergence'] < 0]
            df_r = df.loc[df['divergence'] >= 0]
            for df in [df_l, df_r]:
                df = df.sort_values(by='Foreign_Opt_Signal_a').reset_index(drop=True)
                df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
                df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
                df['cum_daily_ret'] = df['daily_ret'].cumsum()
                plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='Foreign_Opt_Signal_a', sub_ly=['cum_daily_ret'])
            return
        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df = df.sort_values(by='divergence').reset_index(drop=True)
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()
        return plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret'], title='night_price')

    def indicator_spread(self, window: int = 5):
        df = self.df.copy()
        df.dropna(subset=['spread_a'], inplace=True)
        df['next_ret'] = (df['Close_a'].shift(-window) - df['Close_a']) / df['Close_a']
        df['sum_spread'] = df['spread_a'].rolling(window=window).sum()
        df['sum_spread'] = df['sum_spread'].shift(1)
        # df.dropna(subset=['sum_spread'], inplace=True)
        df.sort_values(by='sum_spread', ignore_index=True, inplace=True)
        df['demeaned_daily_ret_a'] = df['next_ret'] - df['next_ret'].mean()
        df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df['cum_demeaned_daily_ret_a'] = df['demeaned_daily_ret_a'].cumsum()
        df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        df['cum_daily_ret_a'] = df['next_ret'].cumsum()
        return plot.plot(df, ly=['cum_demeaned_daily_ret_a'], ry='sum_spread', sub_ly=['cum_daily_ret_a'])

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

        ffill_col = ['US_bond_5y']
        for col in ffill_col:
            if temp_df[col].isna().sum() > 0:
                temp_df[col] = temp_df[col].ffill()

        # 長債
        temp_df['yield_shock'] = temp_df['US_bond_5y'] - temp_df['US_bond_5y'].shift(20)

        temp_df['yield_shock'] = temp_df['yield_shock'].shift(3)

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
            return plot.plot(temp_df, ly=['cum_demean_daily_ret_a', 'cum_demean_daily_ret'], ry=indicator, sub_ly=['cum_daily_ret_a', 'cum_daily_ret'], title=f'us_bond_{indicator}')

    def indicator_structural_weakness(self):
        """
        分析「市場結構轉弱」(Structural Weakness) 對績效的影響
        
        指標定義：
        1. 收盤位置 (CLV): (Close - Low) / (High - Low)
           - CLV < 0.4 代表收盤無力 (收在下半部)
        2. 反彈失敗 (Bounce Failure): 
           - T日上漲 (Ret > 0)
           - 但 T+1日收盤 < T-1日收盤 (漲勢曇花一現，馬上被吞噬)
           
        濾網條件 (Filter): 過去 5 天內
        - CLV < 0.4 的天數 >= 3
        - 下跌天數 (Ret < 0) >= 3
        - 發生過反彈失敗 >= 1 (Optional)
        
        驗證：當濾網觸發 (Signal ON) 時，後續的績效表現是否顯著較差？
        """
        df = self.df.copy()
        df['weakness'] = (df['Close_a'] < df['Close_a'].shift(2))
        df['pos_night'] = np.where(
            df['weakness'].shift(1),
            0,
            1
        )
        df['strat_ret'] = df['daily_ret_a'] * df['pos_night']
        df['cum_strat_ret'] = df['strat_ret'].cumsum()
        df['cum_bnh_ret'] = df['daily_ret_a'].cumsum()
        return plot.plot(df, ly=['cum_strat_ret', 'cum_bnh_ret'])

    def indicator_fear_greed(self, trading_session: str, time_series_analysis: bool = False):
        df = self.df.copy()
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / 'fear_greed_day.csv', index=True)

        if trading_session == 'night':
            df['fear_greed'] = df['fear_greed'].shift(1)
        elif trading_session == 'day':
            df['fear_greed'] = df['fear_greed']

        df['delta_fear_greed'] = df['fear_greed'] - df['fear_greed'].shift(1)
        df.dropna(subset=['daily_ret_a'], inplace=True)
        df['demean_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        df = df.sort_values(by='delta_fear_greed').reset_index(drop=False)
        df['cum_demean_daily_ret_a'] = df['demean_daily_ret_a'].cumsum()
        df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
        df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
        df['cum_daily_ret'] = df['daily_ret'].cumsum()
        OUTPUT_DIR.mkdir(exist_ok=True)
        df.to_csv(OUTPUT_DIR / 'fear_greed_night.csv', index=True)

        if trading_session == 'night':
            return plot.plot(df, ly=['cum_demean_daily_ret_a'], ry='delta_fear_greed', sub_ly=['cum_daily_ret_a'], title='fear_greed_night')
        elif trading_session == 'day':
            return plot.plot(df, ly=['cum_demean_daily_ret'], ry='delta_fear_greed', sub_ly=['cum_daily_ret'], title='fear_greed_day')

    def indicator_move(self, trading_session: str, sub_analysis: bool = False):
        df = self.df.copy()
        # ====== 計算指標 ======


        df['3_ma'] = df['Close_a'].rolling(window=3).mean()
        df['divergence'] = (df['Close_a'] / df['3_ma']) - 1

        df['ind'] = (df['MOVE_close'] / df['MOVE_open']) - 1
        df['MOVE_ma'] = 1
        df['MOVE_divergence'] = (df['MOVE_close'] / df['MOVE_ma']) - 1

        df['gap'] = (df['Close_a'] / df['Close'].shift(1)) - 1

        df['demean_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()

        if trading_session == 'night':
            df['MOVE_vol'] = (df['MOVE_high'] / df['MOVE_low']) - 1
            df['MOVE_vol'] = df['MOVE_vol'].shift(1)
            df['MOVE_divergence'] = df['MOVE_divergence'].shift(1)

            df['ind'] = df['ind'].shift(1)
            df = df.sort_values(by='MOVE_vol').reset_index(drop=True)
            df['cum_demean_daily_ret_a'] = df['demean_daily_ret_a'].cumsum()
            df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
            plot.plot(df, ly=['cum_demean_daily_ret_a'], ry='MOVE_vol', sub_ly=['cum_daily_ret_a'], title='move_night')

            print("==================\n==================")

            if sub_analysis:
                df_l, df_r = df.loc[df['MOVE_vol'] < 0.0145], df.loc[df['MOVE_vol'] >= 0.0145]

                lt = [df_l, df_r]

                for idx, df in enumerate(lt):
                    if idx == 0:
                        df = df.sort_values(by='ind').reset_index(drop=True)
                        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
                        df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
                        df['cum_daily_ret'] = df['daily_ret'].cumsum()
                        plot.plot(df, ly=['cum_demean_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'])

                        sub_df_ll, sub_df_lr = df.loc[df['MOVE_divergence'] < 0.0075].copy(), df.loc[df['MOVE_divergence'] >= 0.0075].copy()
                        # sub_df_ll, sub_df_lr = df.loc[df['SOX_ind'] < -0.0043].copy(), df.loc[df['SOX_ind'] >= -0.0043].copy()

                        # for sub_df in [sub_df_ll, sub_df_lr]:
                        #     sub_df = sub_df.sort_values(by='gap')
                        #     sub_df['demean_daily_ret'] = sub_df['daily_ret'] - sub_df['daily_ret'].mean()
                        #     sub_df['cum_demean_daily_ret'] = sub_df['demean_daily_ret'].cumsum()
                        #     sub_df['cum_daily_ret'] = sub_df['daily_ret'].cumsum()
                        #     plot.plot(sub_df, ly=['cum_demean_daily_ret'], ry='gap', sub_ly=['cum_daily_ret'])

                    elif idx == 1:
                        df = df.sort_values(by='ind').reset_index(drop=True)
                        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
                        df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
                        df['cum_daily_ret'] = df['daily_ret'].cumsum()
                        plot.plot(df, ly=['cum_demean_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'])

                        sub_df_rl, sub_df_rr = df.loc[df['MOVE_divergence'] < -0.0035].copy(), df.loc[df['MOVE_divergence'] >= -0.0035].copy()

                        # for sub_df in [sub_df_rl, sub_df_rr]:
                        #     sub_df = sub_df.sort_values(by='divergence')
                        #     sub_df['demean_daily_ret'] = sub_df['daily_ret'] - sub_df['daily_ret'].mean()
                        #     sub_df['cum_demean_daily_ret'] = sub_df['demean_daily_ret'].cumsum()
                        #     sub_df['cum_daily_ret'] = sub_df['daily_ret'].cumsum()
                        #     plot.plot(sub_df, ly=['cum_demean_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret'])

                    print("----------------------------")

        elif trading_session == 'day':
            df = df.sort_values(by='ind').reset_index(drop=True)
            df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
            df['cum_daily_ret'] = df['daily_ret'].cumsum()
            plot.plot(df, ly=['cum_demean_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'], title='move_day')

            print("==================\n==================")

            if sub_analysis:
                df_l = df.loc[df['ind'] < 0.0001].copy()
                df_r = df.loc[df['ind'] >= 0.0001].copy()

                lt = [df_l, df_r]

                for idx, df in enumerate(lt):
                    if idx == 0:
                        df = df.sort_values(by='SOX_ind')
                        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
                        df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
                        df['cum_daily_ret'] = df['daily_ret'].cumsum()
                        plot.plot(df, ly=['cum_demean_daily_ret'], ry='SOX_ind', sub_ly=['cum_daily_ret'])

                        sub_df_ll, sub_df_lr = df.loc[df['SOX_ind'] < 0.0075].copy(), df.loc[df['SOX_ind'] >= 0.0075].copy()
                        # sub_df_ll, sub_df_lr = df.loc[df['SOX_ind'] < -0.0043].copy(), df.loc[df['SOX_ind'] >= -0.0043].copy()

                        for sub_df in [sub_df_ll, sub_df_lr]:
                            sub_df = sub_df.sort_values(by='gap')
                            sub_df['demean_daily_ret'] = sub_df['daily_ret'] - sub_df['daily_ret'].mean()
                            sub_df['cum_demean_daily_ret'] = sub_df['demean_daily_ret'].cumsum()
                            sub_df['cum_daily_ret'] = sub_df['daily_ret'].cumsum()
                            plot.plot(sub_df, ly=['cum_demean_daily_ret'], ry='gap', sub_ly=['cum_daily_ret'])

                    elif idx == 1:
                        df = df.sort_values(by='Foreign_Opt_Signal_a')
                        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
                        df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
                        df['cum_daily_ret'] = df['daily_ret'].cumsum()
                        plot.plot(df, ly=['cum_demean_daily_ret'], ry='Foreign_Opt_Signal_a', sub_ly=['cum_daily_ret'])

                        sub_df_rl, sub_df_rr = df.loc[df['Foreign_Opt_Signal_a'] < -0.0035].copy(), df.loc[df['Foreign_Opt_Signal_a'] >= -0.0035].copy()

                        for sub_df in [sub_df_rl, sub_df_rr]:
                            sub_df = sub_df.sort_values(by='divergence')
                            sub_df['demean_daily_ret'] = sub_df['daily_ret'] - sub_df['daily_ret'].mean()
                            sub_df['cum_demean_daily_ret'] = sub_df['demean_daily_ret'].cumsum()
                            sub_df['cum_daily_ret'] = sub_df['daily_ret'].cumsum()
                            plot.plot(sub_df, ly=['cum_demean_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret'])

                    print("----------------------------")

    def indicator_sox(self, trading_session: str, sub_analysis: bool = False):
        df = self.df.copy()

        df['ind'] = (df['SOX_close'] / df['SOX_open']) - 1
        # df['ind'] = df['ind'].rolling(window=3).mean()
        df['ind'] = df['ind'].shift(1)
        # df['ind'] = (df['SOX_high'] - df['SOX_low']) / df['SOX_close'].shift(1)
        df = df.dropna(subset='ind')

        df['gap'] = (df['Open'] / df['Close'].shift(1)) - 1

        df['demean_daily_ret_a'] = df['daily_ret_a'] - df['daily_ret_a'].mean()
        df['demean_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        # df.dropna(subset=['ind'], inplace=True)
        if not sub_analysis:
            if trading_session == 'night':
                df['ind'] = df['ind'].shift(1)
                df = df.sort_values(by='ind').reset_index(drop=True)
                df['cum_demean_daily_ret_a'] = df['demean_daily_ret_a'].cumsum()
                df['cum_daily_ret_a'] = df['daily_ret_a'].cumsum()
                return plot.plot(df, ly=['cum_demean_daily_ret_a'], ry='ind', sub_ly=['cum_daily_ret_a'], title='sox_night')
            elif trading_session == 'day':
                df = df.sort_values(by='ind').reset_index(drop=True)
                df['cum_demean_daily_ret'] = df['demean_daily_ret'].cumsum()
                df['cum_daily_ret'] = df['daily_ret'].cumsum()
                mean_l = df.loc[df['ind'] < 0.0025, 'daily_ret'].mean()
                mean_r = df.loc[df['ind'] >= 0.0025, 'daily_ret'].mean()
                print(f"SOX ind threshold=0.0025\nmean_l={mean_l:.6f} | mean_r={mean_r:.6f}")
                return plot.plot(df, ly=['cum_demean_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'], title='sox_day')
        
        # # sub 1：SOX 優先
        # if sub_analysis:
        #     # df_l = df.loc[df['ind'] < 0.0025].copy()
        #     df_l = df.loc[df['ind'] < 0.0095].copy()
        #     df_ll = df_l.loc[df['Foreign_Opt_Signal_a'] < -0.002].copy()
        #     df_lr = df_l.loc[df['Foreign_Opt_Signal_a'] >= -0.002].copy()
        #     df_r = df.loc[df['ind'] >= 0.0025].copy()
        #     df_r = df.loc[df['ind'] >= 0.0025].copy()
        #     df_rl = df_r.loc[df['Foreign_Opt_Signal_a'] < -0.0035].copy()
        #     df_rr = df_r.loc[df['Foreign_Opt_Signal_a'] >= -0.0035].copy()
        #     i = 0
        #     for df in [df_l, df_r]:
        #         df = df.sort_values(by='Foreign_Opt_Signal_a').reset_index(drop=True)
        #         df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        #         df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        #         df['cum_daily_ret'] = df['daily_ret'].cumsum()
        #         plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='Foreign_Opt_Signal_a', sub_ly=['cum_daily_ret'])

        #         if i == 0:
        #             print("--- Sub-analysis: SOX ind < 0.0025 (Left) ---")
        #             lt = [df_ll, df_lr]
        #             threshold_l = -999
        #             threshold_r = -0.001
        #         elif i == 1:
        #             print("--- Sub-analysis: SOX ind >= 0.0025 (Right) ---")
        #             lt = [df_rl, df_rr]
        #             threshold_l = -0.01
        #             threshold_r = 0.02

        #         for idx, sub_df in enumerate(lt):
        #             sub_df['3_ma'] = sub_df['Close_a'].rolling(window=3).mean()
        #             sub_df['divergence'] = (sub_df['Close_a'] / sub_df['3_ma']) - 1

        #             sub_df = sub_df.sort_values(by='divergence').reset_index(drop=True)
        #             sub_df['demeaned_daily_ret'] = sub_df['daily_ret'] - sub_df['daily_ret'].mean()
        #             sub_df['cum_demeaned_daily_ret'] = sub_df['demeaned_daily_ret'].cumsum()
        #             sub_df['cum_daily_ret'] = sub_df['daily_ret'].cumsum()

        #             if idx == 0:
        #                 left_mask = sub_df['divergence'] < threshold_l
        #                 left_mean = sub_df.loc[left_mask, 'daily_ret'].mean()
        #                 right_mean = sub_df.loc[~left_mask, 'daily_ret'].mean()
        #                 print(
        #                     f"lt[0] threshold_l={threshold_l:.4f}\n"
        #                     f"left_mean={left_mean:.6f}"
        #                     f" | right_mean={right_mean:.6f}"
        #                 )
        #             else:
        #                 right_mask = sub_df['divergence'] > threshold_r
        #                 left_mean = sub_df.loc[~right_mask, 'daily_ret'].mean()
        #                 right_mean = sub_df.loc[right_mask, 'daily_ret'].mean()
        #                 print(
        #                     f"lt[1] threshold_r={threshold_r:.4f}\n"
        #                     f"left_mean={left_mean:.6f}"
        #                     f" | right_mean={right_mean:.6f}"
        #                 )

        #             plot.plot(sub_df, ly=['cum_demeaned_daily_ret'], ry='divergence', sub_ly=['cum_daily_ret'])
                
        #         i += 1
        #         print("=======================================================")
        #     print("")

        # # sub 2：foreign opt signal 優先
        # if sub_analysis:
        #     df_l = df.loc[df['Foreign_Opt_Signal_a'] < -0.0035].copy()
        #     df_ll = df_l.loc[df_l['ind'] < 0.007].copy()
        #     df_lr = df_l.loc[df_l['ind'] >= 0.007].copy()
        #     df_r = df.loc[df['Foreign_Opt_Signal_a'] >= -0.0035].copy()
        #     df_rl = df_r.loc[df_r['ind'] < -0.0035].copy()
        #     df_rr = df_r.loc[df_r['ind'] >= -0.0035].copy()

        #     i = 0

        #     for df in [df_l, df_r]:
        #         df = df.sort_values(by='ind').reset_index(drop=False)
        #         df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        #         df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        #         df['cum_daily_ret'] = df['daily_ret'].cumsum()
        #         plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'])
                
        #         if i == 0:
        #             print("--- Sub-analysis: Foreign Opt Signal_a < -0.0035 (Left) ---")
        #             lt = [df_ll, df_lr]
        #         else:
        #             print("--- Sub-analysis: Foreign Opt Signal_a >= -0.0035 (Right) ---")
        #             lt = [df_rl, df_rr]

        #         for idx, sub_df in enumerate(lt):
        #             sub_df['MOVE_ind'] = (sub_df['MOVE_high'] / sub_df['MOVE_low']) - 1

        #             sub_df = sub_df.sort_values(by='MOVE_ind').reset_index(drop=True)
        #             sub_df['demeaned_daily_ret'] = sub_df['daily_ret'] - sub_df['daily_ret'].mean()
        #             sub_df['cum_demeaned_daily_ret'] = sub_df['demeaned_daily_ret'].cumsum()
        #             sub_df['cum_daily_ret'] = sub_df['daily_ret'].cumsum()
        #             if idx == 0:
        #                 left_mask = sub_df['MOVE_ind'] < 0.0035
        #                 left_mean = sub_df.loc[left_mask, 'daily_ret'].mean()
        #                 right_mean = sub_df.loc[~left_mask, 'daily_ret'].mean()
        #                 print(
        #                     f"lt[0] MOVE_ind threshold=0.0035\n"
        #                     f"left_mean={left_mean:.6f}"
        #                     f" | right_mean={right_mean:.6f}"
        #                 )
        #             else:
        #                 right_mask = sub_df['MOVE_ind'] > 0.0035
        #                 left_mean = sub_df.loc[~right_mask, 'daily_ret'].mean()
        #                 right_mean = sub_df.loc[right_mask, 'daily_ret'].mean()
        #                 print(
        #                     f"lt[1] MOVE_ind threshold=0.0035\n"
        #                     f"left_mean={left_mean:.6f}"
        #                     f" | right_mean={right_mean:.6f}"
        #                 )
                    
        #             plot.plot(sub_df, ly=['cum_demeaned_daily_ret'], ry='MOVE_ind', sub_ly=['cum_daily_ret'])
        #         i += 1
        #         print("=======================================================")
        
        # # sub 3：加入下一天跳空
        # if sub_analysis:
        #     df_l, df_r = df.loc[df['ind'] < -0.006].copy(), df.loc[df['ind'] >= -0.006].copy()
        #     for df in [df_l, df_r]:
        #         df = df.sort_values(by='gap').reset_index(drop=True)
        #         df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        #         df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        #         df['cum_daily_ret'] = df['daily_ret'].cumsum()
        #         plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='gap', sub_ly=['cum_daily_ret'])
            
        # # sub 4：permutation test
        # if sub_analysis:
        #     n = len(df)
        #     df['ind'] = np.random.permutation(df['ind'].values)
        #     df_1, df_2, df_3 = df.iloc[: int(n * 0.33)], df.iloc[int(n * 0.33): int(n * 0.66)], df.iloc[int(n * 0.66): int(n-1)]
        #     for df in [df_1, df_2, df_3]:
        #         df = df.sort_values(by='ind').reset_index(drop=True)
        #         df['demeaned_daily_ret'] = df['daily_ret'] - df['daily_ret'].mean()
        #         df['cum_demeaned_daily_ret'] = df['demeaned_daily_ret'].cumsum()
        #         df['cum_daily_ret'] = df['daily_ret'].cumsum()
        #         plot.plot(df, ly=['cum_demeaned_daily_ret'], ry='ind', sub_ly=['cum_daily_ret'])

        # sub 5：bootstrap test
        if sub_analysis:
            def robustness_shift1(df, n_iter=200, frac=0.8):
                results = []

                for _ in range(n_iter):
                    sub = df.sample(frac=frac, replace=False).copy()
                    sub = sub.sort_values(by='ind').reset_index(drop=True)

                    q30 = sub['ind'].quantile(0.3)
                    q70 = sub['ind'].quantile(0.7)

                    low = sub.loc[sub['ind'] <= q30, 'daily_ret'].mean()
                    high = sub.loc[sub['ind'] >= q70, 'daily_ret'].mean()
                    spread = high - low

                    results.append({
                        'low_mean': low,
                        'high_mean': high,
                        'spread': spread
                    })

                return pd.DataFrame(results)
            robust = robustness_shift1(df)
            print(robust['spread'].describe())
            print((robust['spread'] > 0).mean())   # 正向比例

            return

    # Risk review and backtesting -------------------------------------------------
    def evaluate(
        self,
        config: StrategyConfig | None = None,
        *,
        point_version: bool = False,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a backtest result frame without rendering or writing an output file."""
        result = self._build_backtest_frame(point_version=point_version, config=config)
        if start is not None:
            result = result.loc[result.index >= pd.Timestamp(start)]
        if end is not None:
            result = result.loc[result.index <= pd.Timestamp(end)]
        return result

    def summarize_result(
        self,
        result: pd.DataFrame,
        *,
        return_column: str = 'strat_ret',
        point_version: bool = False,
    ) -> pd.Series:
        """Summarize an ``evaluate()`` result or an explicitly sliced validation period."""
        if return_column not in result:
            raise KeyError(f"result missing return column: {return_column}")
        return pd.Series(self._calculate_metrics(result[return_column].copy(), point_version=point_version))

    def _build_backtest_frame(
        self,
        point_version: bool,
        config: StrategyConfig | None = None,
    ) -> pd.DataFrame:
        """Build the result frame without rendering charts or writing files."""
        active_config = config or self.config
        df = StrategyEngine.calculate_factors(self.df)
        df = StrategyEngine.apply_positions(df, active_config)

        if point_version:
            df['daily_pnl_a'] = df['Open'] - df['Open_a']
            df['daily_pnl'] = df['Open_a'].shift(-1) - df['Open']
            df['strat_ret'] = (df['daily_pnl_a'] * df['pos_night']) + (df['daily_pnl'] * df['pos_day'])
            df['benchmark_ret'] = df['daily_pnl']
        else:
            df['daily_ret_a'] = (df['Open'] / df['Open_a']) - 1
            df['daily_ret'] = (df['Open_a'].shift(-1) / df['Open']) - 1
            df['strat_ret'] = (df['daily_ret_a'] * df['pos_night']) + (df['daily_ret'] * df['pos_day'])
            df['benchmark_ret'] = df['daily_ret_a']

        df['cum_strat'] = df['strat_ret'].cumsum()
        df['cum_bnh'] = df['benchmark_ret'].cumsum()
        return df

    def check_risk_events(self, filter_tech_signal: bool = False) -> pd.DataFrame:
        """Return a log that explains each active day-session position."""
        df = StrategyEngine.apply_positions(StrategyEngine.calculate_factors(self.df), self.config)
        move_below_threshold = df['MOVE_ind'] < self.config.move_threshold
        foreign_option_bearish = df['Foreign_Opt_Signal_a'] < self.config.foreign_option_threshold
        divergence_supports_long = df['divergence_v2'] < self.config.divergence_threshold

        log = pd.DataFrame(index=df.index)
        log['Factor'] = np.select(
            [move_below_threshold, foreign_option_bearish, ~divergence_supports_long],
            ['MOVE', 'Foreign option positioning', 'Day divergence'],
            default='No active signal',
        )
        log['Value'] = np.select(
            [move_below_threshold, foreign_option_bearish, ~divergence_supports_long],
            [df['MOVE_ind'], df['Foreign_Opt_Signal_a'], df['divergence_v2']],
            default=np.nan,
        )
        log['Action'] = np.select(
            [df['pos_day'].gt(0), df['pos_day'].lt(0)],
            ['Long day session', 'Short day session'],
            default='Flat',
        )
        log['Tech_Signal'] = np.where(df['pos_day'].gt(0), 'Buy', 'Neutral')
        log['Divergence'] = df['divergence_v2']
        log.index.name = 'Date'
        log = log.reset_index()

        if filter_tech_signal:
            log = log.loc[log['Action'].ne('Flat')]
        return log

    def backtest(
        self,
        risk_log: bool = False,
        point_version: bool = False,
        config: StrategyConfig | None = None,
        start: str | pd.Timestamp | None = None,
        end: str | pd.Timestamp | None = None,
    ):
        """Run the configured strategy, save the result frame, and display performance."""
        df = self.evaluate(config=config, point_version=point_version, start=start, end=end)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_DIR / 'backtest.csv', index=True)
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

        return plot.plot(df, ly=['cum_strat', 'cum_bnh'], ry_dashed=False)

    def show_factor_distributions(self, factors: list = None):
        """
        顯示策略使用之各項指標因子分佈情形 (Histograms & Statistics)
        factors: 指定要分析的因子清單 (預設為常用因子)
        """
        from plotly.subplots import make_subplots
        
        df = self.df.copy()
        # 確保因子已計算
        df = self._calculate_factors(df)
        
        # 定義主要因子 (若未指定 fetch default)
        if factors is None:
            factors = ['Foreign_Opt_Signal_a', 'MOVE_ind', 'SOX_ind', 'divergence']
        
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
