"""足強回踩進場策略回測。

核心假設：
1. 使用 1H K bar 計算 5/55/144/200MA。
2. 訊號在時K收盤確認，交易在下一根時K開盤成交。
3. 第一筆為縮量回踩試單，第二筆為放量突破加碼。
4. 族群主流/集中度若沒有外部資料，預設不啟用。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class BacktestConfig:
    initial_cash: float = 1_000_000
    max_positions: int = 10
    max_position_pct: float = 0.10
    first_entry_pct: float = 0.50
    second_entry_pct: float = 0.50

    ma_fast: int = 5
    ma_trend: int = 55
    ma_mid: int = 144
    ma_long: int = 200
    vol_ma_window: int = 20

    max_ma55_distance: float = 0.25
    support_tolerance: float = 0.03
    pullback_volume_ratio: float = 0.70
    breakout_volume_ratio: float = 1.10
    platform_lookback: int = 12
    platform_max_range: float = 0.08
    platform_max_net_change: float = 0.03

    black_volume_ratio: float = 2.50
    black_body_pct: float = 0.03

    fee_rate: float = 0.001425
    fee_discount: float = 0.28
    sell_tax_rate: float = 0.003


def load_hourly_kbars(
    kbar_dir: str | Path,
    start_date: str | None = None,
    end_date: str | None = None,
    stock_ids: Iterable[str] | None = None,
    skip_bad_files: bool = True,
) -> pd.DataFrame:
    """讀取每日一檔的 1H OHLCV parquet。"""
    kbar_dir = Path(kbar_dir)
    wanted_ids = None if stock_ids is None else {str(stock_id) for stock_id in stock_ids}
    frames: list[pd.DataFrame] = []
    bad_files: list[tuple[str, str]] = []

    files = sorted(kbar_dir.glob("*.parquet"))
    for path in files:
        date_str = path.stem
        if start_date is not None and date_str < start_date:
            continue
        if end_date is not None and date_str > end_date:
            continue

        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            if not skip_bad_files:
                raise
            bad_files.append((path.name, str(exc)))
            continue

        if wanted_ids is not None:
            df["stock_id"] = df["stock_id"].astype(str)
            df = df[df["stock_id"].isin(wanted_ids)]
        if not df.empty:
            frames.append(df)

    if not frames:
        hint = ""
        if bad_files:
            first_name, first_error = bad_files[0]
            hint = (
                f" First bad file: {first_name}: {first_error}. "
                "If these parquet files were written by a newer pyarrow, upgrade pyarrow "
                "or rebuild the files with the local environment."
            )
        raise ValueError(f"No kbar data loaded from {kbar_dir}.{hint}")

    kbars = pd.concat(frames, ignore_index=True)
    kbars = _normalize_kbars(kbars)
    if bad_files:
        kbars.attrs["bad_files"] = bad_files
    return kbars


def _normalize_kbars(kbars: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "hour", "stock_id", "open", "high", "low", "close", "volume"}
    missing = required - set(kbars.columns)
    if missing:
        raise ValueError(f"kbar missing required columns: {sorted(missing)}")

    kbars = kbars.copy()
    kbars["stock_id"] = kbars["stock_id"].astype(str)
    kbars["datetime"] = pd.to_datetime(kbars["date"].astype(str) + " " + kbars["hour"].astype(str))
    for col in ["open", "high", "low", "close", "volume"]:
        kbars[col] = pd.to_numeric(kbars[col], errors="coerce")
    kbars = kbars.dropna(subset=["datetime", "open", "high", "low", "close", "volume"])
    kbars = kbars[kbars["close"] > 0]
    return kbars.sort_values(["stock_id", "datetime"]).reset_index(drop=True)


def add_strategy_features(kbars: pd.DataFrame, config: BacktestConfig | None = None) -> pd.DataFrame:
    """加上 MA、量能、平台、進場篩選等欄位。"""
    config = config or BacktestConfig()
    df = _normalize_kbars(kbars)
    grouped = df.groupby("stock_id", group_keys=False)

    df["ma5"] = grouped["close"].transform(lambda s: s.rolling(config.ma_fast).mean())
    df["ma55"] = grouped["close"].transform(lambda s: s.rolling(config.ma_trend).mean())
    df["ma144"] = grouped["close"].transform(lambda s: s.rolling(config.ma_mid).mean())
    df["ma200"] = grouped["close"].transform(lambda s: s.rolling(config.ma_long).mean())
    df["vol_ma"] = grouped["volume"].transform(lambda s: s.rolling(config.vol_ma_window).mean())
    df["prev_close"] = grouped["close"].shift(1)

    df["platform_high"] = grouped["high"].transform(
        lambda s: s.rolling(config.platform_lookback).max().shift(1)
    )
    df["platform_low"] = grouped["low"].transform(
        lambda s: s.rolling(config.platform_lookback).min().shift(1)
    )
    df["platform_start_close"] = grouped["close"].shift(config.platform_lookback)
    df["platform_end_close"] = grouped["close"].shift(1)
    df["platform_range"] = (df["platform_high"] - df["platform_low"]) / df["close"]
    df["platform_net_change"] = (
        (df["platform_end_close"] - df["platform_start_close"]).abs()
        / df["platform_start_close"]
    )

    df["above_all_ma"] = (df["close"] > df["ma55"]) & (df["close"] > df["ma144"]) & (df["close"] > df["ma200"])
    df["ma55_rising"] = df["ma55"] > grouped["ma55"].shift(1)
    df["bullish_align"] = (df["ma5"] > df["ma55"]) & (df["ma55"] > df["ma144"]) & (df["ma144"] > df["ma200"])
    df["ma55_distance"] = (df["close"] - df["ma55"]) / df["ma55"]
    df["not_too_far"] = df["ma55_distance"] < config.max_ma55_distance
    black_body = (df["open"] - df["close"]) / df["open"]
    df["black_vol"] = (
        (df["volume"] > df["vol_ma"] * config.black_volume_ratio)
        & (df["close"] < df["open"])
        & (black_body > config.black_body_pct)
    )

    df["dist_ma5"] = (df["low"] - df["ma5"]).abs() / df["ma5"]
    df["dist_ma55"] = (df["low"] - df["ma55"]).abs() / df["ma55"]
    df["dist_platform"] = (df["low"] - df["platform_low"]).abs() / df["platform_low"]
    support_distances = df[["dist_ma5", "dist_ma55", "dist_platform"]].copy()
    df["support_distance"] = support_distances.min(axis=1)
    support_name = pd.Series(index=df.index, dtype="object")
    valid_support = support_distances.notna().any(axis=1)
    support_name.loc[valid_support] = support_distances.loc[valid_support].idxmin(axis=1)
    df["support_type"] = support_name.map(
        {
            "dist_ma5": "ma5",
            "dist_ma55": "ma55",
            "dist_platform": "platform",
        }
    )

    df["near_ma5"] = df["dist_ma5"] <= config.support_tolerance
    df["near_ma55"] = df["dist_ma55"] <= config.support_tolerance
    df["near_platform"] = df["dist_platform"] <= config.support_tolerance
    df["near_support"] = df["support_distance"] <= config.support_tolerance
    df["volume_contract"] = df["volume"] < df["vol_ma"] * config.pullback_volume_ratio
    df["stop_falling"] = (df["close"] >= df["open"]) | (df["close"] >= df["prev_close"])
    df["platform_range_ok"] = df["platform_range"] < config.platform_max_range
    df["platform_flat"] = df["platform_net_change"] < config.platform_max_net_change
    df["platform_tight"] = df["platform_range_ok"] & df["platform_flat"]

    df["trend_filter"] = (
        df["above_all_ma"]
        & df["ma55_rising"]
        & df["bullish_align"]
        & df["not_too_far"]
        & ~df["black_vol"]
    )
    df["first_entry_signal"] = (
        df["trend_filter"]
        & df["near_support"]
        & df["volume_contract"]
        & df["stop_falling"]
        & df["platform_tight"]
        & (df["close"] >= df["ma55"])
    )
    df["confirm_signal"] = (
        df["trend_filter"]
        & (df["close"] > df["platform_high"])
        & (df["volume"] > df["vol_ma"] * config.breakout_volume_ratio)
        & (df["close"] > df["open"])
    )
    df["entry_signal"] = df["first_entry_signal"]
    df["raw_stop_signal"] = df["close"] < df["ma55"]
    return df


def build_signal_df(
    kbars: pd.DataFrame,
    config: BacktestConfig | None = None,
    signal_col: str = "entry_signal",
    include_all_rows: bool = False,
) -> pd.DataFrame:
    """建立研究用訊號 df。

    預設只回傳 signal_col 為 True 的列；include_all_rows=True 時回傳完整
    feature df，方便在 notebook 檢查每個中間條件。
    """
    features = add_strategy_features(kbars, config=config)

    if include_all_rows:
        return features.reset_index(drop=True)

    if signal_col not in features.columns:
        raise ValueError(f"signal_col not found: {signal_col}")

    signal_df = features[features[signal_col].fillna(False)].copy()
    return signal_df.reset_index(drop=True)


def build_signal_df_from_directory(
    kbar_dir: str | Path,
    start_date: str,
    end_date: str | None = None,
    stock_ids: Iterable[str] | None = None,
    config: BacktestConfig | None = None,
    signal_col: str = "entry_signal",
    include_all_rows: bool = False,
    output_file: str | Path | None = None,
) -> pd.DataFrame:
    """從 1H parquet 目錄直接建立訊號 df。"""
    kbars = load_hourly_kbars(
        kbar_dir=kbar_dir,
        start_date=start_date,
        end_date=end_date,
        stock_ids=stock_ids,
    )
    signal_df = build_signal_df(
        kbars=kbars,
        config=config,
        signal_col=signal_col,
        include_all_rows=include_all_rows,
    )
    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        signal_df.to_parquet(output_file, index=False)
    return signal_df


def run_signal_backtest(
    feature_df: pd.DataFrame,
    config: BacktestConfig | None = None,
    entry_col: str = "entry_signal",
    exit_col: str = "raw_stop_signal",
    initial_nav: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用完整 feature df 回測第一筆進場訊號。

    訊號在當根 K 的收盤後才成立，因此進、出場都使用下一根 K 的開盤價。
    回傳：
    1. ``trades_df``：實際成交的進出場明細。
    2. ``equity_df``：同時持有股票等權重的組合報酬與淨值。

    此函式用於驗證訊號本身：每個持有中的標的每根 K 都等權重，
    不套用初始資金、持股檔數限制或第二次加碼。
    """
    config = config or BacktestConfig()
    required = {"stock_id", "datetime", "open", "close", entry_col, exit_col}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"feature_df missing required columns: {sorted(missing)}")
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")

    df = feature_df.copy()
    df["stock_id"] = df["stock_id"].astype(str)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["stock_id", "datetime"]).reset_index(drop=True)

    position_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []
    for _, stock_df in df.groupby("stock_id", sort=False):
        positions, trades = _simulate_signal_position(
            stock_df=stock_df,
            entry_col=entry_col,
            exit_col=exit_col,
            config=config,
        )
        position_frames.append(positions)
        if not trades.empty:
            trade_frames.append(trades)

    position_df = pd.concat(position_frames, ignore_index=True)
    position_df = position_df.sort_values(["datetime", "stock_id"]).reset_index(drop=True)
    trades_df = (
        pd.concat(trade_frames, ignore_index=True)
        if trade_frames
        else pd.DataFrame(columns=["datetime", "stock_id", "side", "price", "reason"])
    )
    trades_df = trades_df.sort_values(["datetime", "stock_id", "side"]).reset_index(drop=True)

    active = position_df[position_df["active_bar"]].copy()
    if active.empty:
        equity_df = pd.DataFrame(columns=["datetime", "active_positions", "portfolio_return", "nav"])
        return trades_df, equity_df

    equity_df = (
        active.groupby("datetime", as_index=False)
        .agg(
            active_positions=("stock_id", "nunique"),
            portfolio_return=("strategy_return", "mean"),
        )
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    equity_df["nav"] = initial_nav * (1 + equity_df["portfolio_return"]).cumprod()
    return trades_df, equity_df


def _simulate_signal_position(
    stock_df: pd.DataFrame,
    entry_col: str,
    exit_col: str,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依單一標的的收盤訊號，模擬下一根開盤成交。"""
    stock_df = stock_df.copy().reset_index(drop=True)
    entry_signal = stock_df[entry_col].astype("boolean").fillna(False)
    exit_signal = stock_df[exit_col].astype("boolean").fillna(False)
    stock_df["entry_order"] = entry_signal.shift(1, fill_value=False).astype(bool)
    stock_df["exit_order"] = exit_signal.shift(1, fill_value=False).astype(bool)
    stock_df["prev_close"] = stock_df["close"].shift(1)

    holding = False
    positions: list[bool] = []
    active_bars: list[bool] = []
    entered_flags: list[bool] = []
    exited_flags: list[bool] = []
    returns: list[float] = []
    events: list[dict] = []
    buy_cost = config.fee_rate * config.fee_discount
    sell_cost = buy_cost + config.sell_tax_rate

    for row in stock_df.itertuples(index=False):
        open_price = float(row.open)
        close_price = float(row.close)
        prev_close = row.prev_close
        entered = False
        exited = False
        active_bar = False

        # 已持倉時，停損指令在本根開盤成交；同根不重複進場。
        if holding and row.exit_order:
            active_bar = True
            gross_return = open_price / prev_close - 1 if pd.notna(prev_close) else 0.0
            returns.append((1 + gross_return) * (1 - sell_cost) - 1)
            holding = False
            exited = True
            events.append(
                {
                    "datetime": row.datetime,
                    "stock_id": row.stock_id,
                    "side": "sell",
                    "price": open_price,
                    "reason": exit_col,
                }
            )
        elif not holding and row.entry_order:
            active_bar = True
            gross_return = close_price / open_price - 1
            returns.append((1 + gross_return) * (1 - buy_cost) - 1)
            holding = True
            entered = True
            events.append(
                {
                    "datetime": row.datetime,
                    "stock_id": row.stock_id,
                    "side": "buy",
                    "price": open_price,
                    "reason": entry_col,
                }
            )
        elif holding:
            active_bar = True
            gross_return = close_price / prev_close - 1 if pd.notna(prev_close) else 0.0
            returns.append(gross_return)
        else:
            returns.append(0.0)

        positions.append(holding)
        active_bars.append(active_bar)
        entered_flags.append(entered)
        exited_flags.append(exited)

    stock_df["position"] = positions
    stock_df["active_bar"] = active_bars
    stock_df["strategy_return"] = returns
    stock_df["entered"] = entered_flags
    stock_df["exited"] = exited_flags
    return stock_df, pd.DataFrame(events)


def run_backtest(
    kbars: pd.DataFrame,
    config: BacktestConfig | None = None,
    sector_map: pd.DataFrame | None = None,
    max_positions_per_sector: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """執行多股票投組回測，回傳交易明細、權益曲線、摘要。"""
    config = config or BacktestConfig()
    df = add_strategy_features(kbars, config)
    df = _attach_sector(df, sector_map)
    df = df.sort_values(["datetime", "stock_id"]).reset_index(drop=True)

    cash = float(config.initial_cash)
    positions: dict[str, dict] = {}
    pending_orders: list[dict] = []
    trades: list[dict] = []
    equity_rows: list[dict] = []
    last_close: dict[str, float] = {}

    for ts, bars in df.groupby("datetime", sort=True):
        bars_by_id = {row.stock_id: row for row in bars.itertuples(index=False)}

        still_pending: list[dict] = []
        for order in pending_orders:
            row = bars_by_id.get(order["stock_id"])
            if row is None:
                still_pending.append(order)
                continue
            cash = _execute_order(order, row, ts, cash, positions, trades, config)
        pending_orders = still_pending

        for row in bars.itertuples(index=False):
            stock_id = row.stock_id
            last_close[stock_id] = float(row.close)
            pos = positions.get(stock_id)

            if pos is not None:
                if (
                    bool(row.raw_stop_signal)
                    or float(row.close) < pos["defense_price"]
                    or float(row.close) < pos["platform_low"]
                ):
                    pending_orders.append({"action": "sell", "stock_id": stock_id, "reason": "stop"})
                    continue

                if (not pos["added"]) and bool(row.confirm_signal):
                    target_value = config.initial_cash * config.max_position_pct * config.second_entry_pct
                    pending_orders.append(
                        {
                            "action": "buy_add",
                            "stock_id": stock_id,
                            "cash_value": target_value,
                            "reason": "confirm_breakout",
                        }
                    )
                continue

            if not bool(row.first_entry_signal):
                continue
            if len(positions) >= config.max_positions:
                continue
            if _sector_is_full(row, positions, max_positions_per_sector):
                continue

            target_value = config.initial_cash * config.max_position_pct * config.first_entry_pct
            pending_orders.append(
                {
                    "action": "buy_first",
                    "stock_id": stock_id,
                    "cash_value": target_value,
                    "reason": "pullback_contract",
                    "defense_price": min(float(row.platform_low), float(row.ma55)),
                    "platform_low": float(row.platform_low),
                    "platform_high": float(row.platform_high),
                    "sector": getattr(row, "sector", None),
                }
            )

        equity = cash + sum(pos["shares"] * last_close.get(stock_id, pos["avg_price"]) for stock_id, pos in positions.items())
        equity_rows.append(
            {
                "datetime": ts,
                "cash": cash,
                "position_value": equity - cash,
                "equity": equity,
                "positions": len(positions),
            }
        )

    if positions:
        final_ts = df["datetime"].max()
        for stock_id, pos in list(positions.items()):
            price = last_close.get(stock_id, pos["avg_price"])
            cash = _sell_position(stock_id, price, final_ts, "final_close", cash, positions, trades, config)
        equity_rows.append(
            {
                "datetime": final_ts,
                "cash": cash,
                "position_value": 0.0,
                "equity": cash,
                "positions": 0,
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    summary = summarize_performance(equity_df, trades_df, config)
    summary["config"] = asdict(config)
    return trades_df, equity_df, summary


def _attach_sector(df: pd.DataFrame, sector_map: pd.DataFrame | None) -> pd.DataFrame:
    if sector_map is None:
        df["sector"] = None
        return df
    sector = sector_map.copy()
    if not {"stock_id", "sector"}.issubset(sector.columns):
        raise ValueError("sector_map must include stock_id and sector columns")
    sector["stock_id"] = sector["stock_id"].astype(str)
    return df.merge(sector[["stock_id", "sector"]].drop_duplicates(), on="stock_id", how="left")


def _sector_is_full(row, positions: dict[str, dict], max_positions_per_sector: int | None) -> bool:
    if max_positions_per_sector is None:
        return False
    sector = getattr(row, "sector", None)
    if sector is None or pd.isna(sector):
        return False
    current = sum(1 for pos in positions.values() if pos.get("sector") == sector)
    return current >= max_positions_per_sector


def _execute_order(
    order: dict,
    row,
    ts: pd.Timestamp,
    cash: float,
    positions: dict[str, dict],
    trades: list[dict],
    config: BacktestConfig,
) -> float:
    stock_id = order["stock_id"]
    price = float(row.open)
    if price <= 0:
        return cash

    if order["action"] == "sell":
        return _sell_position(stock_id, price, ts, order["reason"], cash, positions, trades, config)

    if stock_id in positions and order["action"] != "buy_add":
        return cash

    cash_value = min(float(order["cash_value"]), cash)
    shares = int(cash_value // price)
    if shares <= 0:
        return cash

    fee = shares * price * config.fee_rate * config.fee_discount
    total_cost = shares * price + fee
    if total_cost > cash:
        shares = int(cash // (price * (1 + config.fee_rate * config.fee_discount)))
        if shares <= 0:
            return cash
        fee = shares * price * config.fee_rate * config.fee_discount
        total_cost = shares * price + fee

    cash -= total_cost
    pos = positions.get(stock_id)
    if pos is None:
        positions[stock_id] = {
            "shares": shares,
            "avg_price": price,
            "cost": shares * price + fee,
            "entry_time": ts,
            "defense_price": float(order["defense_price"]),
            "platform_low": float(order["platform_low"]),
            "platform_high": float(order["platform_high"]),
            "sector": order.get("sector"),
            "added": False,
        }
    else:
        old_shares = pos["shares"]
        new_shares = old_shares + shares
        pos["avg_price"] = (pos["avg_price"] * old_shares + price * shares) / new_shares
        pos["shares"] = new_shares
        pos["cost"] += shares * price + fee
        pos["added"] = True

    trades.append(
        {
            "datetime": ts,
            "stock_id": stock_id,
            "side": "buy",
            "reason": order["reason"],
            "price": price,
            "shares": shares,
            "fee": fee,
            "tax": 0.0,
            "cash_after": cash,
        }
    )
    return cash


def _sell_position(
    stock_id: str,
    price: float,
    ts: pd.Timestamp,
    reason: str,
    cash: float,
    positions: dict[str, dict],
    trades: list[dict],
    config: BacktestConfig,
) -> float:
    pos = positions.pop(stock_id, None)
    if pos is None or price <= 0:
        return cash

    shares = pos["shares"]
    gross = shares * price
    fee = gross * config.fee_rate * config.fee_discount
    tax = gross * config.sell_tax_rate
    proceeds = gross - fee - tax
    cash += proceeds
    pnl = proceeds - pos["cost"]

    trades.append(
        {
            "datetime": ts,
            "stock_id": stock_id,
            "side": "sell",
            "reason": reason,
            "price": price,
            "shares": shares,
            "fee": fee,
            "tax": tax,
            "pnl": pnl,
            "return_pct": pnl / pos["cost"] if pos["cost"] else 0.0,
            "cash_after": cash,
        }
    )
    return cash


def summarize_performance(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict:
    config = config or BacktestConfig()
    if equity.empty:
        return {
            "initial_cash": config.initial_cash,
            "final_equity": config.initial_cash,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
        }

    eq = equity.copy()
    eq["datetime"] = pd.to_datetime(eq["datetime"])
    eq = eq.sort_values("datetime")
    final_equity = float(eq["equity"].iloc[-1])
    total_return = final_equity / config.initial_cash - 1
    drawdown = eq["equity"] / eq["equity"].cummax() - 1

    sells = trades[trades["side"] == "sell"] if not trades.empty else pd.DataFrame()
    wins = int((sells["pnl"] > 0).sum()) if "pnl" in sells else 0
    closed = int(len(sells))

    return {
        "initial_cash": config.initial_cash,
        "final_equity": final_equity,
        "total_return": total_return,
        "max_drawdown": float(drawdown.min()),
        "trade_count": int(len(trades)),
        "closed_positions": closed,
        "win_rate": wins / closed if closed else None,
        "avg_closed_return": float(sells["return_pct"].mean()) if closed else None,
    }


def run_from_directory(
    kbar_dir: str | Path,
    start_date: str,
    end_date: str | None = None,
    output_dir: str | Path | None = None,
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """方便 notebook/腳本直接從 1H 目錄跑回測。"""
    config = config or BacktestConfig()
    kbars = load_hourly_kbars(kbar_dir, start_date=start_date, end_date=end_date)
    trades, equity, summary = run_backtest(kbars, config=config)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_dir / "trades.csv", index=False)
        equity.to_csv(output_dir / "equity.csv", index=False)
        pd.Series(summary).drop(labels=["config"], errors="ignore").to_csv(output_dir / "summary.csv")

    return trades, equity, summary
