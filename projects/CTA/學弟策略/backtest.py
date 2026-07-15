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


# 0050 在 2025-06-18 執行 1 拆 4；值為分割後每一份舊受益權單位對應的份數。
TW50_SPLIT_EVENTS: dict[str, float] = {"2025-06-18": 4.0}


@dataclass
class BacktestConfig:
    max_positions: int = 10  # 完整資金回測最多同時持有檔數。
    max_position_pct: float = 0.10  # 單一標的完整倉位占初始資金的比例。
    first_entry_pct: float = 0.50  # 第一筆試單占單一完整倉位的比例。
    second_entry_pct: float = 0.50  # 第二筆加碼占單一完整倉位的比例。

    ma_fast: int = 5  # 快均線週期，以 1H K 為單位。
    ma_trend: int = 55  # 趨勢支撐均線週期，以 1H K 為單位。
    ma_mid: int = 144  # 中期均線週期，以 1H K 為單位。
    ma_long: int = 200  # 長期均線週期，以 1H K 為單位。
    vol_ma_window: int = 20  # 成交量均線週期，以 1H K 為單位。

    max_ma55_distance: float = 0.25  # 收盤價最高可比 55MA 高出的比例。
    support_tolerance: float = 0.03  # low 距 MA 或平台低點多近可視為回踩支撐。
    pullback_volume_ratio: float = 0.70  # 回踩量縮門檻：volume 小於 vol_ma 的比例。
    breakout_volume_ratio: float = 1.10  # 確認突破量增門檻：volume 至少為 vol_ma 的比例。
    breakout_setup_lookback: int = 30  # 55MA 突破後，允許等待回踩進場的最大 1H K 根數。
    min_runup_from_ma55: float = 0.05  # 突破後拉升的最低幅度，相對於突破當根 55MA。
    min_lift_volume_ratio: float = 1.20  # 拉升段最大量能至少為 vol_ma 的倍數。
    max_lift_volume_ratio: float = 3.00  # 拉升段最大量能不可高於 vol_ma 的倍數。
    platform_lookback: int = 12  # 判斷整理平台所回看的前幾根 1H K。
    platform_max_range: float = 0.08  # 平台最高低差占現價的最大比例。
    platform_max_net_change: float = 0.03  # 平台起訖收盤價淨變動的最大比例。
    previous_low_lookback: int = 12  # 前低停損採用的前幾根 1H K 最低價。
    defense_price_buffer: float = 0.00  # 綜合防守價相對支撐最低點的額外下方緩衝比例。

    black_volume_ratio: float = 2.50  # 爆量長黑的量能門檻：volume 為 vol_ma 的倍數。
    black_body_pct: float = 0.03  # 爆量長黑的最小實體跌幅：(open-close)/open。

    sector_return_window: int = 20  # 產業強弱採用的日報酬累積窗口，單位為交易日。
    sector_top_quantile: float = 0.20  # 產業近期期報酬排名位於前幾成時視為主流。
    min_sector_members: int = 5  # 產業當日最少可交易成員數，避免小族群排名失真。
    require_sector_main: bool = True  # True 時，entry_signal 必須屬於近期市場主流產業。

    fee_rate: float = 0.001425  # 台股單邊牌告手續費率。
    fee_discount: float = 0.28  # 手續費實際收取比例；0.28 表示收牌告費率的 28%。
    sell_tax_rate: float = 0.003  # 賣出時證交稅率。


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
    df["prev_ma55"] = grouped["close"].transform(
        lambda s: s.rolling(config.ma_trend).mean().shift(1)
    )
    df["volume_multiple"] = df["volume"] / df["vol_ma"]

    # 突破後的拉升只使用當前 K 以前已知的價格與量能，避免偷看未來。
    df["breakout_55"] = (df["close"] > df["ma55"]) & (df["prev_close"] <= df["prev_ma55"])
    df["breakout_cycle"] = df.groupby("stock_id")["breakout_55"].cumsum()
    df["breakout_ma55"] = df["ma55"].where(df["breakout_55"])
    df["breakout_ma55"] = df.groupby("stock_id")["breakout_ma55"].ffill()
    breakout_groups = df.groupby(["stock_id", "breakout_cycle"])
    df["post_breakout_high"] = breakout_groups["high"].transform(
        lambda s: s.cummax().shift(1)
    )
    df["post_breakout_volume_multiple"] = breakout_groups["volume_multiple"].transform(
        lambda s: s.cummax().shift(1)
    )
    df["breakout_within_lookback"] = df.groupby("stock_id")["breakout_55"].transform(
        lambda s: s.shift(1).rolling(config.breakout_setup_lookback, min_periods=1).max()
    ).fillna(False).astype(bool)
    df["runup_from_ma55"] = df["post_breakout_high"] / df["breakout_ma55"] - 1
    df["runup_ok"] = df["runup_from_ma55"] >= config.min_runup_from_ma55
    df["lift_volume_ok"] = df["post_breakout_volume_multiple"].between(
        config.min_lift_volume_ratio,
        config.max_lift_volume_ratio,
    )
    df["breakout_setup_ok"] = (
        df["breakout_within_lookback"]
        & df["runup_ok"]
        & df["lift_volume_ok"]
    )

    df["platform_high"] = grouped["high"].transform(
        lambda s: s.rolling(config.platform_lookback).max().shift(1)
    )
    df["platform_low"] = grouped["low"].transform(
        lambda s: s.rolling(config.platform_lookback).min().shift(1)
    )
    df["platform_start_close"] = grouped["close"].shift(config.platform_lookback)
    df["platform_end_close"] = grouped["close"].shift(1)
    df["previous_low"] = grouped["low"].transform(
        lambda s: s.rolling(config.previous_low_lookback).min().shift(1)
    )
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
    df["platform_support"] = df["support_type"] == "platform"
    df["platform_setup_ok"] = ~df["platform_support"] | df["platform_tight"]
    df["support_level"] = df["ma5"].where(
        df["support_type"] == "ma5",
        df["ma55"].where(df["support_type"] == "ma55", df["platform_low"]),
    )
    df.loc[df["support_type"].isna(), "support_level"] = float("nan")
    df["support_held"] = df["close"] >= df["support_level"]
    df["defense_price"] = df[["ma55", "platform_low", "previous_low"]].min(axis=1)
    df["defense_price"] *= 1 - config.defense_price_buffer

    df["trend_filter"] = (
        df["above_all_ma"]
        & df["ma55_rising"]
        & df["bullish_align"]
        & df["not_too_far"]
        & ~df["black_vol"]
    )
    df["first_entry_signal"] = (
        df["trend_filter"]
        & df["breakout_setup_ok"]
        & df["near_support"]
        & df["volume_contract"]
        & df["stop_falling"]
        & df["support_held"]
        & df["platform_setup_ok"]
        & (df["close"] >= df["ma55"])
    )
    df["confirm_signal"] = (
        df["trend_filter"]
        & (df["close"] > df["platform_high"])
        & (df["volume"] > df["vol_ma"] * config.breakout_volume_ratio)
        & (df["close"] > df["open"])
    )
    df["entry_signal"] = df["first_entry_signal"]
    df["stop_ma55"] = df["close"] < df["ma55"]
    df["stop_platform"] = df["close"] < df["platform_low"]
    df["stop_previous_low"] = df["close"] < df["previous_low"]
    df["stop_defense"] = df["close"] < df["defense_price"]
    df["raw_stop_signal"] = (
        df["stop_ma55"]
        | df["stop_platform"]
        | df["stop_previous_low"]
        | df["stop_defense"]
    )
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


def build_sector_map(stock_info: pd.DataFrame) -> pd.DataFrame:
    """從 FinMind 股票基本資料建立 stock_id 對產業分類表。"""
    required = {"stock_id", "industry_category"}
    missing = required - set(stock_info.columns)
    if missing:
        raise ValueError(f"stock_info missing required columns: {sorted(missing)}")

    sector_map = stock_info.copy()
    if "type" in sector_map.columns:
        sector_map = sector_map[sector_map["type"].isin(["twse", "tpex"])]
    sector_map["stock_id"] = sector_map["stock_id"].astype(str)
    sector_map["sector"] = sector_map["industry_category"].astype("string").str.strip()
    sector_map = sector_map[
        sector_map["sector"].notna()
        & ~sector_map["sector"].isin(["", "大盤", "Index", "所有證券"])
    ]
    return sector_map[["stock_id", "sector"]].drop_duplicates("stock_id").reset_index(drop=True)


def build_sector_daily_features(
    daily_prices: pd.DataFrame,
    sector_map: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """以日 K 等權報酬建立產業強弱訊號，並延後一個交易日使用。"""
    config = config or BacktestConfig()
    if not 0 < config.sector_top_quantile <= 1:
        raise ValueError("sector_top_quantile must be between 0 and 1")

    daily_required = {"date", "stock_id", "close"}
    sector_required = {"stock_id", "sector"}
    missing_daily = daily_required - set(daily_prices.columns)
    missing_sector = sector_required - set(sector_map.columns)
    if missing_daily:
        raise ValueError(f"daily_prices missing required columns: {sorted(missing_daily)}")
    if missing_sector:
        raise ValueError(f"sector_map missing required columns: {sorted(missing_sector)}")

    daily = daily_prices[["date", "stock_id", "close"]].copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily["stock_id"] = daily["stock_id"].astype(str)
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["date", "stock_id", "close"])
    daily = daily[daily["close"] > 0].sort_values(["stock_id", "date"])

    sectors = sector_map[["stock_id", "sector"]].copy()
    sectors["stock_id"] = sectors["stock_id"].astype(str)
    sectors = sectors.dropna(subset=["sector"]).drop_duplicates("stock_id")
    daily = daily.merge(sectors, on="stock_id", how="inner")
    daily["stock_return"] = daily.groupby("stock_id")["close"].pct_change(fill_method=None)

    sector_daily = (
        daily.groupby(["date", "sector"], as_index=False)
        .agg(
            sector_return=("stock_return", "mean"),
            sector_members=("stock_id", "nunique"),
        )
        .sort_values(["sector", "date"])
        .reset_index(drop=True)
    )
    sector_daily["sector_return_20d"] = sector_daily.groupby("sector")["sector_return"].transform(
        lambda s: (1 + s).rolling(config.sector_return_window).apply(lambda x: x.prod(), raw=True) - 1
    )
    eligible = sector_daily["sector_members"] >= config.min_sector_members
    sector_daily["sector_rank_pct"] = (
        sector_daily["sector_return_20d"].where(eligible).groupby(sector_daily["date"]).rank(pct=True)
    )
    sector_daily["sector_main"] = (
        eligible
        & (sector_daily["sector_return_20d"] > 0)
        & (sector_daily["sector_rank_pct"] >= 1 - config.sector_top_quantile)
    )

    signal_cols = ["sector_return_20d", "sector_rank_pct", "sector_members", "sector_main"]
    for col in signal_cols:
        sector_daily[col] = sector_daily.groupby("sector")[col].shift(1)
    sector_daily["sector_main"] = (
        sector_daily["sector_main"].astype("boolean").fillna(False).astype(bool)
    )
    return sector_daily[["date", "sector", *signal_cols]]


def add_sector_features(
    feature_df: pd.DataFrame,
    daily_prices: pd.DataFrame,
    sector_map: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> pd.DataFrame:
    """將無偷看未來的日 K 產業強弱訊號併入 hourly feature df。"""
    config = config or BacktestConfig()
    required = {"date", "stock_id", "first_entry_signal"}
    missing = required - set(feature_df.columns)
    if missing:
        raise ValueError(f"feature_df missing required columns: {sorted(missing)}")

    features = feature_df.copy()
    features["stock_id"] = features["stock_id"].astype(str)
    features["_sector_date"] = pd.to_datetime(features["date"]).dt.normalize()
    sectors = sector_map[["stock_id", "sector"]].copy()
    sectors["stock_id"] = sectors["stock_id"].astype(str)
    sectors = sectors.drop_duplicates("stock_id")
    features = features.drop(columns=["sector"], errors="ignore").merge(
        sectors,
        on="stock_id",
        how="left",
    )

    sector_daily = build_sector_daily_features(daily_prices, sectors, config=config)
    features = features.merge(
        sector_daily,
        left_on=["_sector_date", "sector"],
        right_on=["date", "sector"],
        how="left",
        suffixes=("", "_sector"),
    )
    features = features.drop(columns=["_sector_date", "date_sector"])
    features["sector_main"] = features["sector_main"].astype("boolean").fillna(False).astype(bool)
    features["sector_filter"] = features["sector_main"] if config.require_sector_main else True
    features["entry_signal"] = features["first_entry_signal"] & features["sector_filter"]
    return features.sort_values(["stock_id", "datetime"]).reset_index(drop=True)


def run_signal_backtest(
    feature_df: pd.DataFrame,
    config: BacktestConfig | None = None,
    entry_col: str = "entry_signal",
    exit_col: str | None = None,
    initial_nav: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用完整 feature df 回測第一筆進場訊號。

    訊號在當根 K 的收盤後才成立，因此進、出場都使用下一根 K 的開盤價。
    進場訊號成立時會凍結平台低點、前低與防守價；持倉後以這些固定
    價位及當下的 55MA 判斷停損。``exit_col`` 可額外指定自訂出場訊號；
    預設為 None，不使用會隨 rolling window 改變的 ``raw_stop_signal``。
    回傳：
    1. ``trades_df``：實際成交的進出場明細。
    2. ``equity_df``：同時持有股票等權重的組合報酬與淨值。

    此函式用於驗證訊號本身：每個持有中的標的每根 K 都等權重，
    不套用初始資金、持股檔數限制或第二次加碼。
    """
    config = config or BacktestConfig()
    required = {
        "stock_id",
        "datetime",
        "open",
        "close",
        "ma55",
        "platform_low",
        "previous_low",
        "defense_price",
        entry_col,
    }
    if exit_col is not None:
        required.add(exit_col)
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


def run_buy_and_hold_benchmark(
    price_df: pd.DataFrame,
    stock_id: str = "0050",
    config: BacktestConfig | None = None,
    initial_nav: float = 1.0,
    split_events: dict[str, float] | None = None,
) -> pd.DataFrame:
    """回測單一標的 buy-and-hold，預設為 0050。

    第一根可交易時 K 的開盤買進，最後一根時 K 的收盤賣出。手續費與
    證交稅沿用 ``BacktestConfig``。``split_events`` 的 key 為除權日、value
    為持有份數的倍數；0050 預設套用 2025-06-18 的 1 拆 4。回傳的
    ``benchmark_nav`` 可與策略 ``equity_df['nav']`` 直接比較。
    """
    config = config or BacktestConfig()
    required = {"stock_id", "datetime", "open", "close"}
    missing = required - set(price_df.columns)
    if missing:
        raise ValueError(f"price_df missing required columns: {sorted(missing)}")
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")

    benchmark = price_df.copy()
    benchmark["stock_id"] = benchmark["stock_id"].astype(str)
    benchmark = benchmark[benchmark["stock_id"] == str(stock_id)].copy()
    benchmark["datetime"] = pd.to_datetime(benchmark["datetime"])
    benchmark["open"] = pd.to_numeric(benchmark["open"], errors="coerce")
    benchmark["close"] = pd.to_numeric(benchmark["close"], errors="coerce")
    benchmark = benchmark.dropna(subset=["datetime", "open", "close"])
    benchmark = benchmark[(benchmark["open"] > 0) & (benchmark["close"] > 0)]
    benchmark = benchmark.sort_values("datetime").drop_duplicates("datetime")
    if benchmark.empty:
        raise ValueError(f"No valid price bars found for benchmark stock_id={stock_id}")

    if split_events is None:
        split_events = TW50_SPLIT_EVENTS if str(stock_id) == "0050" else {}
    split_factors = pd.Series(split_events, dtype=float)
    split_factors.index = pd.to_datetime(split_factors.index).normalize()
    benchmark["split_factor"] = benchmark["datetime"].dt.normalize().map(split_factors).fillna(1.0)
    if (benchmark["split_factor"] <= 0).any():
        raise ValueError("split event factors must be positive")

    buy_cost = config.fee_rate * config.fee_discount
    sell_cost = buy_cost + config.sell_tax_rate
    benchmark["prev_close"] = benchmark["close"].shift(1)
    benchmark["benchmark_return"] = (
        benchmark["close"] * benchmark["split_factor"] / benchmark["prev_close"] - 1
    )
    benchmark.loc[benchmark.index[0], "benchmark_return"] = (
        benchmark.iloc[0]["close"] / benchmark.iloc[0]["open"] * (1 - buy_cost) - 1
    )
    benchmark.loc[benchmark.index[-1], "benchmark_return"] = (
        (1 + benchmark.iloc[-1]["benchmark_return"]) * (1 - sell_cost) - 1
    )
    benchmark["benchmark_nav"] = initial_nav * (1 + benchmark["benchmark_return"]).cumprod()
    benchmark["benchmark_stock_id"] = str(stock_id)
    return benchmark[["datetime", "benchmark_stock_id", "benchmark_return", "benchmark_nav"]]


def add_buy_and_hold_benchmark(
    equity_df: pd.DataFrame,
    benchmark_price_file: str | Path,
    stock_id: str = "0050",
    config: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """讀取 benchmark 日 K，並以當日最近可得 NAV 對齊策略的時 K NAV。"""
    required = {"datetime", "nav"}
    missing = required - set(equity_df.columns)
    if missing:
        raise ValueError(f"equity_df missing required columns: {sorted(missing)}")
    if equity_df.empty:
        raise ValueError("equity_df is empty; cannot align a benchmark")

    benchmark_price_file = Path(benchmark_price_file)
    if not benchmark_price_file.exists():
        raise FileNotFoundError(f"benchmark price file not found: {benchmark_price_file}")

    equity = equity_df.copy()
    equity["datetime"] = pd.to_datetime(equity["datetime"])
    start = equity["datetime"].min().normalize()
    end = equity["datetime"].max().normalize()

    prices = pd.read_parquet(benchmark_price_file)
    if "date" in prices.columns and "datetime" not in prices.columns:
        prices = prices.rename(columns={"date": "datetime"})
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    prices = prices[prices["datetime"].between(start, end)]
    benchmark_df = run_buy_and_hold_benchmark(
        prices,
        stock_id=stock_id,
        config=config,
    )
    aligned = pd.merge_asof(
        equity.sort_values("datetime"),
        benchmark_df[["datetime", "benchmark_nav"]].sort_values("datetime"),
        on="datetime",
        direction="backward",
    )
    return aligned, benchmark_df


def plot_strategy_performance(equity_df: pd.DataFrame):
    """建立策略與 benchmark NAV、回撤的 Plotly 圖及績效摘要。"""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError("plot_strategy_performance requires plotly") from exc

    required = {"datetime", "nav", "benchmark_nav"}
    missing = required - set(equity_df.columns)
    if missing:
        raise ValueError(f"equity_df missing required columns: {sorted(missing)}")

    plot_df = equity_df.dropna(subset=["nav", "benchmark_nav"]).copy()
    plot_df = plot_df.sort_values("datetime")
    if plot_df.empty:
        raise ValueError("No overlapping strategy and benchmark NAV observations")

    strategy_drawdown = plot_df["nav"] / plot_df["nav"].cummax() - 1
    benchmark_drawdown = plot_df["benchmark_nav"] / plot_df["benchmark_nav"].cummax() - 1
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.68, 0.32],
        vertical_spacing=0.08,
        subplot_titles=("Strategy vs 0050 Buy & Hold", "Drawdown"),
    )
    fig.add_trace(go.Scatter(x=plot_df["datetime"], y=plot_df["nav"], name="Strategy"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=plot_df["datetime"], y=plot_df["benchmark_nav"], name="0050 Buy & Hold"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["datetime"],
            y=strategy_drawdown,
            name="Strategy Drawdown",
            fill="tozeroy",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["datetime"],
            y=benchmark_drawdown,
            name="0050 Drawdown",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(height=720, hovermode="x unified", legend_title_text="")

    summary = pd.DataFrame(
        {
            "strategy_total_return": [plot_df["nav"].iloc[-1] / plot_df["nav"].iloc[0] - 1],
            "0050_total_return": [
                plot_df["benchmark_nav"].iloc[-1] / plot_df["benchmark_nav"].iloc[0] - 1
            ],
            "strategy_max_drawdown": [strategy_drawdown.min()],
            "0050_max_drawdown": [benchmark_drawdown.min()],
        }
    )
    return fig, summary


def build_performance_metrics(
    equity_df: pd.DataFrame,
    benchmark_df: pd.DataFrame,
    annualization: int = 252,
    risk_free_rate: float = 0.0,
    initial_nav: float = 1.0,
) -> pd.DataFrame:
    """以每日 NAV 報酬比較策略與 benchmark 的完整績效指標。"""
    if annualization <= 0:
        raise ValueError("annualization must be positive")
    if initial_nav <= 0:
        raise ValueError("initial_nav must be positive")
    required_equity = {"datetime", "nav"}
    required_benchmark = {"datetime", "benchmark_nav"}
    missing_equity = required_equity - set(equity_df.columns)
    missing_benchmark = required_benchmark - set(benchmark_df.columns)
    if missing_equity:
        raise ValueError(f"equity_df missing required columns: {sorted(missing_equity)}")
    if missing_benchmark:
        raise ValueError(
            f"benchmark_df missing required columns: {sorted(missing_benchmark)}"
        )

    equity = equity_df[["datetime", "nav"]].copy()
    equity["date"] = pd.to_datetime(equity["datetime"]).dt.normalize()
    strategy_daily = equity.groupby("date", as_index=True)["nav"].last()

    benchmark = benchmark_df[["datetime", "benchmark_nav"]].copy()
    benchmark["date"] = pd.to_datetime(benchmark["datetime"]).dt.normalize()
    benchmark_daily = benchmark.groupby("date", as_index=True)["benchmark_nav"].last()

    start = max(strategy_daily.index.min(), benchmark_daily.index.min())
    end = min(strategy_daily.index.max(), benchmark_daily.index.max())
    benchmark_daily = benchmark_daily.loc[start:end]
    strategy_daily = strategy_daily.reindex(benchmark_daily.index).ffill()
    daily_nav = pd.DataFrame(
        {"Strategy": strategy_daily, "Benchmark": benchmark_daily}
    ).dropna()
    if len(daily_nav) < 2:
        raise ValueError("Need at least two overlapping daily NAV observations")

    return pd.DataFrame(
        [
            _performance_metrics_from_nav(
                daily_nav["Strategy"], annualization, risk_free_rate, initial_nav
            ),
            _performance_metrics_from_nav(
                daily_nav["Benchmark"], annualization, risk_free_rate, initial_nav
            ),
        ],
        index=["Strategy", "Benchmark"],
    )


def _performance_metrics_from_nav(
    nav: pd.Series,
    annualization: int,
    risk_free_rate: float,
    initial_nav: float,
) -> dict[str, float | int | pd.Timedelta | None]:
    nav = nav.dropna().astype(float)
    daily_returns = nav.pct_change()
    daily_returns.iloc[0] = nav.iloc[0] / initial_nav - 1
    daily_returns = daily_returns.dropna()
    total_return = nav.iloc[-1] / initial_nav - 1
    calendar_days = max((nav.index[-1] - nav.index[0]).days, 1)
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / calendar_days) - 1
    volatility = daily_returns.std(ddof=1) * annualization**0.5 if len(daily_returns) > 1 else 0.0
    daily_risk_free = (1 + risk_free_rate) ** (1 / annualization) - 1
    sharpe = (
        (daily_returns.mean() - daily_risk_free) / daily_returns.std(ddof=1) * annualization**0.5
        if len(daily_returns) > 1 and daily_returns.std(ddof=1) > 0
        else None
    )

    drawdown = nav / nav.cummax() - 1
    max_drawdown = drawdown.min()
    max_dd_duration = _max_drawdown_duration(drawdown)
    wins = daily_returns[daily_returns > 0]
    losses = daily_returns[daily_returns < 0]
    nonzero = daily_returns[daily_returns != 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    win_rate = len(wins) / len(nonzero) if len(nonzero) else None
    avg_win = wins.mean() if len(wins) else None
    avg_loss = losses.mean() if len(losses) else None
    odds = avg_win / abs(avg_loss) if avg_win is not None and avg_loss is not None else None
    expected_return = daily_returns.mean() if len(daily_returns) else None
    kelly = (
        win_rate - (1 - win_rate) / odds
        if win_rate is not None and odds is not None and odds > 0
        else None
    )
    return {
        "Total Return": total_return,
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Max DD Duration": max_dd_duration,
        "Profit Factor": profit_factor,
        "Win Rate": win_rate,
        "Odds": odds,
        "Avg Win": avg_win,
        "Avg Loss": avg_loss,
        "Avg Return (Exp)": expected_return,
        "Kelly": kelly,
    }


def _max_drawdown_duration(drawdown: pd.Series) -> int:
    """回傳最長連續水下期的日曆天數。"""
    underwater_start: pd.Timestamp | None = None
    max_duration = 0
    for timestamp, value in drawdown.items():
        if value < 0 and underwater_start is None:
            underwater_start = timestamp
        elif value >= 0 and underwater_start is not None:
            max_duration = max(max_duration, (timestamp - underwater_start).days)
            underwater_start = None
    if underwater_start is not None:
        max_duration = max(max_duration, (drawdown.index[-1] - underwater_start).days)
    return max_duration


def _simulate_signal_position(
    stock_df: pd.DataFrame,
    entry_col: str,
    exit_col: str | None,
    config: BacktestConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """依單一標的的收盤訊號，模擬下一根開盤成交。"""
    stock_df = stock_df.copy().reset_index(drop=True)
    entry_signal = stock_df[entry_col].astype("boolean").fillna(False)
    exit_signal = (
        stock_df[exit_col].astype("boolean").fillna(False)
        if exit_col is not None
        else pd.Series(False, index=stock_df.index, dtype=bool)
    )
    stock_df["entry_order"] = entry_signal.shift(1, fill_value=False).astype(bool)
    stock_df["custom_exit_order"] = exit_signal.shift(1, fill_value=False).astype(bool)
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
    frozen_stops: dict[str, float | None] | None = None
    pending_exit_reason: str | None = None

    def valid_price(value: object) -> float | None:
        price = pd.to_numeric(value, errors="coerce")
        return float(price) if pd.notna(price) and price > 0 else None

    for row_pos, row in enumerate(stock_df.itertuples(index=False)):
        open_price = float(row.open)
        close_price = float(row.close)
        prev_close = row.prev_close
        entered = False
        exited = False
        active_bar = False

        # 前一根收盤觸發的停損，在本根開盤成交；同根不重複進場。
        if holding and pending_exit_reason is not None:
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
                    "reason": pending_exit_reason,
                }
            )
            frozen_stops = None
            pending_exit_reason = None
        elif not holding and row.entry_order:
            active_bar = True
            gross_return = close_price / open_price - 1
            returns.append((1 + gross_return) * (1 - buy_cost) - 1)
            holding = True
            entered = True
            signal_row = stock_df.iloc[row_pos - 1]
            frozen_stops = {
                "platform_low": valid_price(signal_row.platform_low),
                "previous_low": valid_price(signal_row.previous_low),
                "defense_price": valid_price(signal_row.defense_price),
            }
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

        # 停損在本根收盤才確認，下一根開盤成交。平台／前低／防守價
        # 固定取進場訊號那根 K 棒的值，只有 55MA 會隨時間更新。
        if holding and pending_exit_reason is None:
            if bool(row.custom_exit_order):
                pending_exit_reason = exit_col or "custom_exit"
            elif pd.notna(row.ma55) and close_price < float(row.ma55):
                pending_exit_reason = "stop_ma55"
            elif frozen_stops and frozen_stops["platform_low"] is not None and close_price < frozen_stops["platform_low"]:
                pending_exit_reason = "stop_platform"
            elif frozen_stops and frozen_stops["previous_low"] is not None and close_price < frozen_stops["previous_low"]:
                pending_exit_reason = "stop_previous_low"
            elif frozen_stops and frozen_stops["defense_price"] is not None and close_price < frozen_stops["defense_price"]:
                pending_exit_reason = "stop_defense"

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
    daily_prices: pd.DataFrame | None = None,
    max_positions_per_sector: int | None = None,
    initial_cash: float = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """執行兩筆進場的完整資金回測，回傳交易明細、權益曲線、摘要。

    若 ``require_sector_main=True``，必須同時提供 ``sector_map`` 與
    ``daily_prices``，才能在進場前套用無偷看未來的產業主流篩選。
    """
    config = config or BacktestConfig()
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    df = add_strategy_features(kbars, config)
    if config.require_sector_main:
        if sector_map is None or daily_prices is None:
            raise ValueError(
                "sector_map and daily_prices are required when require_sector_main=True"
            )
        df = add_sector_features(df, daily_prices, sector_map, config=config)
    else:
        df = _attach_sector(df, sector_map)
    df = df.sort_values(["datetime", "stock_id"]).reset_index(drop=True)

    cash = float(initial_cash)
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
                stop_reason = _frozen_stop_reason(row, pos)
                if stop_reason is not None:
                    pending_orders.append(
                        {"action": "sell", "stock_id": stock_id, "reason": stop_reason}
                    )
                    continue

                if (not pos["added"]) and bool(row.confirm_signal):
                    target_value = initial_cash * config.max_position_pct * config.second_entry_pct
                    pending_orders.append(
                        {
                            "action": "buy_add",
                            "stock_id": stock_id,
                            "cash_value": target_value,
                            "reason": "confirm_breakout",
                        }
                    )
                continue

            if not bool(row.entry_signal):
                continue
            pending_entries = [
                order for order in pending_orders if order["action"] == "buy_first"
            ]
            pending_entry_ids = {order["stock_id"] for order in pending_entries}
            if stock_id in pending_entry_ids:
                continue
            if len(positions) + len(pending_entry_ids) >= config.max_positions:
                continue
            if _sector_is_full(
                row,
                positions,
                pending_entries,
                max_positions_per_sector,
            ):
                continue

            target_value = initial_cash * config.max_position_pct * config.first_entry_pct
            pending_orders.append(
                {
                    "action": "buy_first",
                    "stock_id": stock_id,
                    "cash_value": target_value,
                    "reason": "pullback_contract",
                    "defense_price": float(row.defense_price),
                    "platform_low": float(row.platform_low),
                    "previous_low": float(row.previous_low),
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
    summary = summarize_performance(equity_df, trades_df, initial_cash=initial_cash)
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


def _sector_is_full(
    row,
    positions: dict[str, dict],
    pending_entries: list[dict],
    max_positions_per_sector: int | None,
) -> bool:
    if max_positions_per_sector is None:
        return False
    sector = getattr(row, "sector", None)
    if sector is None or pd.isna(sector):
        return False
    current = sum(1 for pos in positions.values() if pos.get("sector") == sector)
    current += sum(1 for order in pending_entries if order.get("sector") == sector)
    return current >= max_positions_per_sector


def _frozen_stop_reason(row, position: dict) -> str | None:
    """以當根收盤確認停損；只有 55MA 隨時間更新。"""
    close = float(row.close)
    if close < float(row.ma55):
        return "stop_ma55"
    if close < position["platform_low"]:
        return "stop_platform"
    if close < position["previous_low"]:
        return "stop_previous_low"
    if close < position["defense_price"]:
        return "stop_defense"
    return None


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
            "previous_low": float(order["previous_low"]),
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
    initial_cash: float = 1_000_000,
) -> dict:
    if initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if equity.empty:
        return {
            "initial_cash": initial_cash,
            "final_equity": initial_cash,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
        }

    eq = equity.copy()
    eq["datetime"] = pd.to_datetime(eq["datetime"])
    eq = eq.sort_values("datetime")
    final_equity = float(eq["equity"].iloc[-1])
    total_return = final_equity / initial_cash - 1
    drawdown = eq["equity"] / eq["equity"].cummax() - 1

    sells = trades[trades["side"] == "sell"] if not trades.empty else pd.DataFrame()
    wins = int((sells["pnl"] > 0).sum()) if "pnl" in sells else 0
    closed = int(len(sells))

    return {
        "initial_cash": initial_cash,
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
    sector_map: pd.DataFrame | None = None,
    daily_prices: pd.DataFrame | None = None,
    max_positions_per_sector: int | None = None,
    initial_cash: float = 1_000_000,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """方便 notebook/腳本直接從 1H 目錄跑回測。"""
    config = config or BacktestConfig()
    kbars = load_hourly_kbars(kbar_dir, start_date=start_date, end_date=end_date)
    trades, equity, summary = run_backtest(
        kbars,
        config=config,
        sector_map=sector_map,
        daily_prices=daily_prices,
        max_positions_per_sector=max_positions_per_sector,
        initial_cash=initial_cash,
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(output_dir / "trades.csv", index=False)
        equity.to_csv(output_dir / "equity.csv", index=False)
        pd.Series(summary).drop(labels=["config"], errors="ignore").to_csv(output_dir / "summary.csv")

    return trades, equity, summary
