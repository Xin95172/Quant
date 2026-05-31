from pathlib import Path
import time

import pandas as pd
from tqdm import tqdm


def get_valid_stock_ids(client, output_file: Path | None = None) -> list[str]:
    if output_file is not None and output_file.exists():
        cached = pd.read_parquet(output_file)
        return sorted(cached["stock_id"].astype(str).unique().tolist())

    info = client.taiwan_stock_info()
    info = info[info["type"].isin(["twse", "tpex"])]
    cate_mask = info["industry_category"].isin(["大盤", "Index", "所有證券"])
    id_mask = info["stock_id"].isin(["TAIEX", "TPEx"])
    info = info[~(cate_mask | id_mask)]
    stock_ids = sorted(info["stock_id"].astype(str).unique().tolist())

    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"stock_id": stock_ids}).to_parquet(output_file, index=False)
    return stock_ids


def infer_stock_ids_from_kbar_dir(kbar_dir: Path, max_files: int = 20) -> list[str]:
    stock_ids: set[str] = set()
    files = sorted(kbar_dir.glob("*.parquet"), reverse=True)[:max_files]
    for path in files:
        stock_ids.update(read_kbar_ids(path))
    return sorted(stock_ids)


def compact_daily_price_parquet(
    source_file: Path,
    output_file: Path,
    stock_ids: list[str],
    columns: list[str] | None = None,
    batch_size: int = 500_000,
) -> pd.DataFrame:
    """Create a smaller daily-price parquet without loading the huge file at once."""
    import pyarrow.parquet as pq

    if columns is None:
        columns = ["date", "stock_id", "open", "max", "min", "close", "Trading_Volume"]

    expected_ids = {str(stock_id) for stock_id in stock_ids}
    parquet_file = pq.ParquetFile(source_file)
    frames: list[pd.DataFrame] = []

    for batch in tqdm(
        parquet_file.iter_batches(batch_size=batch_size, columns=columns),
        total=parquet_file.metadata.num_rows // batch_size + 1,
        desc="Compact daily price",
        unit="batch",
    ):
        chunk = batch.to_pandas()
        chunk["stock_id"] = chunk["stock_id"].astype(str)
        chunk = chunk[chunk["stock_id"].isin(expected_ids)]
        if not chunk.empty:
            frames.append(chunk)

    if frames:
        daily = pd.concat(frames, ignore_index=True)
        daily["date"] = pd.to_datetime(daily["date"])
        daily["stock_id"] = daily["stock_id"].astype(str)
        daily = daily.drop_duplicates(subset=["date", "stock_id"], keep="last")
        daily = daily.sort_values(["date", "stock_id"]).reset_index(drop=True)
    else:
        daily = pd.DataFrame(columns=columns)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_file, index=False)
    return daily


def read_kbar_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        return set(pd.read_parquet(path, columns=["stock_id"])["stock_id"].astype(str))
    except Exception:
        return set()


def dedupe_kbar(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_cols = [
        col for col in ["date", "stock_id", "Time", "time", "minute"]
        if col in df.columns
    ]
    if {"date", "stock_id"}.issubset(dedupe_cols) and any(
        col in dedupe_cols for col in ["Time", "time", "minute"]
    ):
        df = df.drop_duplicates(subset=dedupe_cols, keep="last")
    else:
        df = df.drop_duplicates(keep="last")

    sort_cols = [
        col for col in ["date", "Time", "time", "minute", "stock_id"]
        if col in df.columns
    ]
    if sort_cols:
        df = df.sort_values(sort_cols)
    return df.reset_index(drop=True)


def write_filtered_from_full_kbar(
    date_str: str,
    stock_ids: list[str],
    full_kbar_dir: Path,
    output_dir: Path,
) -> bool:
    full_file = full_kbar_dir / f"{date_str}.parquet"
    out_file = output_dir / f"{date_str}.parquet"

    if not full_file.exists():
        return False

    expected_ids = {str(stock_id) for stock_id in stock_ids}
    full_ids = read_kbar_ids(full_file)
    if not expected_ids.issubset(full_ids):
        return False

    full_df = pd.read_parquet(full_file)
    filtered = full_df[full_df["stock_id"].astype(str).isin(expected_ids)].copy()
    if filtered.empty:
        return False

    if out_file.exists():
        old_df = pd.read_parquet(out_file)
        filtered = pd.concat([old_df, filtered], ignore_index=True)

    filtered = dedupe_kbar(filtered)
    filtered.to_parquet(out_file, index=False)
    return expected_ids.issubset(read_kbar_ids(out_file))


def load_or_refresh_daily_prices(
    client,
    output_file: Path,
    start_date: str,
    end_date: str,
    stock_ids: list[str] | None = None,
    retry_wait: int = 60,
) -> pd.DataFrame:
    requested_ids = None if stock_ids is None else {str(stock_id) for stock_id in stock_ids}
    trading_dates = client.get_data(
        dataset="TaiwanStockTradingDate",
        start_date=start_date,
        end_date=end_date,
    )
    if "is_trading_day" in trading_dates.columns:
        trading_dates = trading_dates[trading_dates["is_trading_day"] == "Y"]
    trading_dates = sorted(trading_dates["date"].astype(str).unique().tolist())

    if output_file.exists():
        daily = pd.read_parquet(output_file, columns=["date", "stock_id", "close"])
        daily["stock_id"] = daily["stock_id"].astype(str)
        if requested_ids is not None:
            daily = daily[daily["stock_id"].isin(requested_ids)]
        if not pd.api.types.is_datetime64_any_dtype(daily["date"]):
            daily["date"] = pd.to_datetime(daily["date"])
    else:
        daily = pd.DataFrame()

    if not daily.empty:
        cached_dates = set(daily["date"].dt.strftime("%Y-%m-%d"))
    else:
        cached_dates = set()

    missing_dates = [d for d in trading_dates if d not in cached_dates]
    if not missing_dates:
        return daily

    frames = [daily] if not daily.empty else []
    for date_str in tqdm(missing_dates, desc="Download daily price", unit="day"):
        try:
            one_day = client.get_data(
                dataset="TaiwanStockPrice",
                start_date=date_str,
                end_date=date_str,
            )
        except Exception:
            time.sleep(retry_wait)
            one_day = client.get_data(
                dataset="TaiwanStockPrice",
                start_date=date_str,
                end_date=date_str,
            )

        if one_day is not None and not one_day.empty:
            one_day["stock_id"] = one_day["stock_id"].astype(str)
            if requested_ids is not None:
                one_day = one_day[one_day["stock_id"].isin(requested_ids)]
            keep_cols = [col for col in ["date", "stock_id", "open", "max", "min", "close", "Trading_Volume"] if col in one_day.columns]
            one_day = one_day[keep_cols]
            frames.append(one_day)

    if frames:
        daily = pd.concat(frames, ignore_index=True)
        if not pd.api.types.is_datetime64_any_dtype(daily["date"]):
            daily["date"] = pd.to_datetime(daily["date"])
        daily["stock_id"] = daily["stock_id"].astype(str)
        daily = daily.drop_duplicates(subset=["date", "stock_id"], keep="last")
        daily = daily.sort_values(["date", "stock_id"]).reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_file, index=False)
    return daily


def build_above_ma_signal(
    daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    ma_window: int = 60,
) -> pd.DataFrame:
    """Return a date x stock_id boolean signal matrix.

    The signal is shifted by one row per stock, so each date uses the prior
    trading day's close-vs-MA result and avoids lookahead.
    """
    daily = daily[["date", "stock_id", "close"]].copy()
    if not pd.api.types.is_datetime64_any_dtype(daily["date"]):
        daily["date"] = pd.to_datetime(daily["date"])
    daily["stock_id"] = daily["stock_id"].astype(str)
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily = daily.dropna(subset=["close"])

    daily = daily.sort_values(["stock_id", "date"])
    daily["ma"] = (
        daily.groupby("stock_id")["close"]
        .transform(lambda s: s.rolling(ma_window, min_periods=ma_window).mean())
    )

    daily["above_ma"] = daily["close"] > daily["ma"]
    shifted_signal = daily.groupby("stock_id")["above_ma"].shift(1)
    daily["signal"] = shifted_signal.where(shifted_signal.notna(), False).astype(bool)

    signal = daily.pivot(index="date", columns="stock_id", values="signal")
    signal = signal.astype("boolean").fillna(False).astype(bool)
    signal = signal.loc[
        (signal.index >= pd.Timestamp(start_date))
        & (signal.index <= pd.Timestamp(end_date))
    ]
    signal.index.name = "date"
    return signal


def load_or_build_above_ma_signal(
    daily: pd.DataFrame,
    output_file: Path,
    start_date: str,
    end_date: str,
    ma_window: int = 60,
    rebuild: bool = False,
) -> pd.DataFrame:
    if output_file.exists() and not rebuild:
        signal = pd.read_parquet(output_file)
        signal.index = pd.to_datetime(signal.index)
        if (
            signal.index.min() <= pd.Timestamp(start_date)
            and signal.index.max() >= pd.Timestamp(end_date)
        ):
            return signal.loc[
                (signal.index >= pd.Timestamp(start_date))
                & (signal.index <= pd.Timestamp(end_date))
            ].astype(bool)

    signal = build_above_ma_signal(
        daily=daily,
        start_date=start_date,
        end_date=end_date,
        ma_window=ma_window,
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    signal.to_parquet(output_file)
    return signal


def signal_row_to_stock_ids(row: pd.Series) -> list[str]:
    return row[row.astype(bool)].index.astype(str).tolist()


def build_above_ma_universe(
    daily: pd.DataFrame,
    start_date: str,
    end_date: str,
    ma_window: int = 60,
) -> dict[pd.Timestamp, list[str]]:
    signal = build_above_ma_signal(
        daily=daily,
        start_date=start_date,
        end_date=end_date,
        ma_window=ma_window,
    )
    universe_by_date = {
        date: signal_row_to_stock_ids(row)
        for date, row in signal.iterrows()
        if row.any()
    }
    return universe_by_date
