"""
分K → 時K resample 工具

用法（notebook）:
    from resample_kbar import resample_kbar_to_1h
    resample_kbar_to_1h(source_dir=TW_STOCK_KBAR_1MIN, output_dir=TW_STOCK_KBAR_1H)

輸出格式（每日一個 parquet）:
    date        object  '2025-01-02'
    hour        object  '09:00:00'
    stock_id    object  '2330'
    open       float64
    high       float64
    low        float64
    close      float64
    volume       int64
"""

from pathlib import Path

import pandas as pd
from tqdm import tqdm


_OHLCV_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def _resample_one_day(df: pd.DataFrame) -> pd.DataFrame:
    """將單日分K DataFrame resample 成 1H OHLCV。

    bar 標籤為開始時間（left-label，與 XQ/元大等看盤軟體一致）：
        標為 09:00 的 bar 涵蓋 09:00~09:59
        標為 13:00 的 bar 涵蓋 13:00~13:30
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["minute"])

    # 向量化 resample：floor 到小時即為左標籤
    df["hour_dt"] = df["datetime"].dt.floor("1h")

    ohlcv_cols = list(_OHLCV_AGG)
    result = (
        df.groupby(["stock_id", "hour_dt"])[ohlcv_cols]
        .agg(_OHLCV_AGG)
        .reset_index()
    )
    result = result.dropna(subset=["close"])
    result["date"] = result["hour_dt"].dt.strftime("%Y-%m-%d")
    result["hour"] = result["hour_dt"].dt.strftime("%H:%M:%S")
    result = result.drop(columns=["hour_dt"])
    result = result[["date", "hour", "stock_id", "open", "high", "low", "close", "volume"]]
    result = result.sort_values(["hour", "stock_id"]).reset_index(drop=True)
    return result


def _read_and_merge(
    date_str: str,
    source_dir: Path,
    supplement_dir: Path | None,
) -> pd.DataFrame:
    """讀取 source_dir 和 supplement_dir 的同一天資料，合併去重。"""
    frames = []

    src = source_dir / f"{date_str}.parquet"
    if src.exists():
        frames.append(pd.read_parquet(src))

    if supplement_dir is not None:
        sup = supplement_dir / f"{date_str}.parquet"
        if sup.exists():
            frames.append(pd.read_parquet(sup))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    # 同一 (minute, stock_id) 以 source_dir（全市場）優先，supplement 補缺
    df = df.drop_duplicates(subset=["minute", "stock_id"], keep="first")
    return df


def resample_kbar_to_1h(
    source_dir: Path,
    output_dir: Path,
    supplement_dir: Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    force: bool = False,
) -> list[str]:
    """
    將 source_dir 中的分K parquet（每日一檔）resample 成 1H，存至 output_dir。
    若提供 supplement_dir，會合併補充資料（above_ma60）後再 resample。

    Args:
        source_dir:     全市場分K目錄（TW_STOCK_KBAR_1MIN）
        output_dir:     輸出目錄（TW_STOCK_KBAR_1H）
        supplement_dir: 補充分K目錄（TW_STOCK_KBAR_ABOVE_MA60），可不傳
        start_date:     起始日期（含），格式 'YYYY-MM-DD'，None 表示全部
        end_date:       結束日期（含），格式 'YYYY-MM-DD'，None 表示全部
        force:          True 則重新計算已存在的日期

    Returns:
        已處理的日期列表
    """
    import re
    _DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    supplement_dir = Path(supplement_dir) if supplement_dir is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取得所有有資料的日期（兩個來源的聯集）
    date_set = {f.stem for f in source_dir.glob("*.parquet") if _DATE_RE.match(f.stem)}
    if supplement_dir is not None:
        date_set |= {f.stem for f in supplement_dir.glob("*.parquet") if _DATE_RE.match(f.stem)}
    date_strs = sorted(date_set)

    # 日期範圍過濾
    if start_date:
        date_strs = [d for d in date_strs if d >= start_date]
    if end_date:
        date_strs = [d for d in date_strs if d <= end_date]

    # 斷點續傳：跳過已存在且股票數相符的日期
    if not force:
        existing = {f.stem for f in output_dir.glob("*.parquet")}
        todo = []
        for d in date_strs:
            if d not in existing:
                todo.append(d)
            else:
                try:
                    src_n = _read_and_merge(d, source_dir, supplement_dir)
                    src_n = src_n["stock_id"].nunique() if not src_n.empty else 0
                    out_n = pd.read_parquet(
                        output_dir / f"{d}.parquet", columns=["stock_id"]
                    )["stock_id"].nunique()
                    if src_n != out_n:
                        todo.append(d)
                except Exception:
                    todo.append(d)
    else:
        todo = date_strs

    if not todo:
        print(f"全部已完成，共 {len(date_strs)} 天。")
        return []

    print(f"待處理: {len(todo)} 天（已有 {len(date_strs) - len(todo)} 天）")

    processed = []
    errors = []
    for date_str in tqdm(todo, desc="Resample 1min→1H", unit="day"):
        dst = output_dir / f"{date_str}.parquet"
        try:
            df = _read_and_merge(date_str, source_dir, supplement_dir)
            if df.empty:
                continue
            result = _resample_one_day(df)
            if not result.empty:
                result.to_parquet(dst, index=False)
                processed.append(date_str)
        except Exception as exc:
            errors.append(date_str)
            print(f"  ✗ {date_str}: {exc}")

    print(f"完成: {len(processed)} 天，錯誤: {len(errors)} 天")
    return processed
