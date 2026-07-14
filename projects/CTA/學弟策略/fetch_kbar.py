"""
台股分K資料批次下載腳本

使用 FinMind SDK 的 taiwan_stock_kbar + use_async=True
一次抓取一天全部股票的分K資料（約 2~3 分鐘/天）。

限制：
  - Sponsor 會員專用
  - 資料從 2019-01-01 起
  - ~1500 個交易日 × ~2.5 分/天 ≈ 62 小時

儲存：每個交易日一個 parquet，存於 kbar/1min/
斷點續傳：progress.json 記錄已完成日期
"""

import json
import time
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from FinMind.data import DataLoader

# ── 設定 ──
API_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9."
    "eyJkYXRlIjoiMjAyNS0wNy0xNiAxNjozODozMSIsInVzZXJfaWQiOiJKcUNh"
    "cGl0YWxCYWNrdXAiLCJpcCI6IjYxLjIxOS4xOS4xNTAifQ."
    "cMC_wR6cGdwxtnqdfY_Huc2llBVpiCIcoJz5BuXqoms"
)
try:
    _SCRIPT_DIR = Path(__file__).parent
except NameError:
    _SCRIPT_DIR = Path.cwd()  # notebook 環境
_QUANT_DIR = next(
    (
        root
        for root in (_SCRIPT_DIR, *_SCRIPT_DIR.parents)
        if (root / "cloud_data.py").exists()
    ),
    None,
)
if _QUANT_DIR is not None:
    sys.path.insert(0, str(_QUANT_DIR))
    from cloud_data import TW_STOCK_KBAR_1MIN

    OUTPUT_DIR = TW_STOCK_KBAR_1MIN
else:
    OUTPUT_DIR = _SCRIPT_DIR / "kbar" / "1min"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

START_DATE = "2019-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

MAX_RETRIES = 3
RETRY_WAIT = 60  # 失敗重試間隔秒數


# ── 工具函數 ──
def setup_logging() -> logging.Logger:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # 抑制 FinMind / loguru 的 info 訊息
    logging.getLogger("FinMind").setLevel(logging.WARNING)
    try:
        from loguru import logger as loguru_logger
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="WARNING")
    except ImportError:
        pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(OUTPUT_DIR / "download.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("fetch_kbar")


def load_progress() -> set[str]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f).get("completed_dates", []))
    return set()


def save_progress(completed: set[str]):
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump({"completed_dates": sorted(completed)}, f, ensure_ascii=False)


def get_stock_ids(dl: DataLoader) -> list[str]:
    """取得有效的台股股票代號清單（排除大盤/指數等）"""
    info = dl.taiwan_stock_info()
    info = info[info["type"].isin(["twse", "tpex"])]
    cate_mask = info["industry_category"].isin(["大盤", "Index", "所有證券"])
    id_mask = info["stock_id"].isin(["TAIEX", "TPEx"])
    info = info[~(cate_mask | id_mask)]
    return sorted(info["stock_id"].unique().tolist())


def get_trading_dates(dl: DataLoader, start: str, end: str) -> list[str]:
    """取得交易日列表"""
    df = dl.get_data(
        dataset="TaiwanStockTradingDate",
        start_date=start,
        end_date=end,
    )
    return sorted(df.loc[df["is_trading_day"] == "Y", "date"].tolist())


def main():
    logger = setup_logging()
    completed = load_progress()

    logger.info("=" * 60)
    logger.info(f"台股分K資料下載 | {START_DATE} ~ {END_DATE}")
    logger.info(f"已完成: {len(completed)} 天")
    logger.info("=" * 60)

    # 初始化 FinMind
    dl = DataLoader()
    dl.login_by_token(api_token=API_TOKEN)

    # 股票清單
    logger.info("取得股票清單...")
    stock_ids = get_stock_ids(dl)
    logger.info(f"共 {len(stock_ids)} 檔股票")

    # 交易日
    logger.info("取得交易日...")
    trading_dates = get_trading_dates(dl, START_DATE, END_DATE)
    logger.info(f"共 {len(trading_dates)} 個交易日")

    remaining = [d for d in trading_dates if d not in completed]
    logger.info(f"待下載: {len(remaining)} 天")

    if not remaining:
        logger.info("全部已完成！")
        return

    est_hours = len(remaining) * 2.5 / 60
    logger.info(f"預估剩餘: {est_hours:.1f} 小時")

    # 逐日下載
    for idx, date_str in enumerate(remaining):
        out_file = OUTPUT_DIR / f"{date_str}.parquet"
        logger.info(f"[{idx+1}/{len(remaining)}] {date_str} ...")

        retries = 0
        success = False

        while retries < MAX_RETRIES:
            try:
                t0 = time.time()
                df = dl.taiwan_stock_kbar(
                    stock_id_list=stock_ids,
                    date=date_str,
                    use_async=True,
                )
                elapsed = time.time() - t0

                if df is not None and not df.empty:
                    df.to_parquet(out_file, index=False)
                    n_stocks = df["stock_id"].nunique()
                    logger.info(
                        f"  ✓ {len(df)} 筆 / {n_stocks} 檔 "
                        f"({elapsed:.1f}s) → {out_file.name}"
                    )
                else:
                    logger.warning(f"  ✗ 無資料 ({elapsed:.1f}s)")

                completed.add(date_str)
                save_progress(completed)
                success = True
                break

            except Exception as exc:
                retries += 1
                logger.error(
                    f"  失敗({retries}/{MAX_RETRIES}): {exc}"
                )
                if retries < MAX_RETRIES:
                    logger.info(f"  等待 {RETRY_WAIT}s 後重試...")
                    time.sleep(RETRY_WAIT)

        if not success:
            logger.error("超過重試上限，中止。重新執行即可續傳。")
            break

    logger.info(
        f"結束。已完成 {len(completed)}/{len(trading_dates)} 個交易日。"
    )


if __name__ == "__main__":
    main()
