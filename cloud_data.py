from pathlib import Path
import os


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/Users/xinc/GitHub/google_drive/Data"))


def data_path(*parts: str) -> Path:
    return DATA_ROOT.joinpath(*parts)


def manifest_path() -> Path:
    return data_path("_index", "file_manifest.csv")


def dataset_index_path() -> Path:
    return data_path("_index", "dataset_index.csv")


TM_ROOT = data_path("tm")
TRADING_ROOT = data_path("trading")

TW_STOCK_DAILY_PRICE = data_path("trading", "tw_stock", "daily_price", "daily_stock_price.parquet")
TW_STOCK_KBAR = data_path("trading", "tw_stock", "kbar")
TW_STOCK_KBAR_ABOVE_MA60 = data_path("trading", "tw_stock", "kbar", "above_ma60")
TW_STOCK_KBAR_1MIN = data_path("trading", "tw_stock", "kbar_1min")
TW_FUTURES_TX = data_path("trading", "tw_futures", "TX.csv")
TW_OPTIONS_TICK = data_path("trading", "tw_options", "tick")

CRYPTO_COINTEGRATION = data_path("trading", "crypto", "cointegration")
CRYPTO_SPOT = CRYPTO_COINTEGRATION / "spot"
CRYPTO_FUTURES = CRYPTO_COINTEGRATION / "futures"
CRYPTO_FUNDING_RATE = CRYPTO_COINTEGRATION / "funding_rate"
CRYPTO_METADATA = CRYPTO_COINTEGRATION / "metadata"
CRYPTO_CACHE = CRYPTO_COINTEGRATION / "cache"
CRYPTO_OHLCV_1D = data_path("trading", "crypto", "ohlcv_1d")
