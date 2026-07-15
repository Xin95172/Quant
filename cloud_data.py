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

TW_STOCK_ROOT = data_path("trading", "tw_stock")
TW_STOCK_REFERENCE = TW_STOCK_ROOT / "reference"
TW_STOCK_DISPOSAL = TW_STOCK_ROOT / "disposal"
TW_STOCK_KBAR_ROOT = TW_STOCK_ROOT / "kbar"
TW_STOCK_KBAR_1D = TW_STOCK_KBAR_ROOT / "1D"
TW_STOCK_KBAR_1MIN = TW_STOCK_KBAR_ROOT / "1min"
TW_STOCK_KBAR_ABOVE_MA60 = TW_STOCK_KBAR_1MIN / "above_ma60"
TW_STOCK_KBAR_1H = TW_STOCK_KBAR_ROOT / "1H"
TW_STOCK_DAILY_PRICE = TW_STOCK_KBAR_1D / "daily_stock_price.parquet"
TW_STOCK_SECTOR_MAP = TW_STOCK_ROOT / "sector_map.parquet"
TW_STOCK_MARGIN_INFO = TW_STOCK_REFERENCE / "margin_info.csv"
TW_STOCK_OVERVIEW = TW_STOCK_REFERENCE / "台股總覽.csv"
TW_STOCK_INSTITUTIONAL_TRADING = TW_STOCK_REFERENCE / "整體市場三大法人買賣表.csv"
TW_FUTURES_TX = data_path("trading", "tw_futures", "TX.csv")
TW_OPTIONS_TICK = data_path("trading", "tw_options", "tick")

CRYPTO_COINTEGRATION = data_path("trading", "crypto", "cointegration")
CRYPTO_SPOT = CRYPTO_COINTEGRATION / "spot"
CRYPTO_FUTURES = CRYPTO_COINTEGRATION / "futures"
CRYPTO_FUNDING_RATE = CRYPTO_COINTEGRATION / "funding_rate"
CRYPTO_METADATA = CRYPTO_COINTEGRATION / "metadata"
CRYPTO_CACHE = CRYPTO_COINTEGRATION / "cache"
CRYPTO_OHLCV_1D = data_path("trading", "crypto", "ohlcv_1d")
