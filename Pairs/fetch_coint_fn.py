import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
import itertools
from multiprocessing import Pool, cpu_count
from functools import partial

def adf_test(series):
    result = adfuller(series)
    return result[1]

def check_cointegration(series1, series2):
    score, p_value, _ = coint(series1, series2)
    
    if p_value < 0.05:
        return True, p_value
    else:
        return False, p_value
    
def process_pair(pair, spot_folder, start_time, end_time):
    symbol1, symbol2 = pair
    start_time, end_time = pd.to_datetime(start_time), pd.to_datetime(end_time)
    data1 = pd.read_csv(f"{spot_folder}/{symbol1}.csv")
    data2 = pd.read_csv(f"{spot_folder}/{symbol2}.csv")
    data1["Timestamp"], data2["Timestamp"] = pd.to_datetime(data1["Timestamp"]), pd.to_datetime(data2["Timestamp"])
    
    sliced_data1 = data1[(data1["Timestamp"] >= start_time) & (data1["Timestamp"] <= end_time)]
    sliced_data2 = data2[(data2["Timestamp"] >= start_time) & (data2["Timestamp"] <= end_time)]
    
    if sliced_data1.empty or sliced_data2.empty:
        return None
    if len(sliced_data1) != len(sliced_data2):
        return None
    
    data1_close = sliced_data1["Close"]
    data2_close = sliced_data2["Close"]

    if adf_test(data1_close) > 0.05 and adf_test(data2_close) > 0.05:
        is_coint, p_value = check_cointegration(data1_close, data2_close)
        if is_coint:
            return {
                "pair": (symbol1, symbol2),
                "p_value": p_value
            }
        
    return None

def fetch_coint_pairs(sector_map, spot_folder, start_time, end_time, max_workers = None):
    coint_pairs = []
    for sector, info in sector_map.items():
        symbols = info["symbols"]
        pairs = list(itertools.combinations(symbols, 2))
        num_pairs = len(pairs)
        num_workers = max_workers or cpu_count()

        with Pool(num_workers) as pool:
            fn = partial(process_pair, spot_folder = spot_folder, start_time = str(start_time), end_time = str(end_time))
            results = list(pool.imap(fn, pairs))

        for pair_info in results:
            if pair_info is not None:
                pair = pair_info.get("pair")
                coint_pairs.append({
                    "sector": sector,
                    "pair": pair,
                    "start_time": start_time,
                    "end_time": end_time,
                    "p_value": pair_info.get("p_value")
                })

    return coint_pairs