import yfinance as yfin
import os
from datetime import timedelta
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime, timedelta

symbals = []

for i in range(1, int(page) + 1):
    url = f'https://histock.tw/stock/rank.aspx?&p={i}&d=1'
    res = requests.get(url)
    soup = BeautifulSoup(res.text, 'html.parser')

    numbers = soup.find_all('td', string=re.compile(r'^\d{4}$'))
    for num in numbers:
        symbals.append(num.get_text(strip=True))

start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
end_date = datetime.now().strftime('%Y-%m-%d')

save_directory = '/Users/xinc./Documents/GitHub/Quant/data/stock_price'
os.makedirs(save_directory, exist_ok=True)

for sym in symbals:
    ticker = f'{sym}.TW'
    try:
        df = yfin.download(ticker, start = start_date, end = end_date)

        if df.empty:
            print(f'No data for {ticker}. It might be delisted or does not exist.')
            continue

        df.to_csv(f'/Users/xinc./Documents/GitHub/Quant/data/stock_price/{sym}.csv')
        
    except Exception as e:
        print(f'Failed to download {ticker}: {e}')


