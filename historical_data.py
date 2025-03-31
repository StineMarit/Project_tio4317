import pandas as pd
import requests
import sys
import yfinance as yf
import math


# Todo: group questions
"""
    - Freq: average to monthly
    - SSB categories check (cpi, gdp), trade balance of commodities (aka goods)
    - Seasonally adjusted: cpi, gdp. Unadjusted: trade, oil, gas.
    - Remaining: check stationarity (mean and variance) for differencing needs
"""


def get_request_dataframe(url, params):
    # Make request
    r = requests.get(url, params=params)
    if r.status_code != 200:
        print('ERROR: Request for historical exchange rate failed.')
        sys.exit(1)

    # Make dataframe
    temp_key = list(r.json()['data']['dataSets'][0]['series'].keys())[0]
    data = pd.DataFrame(r.json()['data']['dataSets'][0]['series'][temp_key]['observations']).transpose().astype(float)

    # Make timestamp index
    idx = pd.DatetimeIndex(pd.DataFrame(r.json()['data']['structure']['dimensions']['observation'][0]['values'])['start'])
    data.set_index(idx, drop=True, inplace=True)
    data.index.name = 'index'
    return data.resample('MS').mean()


def get_exchange_rates(start_date, end_date, rate):
    url = f"https://data.norges-bank.no/api/data/EXR/B.{rate.upper()}.NOK.SP?"
    params = {
        'format': 'sdmx-json',
        'startPeriod': start_date.strftime('%Y-%m-%d'),
        'endPeriod': end_date.strftime('%Y-%m-%d'),
        'locale': 'en'
    }
    df = get_request_dataframe(url, params)
    df.rename(columns={0: f'exchange_rate'.lower()}, inplace=True)
    return df


def get_interest_rates(start_date, end_date):
    # Iterate through data for government bonds and treasury bills
    url = f"https://data.norges-bank.no/api/data/IR/B.KPRA.SD.?"
    params = {
        'format': 'sdmx-json',
        'startPeriod': start_date.strftime('%Y-%m-%d'),
        'endPeriod': end_date.strftime('%Y-%m-%d'),
        'locale': 'en'
    }
    df = get_request_dataframe(url, params)
    df.rename(columns={0: 'interest_rate'}, inplace=True)
    return df.resample('MS').mean()


def get_cpi(start_date, end_date):
    df = pd.read_csv("https://data.ssb.no/api/v0/dataset/1086.csv?lang=en", encoding="ISO-8859-1")
    df = df[df['contents'] == 'Monthly change (per cent)']  # Consumer Price Index (2015=100), 12-month rate (percent), Monthly change (per cent)
    df.index = df['month'].apply(lambda row: pd.Timestamp(row.replace('M', '-')+'-01'))
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df.drop(columns=['consumption group', 'month', 'contents'], inplace=True)
    df.columns = ['cpi']
    return df.astype(float)


def get_gdp(start_date, end_date):
    df = pd.read_csv("https://data.ssb.no/api/v0/dataset/615167.csv?lang=en", encoding="ISO-8859-1")
    df = df[df['macroeconomic indicator'] == 'bnpb.nr23_9 Gross domestic product, market values']
    df = df[df['contents'] == 'Change in value from the previous month, seasonally adjusted (per cent)']  # 'Change in <value/volume> from the previous month, seasonally adjusted (per cent)'
    df.index = df['month'].apply(lambda row: pd.Timestamp(row.replace('M', '-')+'-01'))
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df.drop(columns=['macroeconomic indicator', 'month', 'contents'], inplace=True)
    df.columns = ['gdp']
    return df.astype(float)


def get_trade_balance(start_date, end_date):
    df = pd.read_csv("https://data.ssb.no/api/v0/dataset/179421.csv?lang=en", encoding="ISO-8859-1")
    df = df[df['trade flow'] == 'Hbtot Trade balance, goods (Total exports - total imports)']
    df = df[df['contents'] == 'Seasonal adjusted']  # 'Unadjusted'
    df.index = df['month'].apply(lambda row: pd.Timestamp(row.replace('M', '-')+'-01'))
    df.drop(columns=['trade flow', 'month', 'contents'], inplace=True)
    df.columns = ['trade_balance']
    df = df.pct_change() * 100  # Find month-to-month change rate
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df.astype(float)


def get_yahoo_data(ticker_code, start_date, end_date):
    c = yf.Ticker(ticker_code)
    load_months_back = math.ceil((pd.Timestamp.today()-start_date).days / 20)
    df = c.history(period=f"{load_months_back}mo")
    df.index = df.index.tz_localize(None)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df = df[['Close']].rename(columns={'Close': 0})
    return df.resample('MS').mean()


def get_oil_price(start_date, end_date):
    return get_yahoo_data('BZ=F', start_date, end_date).rename(columns={0: 'oil_price'})


def get_gas_price(start_date, end_date):
    return get_yahoo_data('TTF=F', start_date, end_date).rename(columns={0: 'gas_price'})


def get_historical_data(start_date, end_date, rate):
    frames = [get_exchange_rates(start_date, end_date, rate),
              get_interest_rates(start_date, end_date),
              get_cpi(start_date, end_date),
              get_gdp(start_date, end_date),
              get_trade_balance(start_date, end_date),
              get_oil_price(start_date, end_date),
              get_gas_price(start_date, end_date)]
    return pd.concat(frames, axis=1)
