import pandas as pd
import requests
import sys
import yfinance as yf
import math


def get_request_dataframe(url, params):
    """
    Functions that uses HTTP requests to load data and return a dataframe with the monthly average.
    :param url: URL for data location
    :param params: Parameters for request
    :return: Monthly average in Pandas Dataframe with timestamp index
    """
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
    """
    Loads the exchange rate for the given period using Norges Bank API
    """
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
    """
    Loads the interest rate for the given period using Norges Bank API
    """
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
    """
    Loads the monthly change in percent for the consumer price index for the given period using SSB API
    """
    df = pd.read_csv("https://data.ssb.no/api/v0/dataset/1086.csv?lang=en", encoding="ISO-8859-1")
    df = df[df['contents'] == 'Monthly change (per cent)']
    df.index = df['month'].apply(lambda row: pd.Timestamp(row.replace('M', '-')+'-01'))
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df.drop(columns=['consumption group', 'month', 'contents'], inplace=True)
    df.columns = ['cpi']
    return df.astype(float)


def get_gdp(start_date, end_date):
    """
    Loads the seasonally adjusted monthly change in percent for the
    gross domestic product for the given period using SSB API
    """
    df = pd.read_csv("https://data.ssb.no/api/v0/dataset/615167.csv?lang=en", encoding="ISO-8859-1")
    df = df[df['macroeconomic indicator'] == 'bnpb.nr23_9 Gross domestic product, market values']
    df = df[df['contents'] == 'Change in value from the previous month, seasonally adjusted (per cent)']
    df.index = df['month'].apply(lambda row: pd.Timestamp(row.replace('M', '-')+'-01'))
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df.drop(columns=['macroeconomic indicator', 'month', 'contents'], inplace=True)
    df.columns = ['gdp']
    return df.astype(float)


def get_trade_balance(start_date, end_date):
    """
    Loads the seasonally adjusted monthly trade balance for the given period using SSB API
    """
    df = pd.read_csv("https://data.ssb.no/api/v0/dataset/179421.csv?lang=en", encoding="ISO-8859-1")
    df = df[df['trade flow'] == 'Hbtot Trade balance, goods (Total exports - total imports)']
    df = df[df['contents'] == 'Seasonal adjusted']
    df.index = df['month'].apply(lambda row: pd.Timestamp(row.replace('M', '-')+'-01'))
    df.drop(columns=['trade flow', 'month', 'contents'], inplace=True)
    df.columns = ['trade_balance']
    df = df.pct_change() * 100  # Find month-to-month change rate
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    return df.astype(float)


def get_yahoo_data(ticker_code, start_date, end_date):
    """
    Function that uses the Yahoo API to return the monthly average of the historical closing price in the given period
    """
    c = yf.Ticker(ticker_code)
    load_months_back = math.ceil((pd.Timestamp.today()-start_date).days / 20)
    df = c.history(period=f"{load_months_back}mo")
    df.index = df.index.tz_localize(None)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    df = df[['Close']].rename(columns={'Close': 0})
    return df.resample('MS').mean()


def get_oil_price(start_date, end_date):
    """
    Loads the monthly Brent Crude Oil price in the given period
    """
    return get_yahoo_data('BZ=F', start_date, end_date).rename(columns={0: 'oil_price'})


def get_gas_price(start_date, end_date):
    """
    Loads the monthly Dutch TTF Natural Gas price in the given period
    """
    return get_yahoo_data('TTF=F', start_date, end_date).rename(columns={0: 'gas_price'})


def get_historical_data(start_date, end_date, rate):
    """
    Loads all the input data in the given period and returns it in a monthly Pandas Dataframe
    """
    frames = [get_exchange_rates(start_date, end_date, rate),
              get_interest_rates(start_date, end_date),
              get_cpi(start_date, end_date),
              get_gdp(start_date, end_date),
              get_trade_balance(start_date, end_date),
              get_oil_price(start_date, end_date),
              get_gas_price(start_date, end_date)]
    return pd.concat(frames, axis=1)
