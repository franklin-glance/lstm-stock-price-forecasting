# using alphavantage api
# https://www.alphavantage.co/documentation/

import pandas as pd
import requests
import env
import os
import json
import datetime
import csv, json
import re

api_key = env.get_api_key()


def get_timeseries(symbol, function='TIME_SERIES_DAILY_ADJUSTED', verbose=False):
    # check if symbol data is stored in cache
    # if not, get it and store it
    # if yes, return it
    if os.path.exists(f'./cache/{symbol}_{function}.json'):
        if verbose: print(f'loading {function} data from cache for {symbol}...')
        with open(f'cache/{symbol}_{function}.json') as f:
            data = json.load(f)
    else:
        if verbose: print(f'loading {function} data from api and storing to cache for {symbol}...')
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}&outputsize=full'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        # print(keys[0])
        if keys and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data for {symbol}')
            return [False]
        with open(f'./cache/{symbol}_{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    keys = list(data.keys())
    metadata = data[keys[0]]
    time_series = pd.DataFrame(data[keys[1]])
    time_series = time_series.transpose()
    if function == 'TIME_SERIES_DAILY_ADJUSTED':
        column_mapping = {'1. open': 'open', '2. high': 'high', '3. low': 'low', '4. close': 'close',
                          '5. adjusted close': 'adjusted_close', '6. volume': 'volume', '7. dividend amount': 'dividend_amount', '8. split coefficient':'split_coefficient'}
    elif function == 'TIME_SERIES_WEEKLY_ADJUSTED':
        column_mapping = {'1. open': 'weekly_open', '2. high': 'weekly_high', '3. low': 'weekly_low',
                          '4. close': 'weekly_close', '5. adjusted close': 'weekly_adjusted_close', '6. volume': 'weekly_volume', '7. dividend amount': 'weekly_dividend_amount', '8. split coefficient':'weekly_split_coefficient'}
    elif function == 'TIME_SERIES_MONTHLY_ADJUSTED':
        column_mapping = {'1. open': 'monthly_open', '2. high': 'monthly_high', '3. low': 'monthly_low',
                          '4. close': 'monthly_close', '5. adjusted close': 'monthly_adjusted_close', '6. volume': 'monthly_volume', '7. dividend amount': 'monthly_dividend_amount', '8. split coefficient':'monthly_split_coefficient'}
    else:
        # error, function not supported
        return [False, 'function not supported']

    time_series = time_series.rename(columns=column_mapping)
    # change index data type to datetime
    time_series.index = pd.to_datetime(time_series.index)
    # change time_series data to numeric
    time_series = time_series.apply(pd.to_numeric, errors='coerce')
    return [True, metadata, time_series]


def get_income_statement(symbol, verbose=False):
    function = 'INCOME_STATEMENT'
    if os.path.exists(f'./cache/{symbol}_{function}.json'):
        if verbose: print(f'loading {function} data from cache for {symbol}...')
        with open(f'cache/{symbol}_{function}.json') as f:
            data = json.load(f)
    else:
        if verbose: print(f'loading {function} data from api and storing to cache for {symbol}...')
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        if keys and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data for {symbol}')
            return [False]
        with open(f'./cache/{symbol}_{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    keys = list(data.keys())

    return [True, data]


def get_balance_sheet(symbol, verbose=False):
    function = 'BALANCE_SHEET'
    if os.path.exists(f'./cache/{symbol}_{function}.json'):
        if verbose: print(f'loading {function} data from cache for {symbol}...')
        with open(f'cache/{symbol}_{function}.json') as f:
            data = json.load(f)
    else:
        if verbose: print(f'loading {function} data from api and storing to cache for {symbol}...')
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        if keys and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data for {symbol}')
            return [False]
        with open(f'./cache/{symbol}_{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    return [True, data]


def get_earnings(symbol, verbose=False):
    function = 'EARNINGS'
    if os.path.exists(f'./cache/{symbol}_{function}.json'):
        if verbose: print(f'loading {function} data from cache for {symbol}...')
        with open(f'cache/{symbol}_{function}.json') as f:
            data = json.load(f)
    else:
        if verbose: print(f'loading {function} data from api and storing to cache for {symbol}...')
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        if keys and len(keys) > 0 and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data for {symbol}')
            return [False]
        with open(f'./cache/{symbol}_{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    keys = list(data.keys())
    # symbol= data[keys[0]]

    annualEarnings = pd.DataFrame(data[keys[1]])

    annualEarnings = annualEarnings.set_index(annualEarnings.columns[0])

    quarterlyEarnings = pd.DataFrame(data[keys[2]])
    quarterlyEarnings = quarterlyEarnings.set_index(quarterlyEarnings.columns[1])

    annualEarnings.index = pd.to_datetime(annualEarnings.index)
    quarterlyEarnings.index = pd.to_datetime(quarterlyEarnings.index)

    return [True, annualEarnings, quarterlyEarnings]


def get_company_overview(symbol):
    function = 'OVERVIEW'
    if os.path.exists(f'./cache/{symbol}_{function}.json'):
        print(f'loading {function} data from cache for {symbol}...')
        with open(f'cache/{symbol}_{function}.json') as f:
            data = json.load(f)
    else:
        print(f'loading {function} data from api and storing to cache for {symbol}...')
        url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        if keys and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data for {symbol}')
            return [False]
        with open(f'./cache/{symbol}_{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    keys = list(data.keys())
    # symbol= data[keys[0]]

    companyOverview = data

    return [True, companyOverview]


# note, date must be later than 2010-01-01
def get_status(date=datetime.datetime.now().strftime('%Y-%m-%d'), state='active', verbose=False):
    function = 'LISTING_STATUS'
    if os.path.exists(f'./cache/{date}_{function}_{state}.json'):
        if verbose: print(f'loading {function} data from cache for {date}...')
        with open(f'cache/{date}_{function}_{state}.json') as f:
            data = json.load(f)
    else:
        if verbose: print(f'loading {function} data from api and storing to cache for {date}...')
        url = f'https://www.alphavantage.co/query?function={function}&date={date}&apikey={api_key}&state={state}'
        with requests.Session() as s:
            download = s.get(url)
            decoded_content = download.content.decode('utf-8')
            cr = csv.reader(decoded_content.splitlines(), delimiter=',')
            data = list(cr)
        with open(f'./cache/{date}_{function}_{state}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    status = pd.DataFrame(data, columns=data[0]).iloc[1:]
    status = status.set_index(status.columns[0])

    status['ipoDate'] = pd.to_datetime(status['ipoDate'])
    # drop rows with ipoDate after 2021-01-01
    status = status[status['ipoDate'] < '2021-01-01']
    # drop rows with

    return [True, status]


def get_real_gdp(function='REAL_GDP', interval='quarterly', verbose=False):
    if os.path.exists(f'./cache/{interval}_{function}.json'):
        print(f'loading {function} data from cache for {interval}...')
        with open(f'cache/{interval}_{function}.json') as f:
            data = json.load(f)
    else:
        print(f'loading {function} data from api and storing to cache for {interval}...')
        url = f'https://www.alphavantage.co/query?function={function}&apikey={api_key}&interval={interval}'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        if keys and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data for {interval}')
            return [False]

        with open(f'./cache/{interval}_{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    gdp = pd.DataFrame(data['data']).set_index('date')
    gdp.index = pd.to_datetime(gdp.index)
    gdp[f'gdp_{interval}'] = gdp['value'].astype(float)
    gdp = gdp.drop(columns=['value'])
    return [True, gdp]


def get_consumer_sentiment():
    function = 'CONSUMER_SENTIMENT'
    if os.path.exists(f'./cache/{function}.json'):
        print(f'loading {function} data from cache ...')
        with open(f'cache/{function}.json') as f:
            data = json.load(f)
    else:
        print(f'loading {function} data from api and storing to cache for...')
        url = f'https://www.alphavantage.co/query?function={function}&apikey={api_key}'
        r = requests.get(url)
        data = r.json()
        keys = list(data.keys())
        if keys and keys[0] == 'Information':
            print("api request limit reached")
            return [False]
        elif not keys:
            print(f'no {function} data')
            return [False]

        with open(f'./cache/{function}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    df = pd.DataFrame(data['data']).set_index('date')
    df.index = pd.to_datetime(df.index)
    df['consumer_sentiment'] = df['value']
    df = df.drop(columns=['value'])

    return [True, df]
