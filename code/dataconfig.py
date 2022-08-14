import os

import api
import pickle
'''
This file handles the loading and storage of stock data from api/cache
'''

import env
import os

E = env.env()


class DataConfig:
    def __init__(self):
        # List of Tickers
        self.tickers_available = api.get_status()[1]
        # self.consumer_sentiment = api.get_consumer_sentiment()
        self.tickers = []
        self.data = {}

    def getdata(self, request, verbose=False, new=False, allstocks=False):
        datalist = []
        if verbose: print('Preparing stock data for request')
        E.reset_api()
        count = 0
        if new:
            if request in self.tickers_available.index and not request[-1].isnumeric():
                if self.tickers_available.loc[request, 'assetType'] == 'Stock':
                    return Stock(request, verbose=verbose, new=new)
                else:
                    return ETF(request, verbose=verbose, new=new)
        else:
            if allstocks:
                print('checking if allstocks is stored locally')
                # check if file exists
                if os.path.isfile(os.getcwd() + '/cache/allstocks.pkl'):
                    with open(os.getcwd() + '/cache/allstocks.pkl', 'rb') as f:
                        datalist = pickle.load(f)
                    print('allstocks loaded from local storage')
                    return datalist
                print('allstocks not found locally')
            for t in request:
                count += 1
                if verbose:
                    print(f'Retrieving data for {t}, request {count} of {len(request)}')
                # TODO: fix this so tickers can have numbers
                if t in self.tickers_available.index and not t[-1].isnumeric():
                    self.tickers.append(t)
                    if self.tickers_available.loc[t, 'assetType'] == 'Stock':
                        self.data[t] = Stock(t, verbose, new)
                        datalist.append(self.data[t])
                    else:
                        self.data[t] = ETF(t, verbose, new)
                        datalist.append(self.data[t])
                else:
                    print(f'ERROR: {t} is not in the list of tickers')

            if len(request) != len(datalist):
                print(f'col len: {len(datalist)}, req len: {len(request)}')
                print('ERROR: Some tickers were not found')
            else:
                if verbose: print('All tickers were found successfully')
            if allstocks:
                # save datalist to file
                print('saving allstocks to local storage')
                with open(os.getcwd() + '/cache/allstocks.pkl', 'wb') as f:
                    pickle.dump(datalist, f)
            return datalist


class ETF:
    def __init__(self, t, verbose=False, alldata=False, new=False):
        temp = api.get_timeseries(t, 'TIME_SERIES_DAILY_ADJUSTED', verbose, new)
        if temp[0]:

            self.daily_metadata, self.daily = temp[1], temp[2]
            self.symbol = self.daily_metadata['2. Symbol']
            self.info = self.daily_metadata['1. Information']
            self.last_refreshed = self.daily_metadata['3. Last Refreshed']
            self.outputsize = self.daily_metadata['4. Output Size']
            self.timezone = self.daily_metadata['5. Time Zone']
        else:
            print(f'Time Series Daily data unavailable for {t}')
            self.daily_metadata, self.daily = None, None

        if alldata:
            temp = api.get_timeseries(t, 'TIME_SERIES_WEEKLY_ADJUSTED', verbose)
            if temp[0]:
                self.weekly_metadata, self.weekly = temp[1], temp[2]
            else:
                print(f'Time Series Weekly data unavailable for {t}')
                self.weekly_metadata, self.weekly = None, None

            temp = api.get_timeseries(t, 'TIME_SERIES_MONTHLY_ADJUSTED', verbose)
            if temp[0]:
                self.monthly_metadata, self.monthly = temp[1], temp[2]
            else:
                print(f'Time Series Monthly data unavailable for {t}')
                self.monthly_metadata, self.monthly = None, None


class Stock:
    def __init__(self, t, verbose=False, alldata=False, new=False):
        temp = api.get_timeseries(t, 'TIME_SERIES_DAILY_ADJUSTED', verbose, new)
        if temp[0]:
            self.daily_metadata, self.daily = temp[1], temp[2]
            self.symbol = self.daily_metadata['2. Symbol']
            self.info = self.daily_metadata['1. Information']
            self.last_refreshed = self.daily_metadata['3. Last Refreshed']
            self.outputsize = self.daily_metadata['4. Output Size']
            self.timezone = self.daily_metadata['5. Time Zone']
        else:
            print(f'Time Series Daily data unavailable for {t}')
            self.daily_metadata, self.daily = None, None

        if alldata:
            temp = api.get_timeseries(t, 'TIME_SERIES_WEEKLY_ADJUSTED', verbose)
            if temp[0]:
                self.weekly_metadata, self.weekly = temp[1], temp[2]
            else:
                print(f'Time Series Weekly data unavailable for {t}')
                self.weekly_metadata, self.weekly = None, None

            temp = api.get_timeseries(t, 'TIME_SERIES_MONTHLY_ADJUSTED', verbose)
            if temp[0]:
                self.monthly_metadata, self.monthly = temp[1], temp[2]
            else:
                print(f'Time Series Monthly data unavailable for {t}')
                self.monthly_metadata, self.monthly = None, None

    def buildDF(self):
        return self.daily


if __name__ == '__main__':
    '''
    This will test the functions of the DataConfig class 
    '''

    dc = DataConfig()
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
               'JNJ']
    # data = dc.getdata(request)
    data2 = dc.getdata('AAPL', new=True)
