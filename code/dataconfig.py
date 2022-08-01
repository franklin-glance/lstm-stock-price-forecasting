import pandas as pd
import api

'''
This file handles the loading and storage of stock data from api/cache
'''


class DataConfig:
    def __init__(self):
        # List of Tickers
        self.tickers_available = api.get_status()[1]
        # self.consumer_sentiment = api.get_consumer_sentiment()
        self.tickers = []
        self.data = {}

    def getdata(self, request, verbose=False):
        datalist = []
        if verbose: print('Preparing stock data for request')
        for t in request:
            if verbose: print(f'Retrieving data for {t}')
            if t in self.tickers_available.index:
                self.tickers.append(t)
                if self.tickers_available.loc[t, 'assetType'] == 'Stock':
                    self.data[t] = Stock(t, verbose)
                    datalist.append(self.data[t])
                else:
                    self.data[t] = ETF(t, verbose)
                    datalist.append(self.data[t])
            else:
                print(f'ERROR: {t} is not in the list of tickers')

        # create daily prices dataframe with all tickers and dates, and fill with NaN.
        # df = pd.DataFrame(index=pd.date_range('1/1/2000', '1/1/2050'), columns=self.tickers)
        # df.fillna(value=pd.np.nan, inplace=True)
        # # add column for each ticker
        # for t in self.tickers:
        #     df[t] = self.data[t].open
        #

        df = pd.DataFrame(self.data, index=self.data.keys())
        if len(df.columns) != len(request):
            print(f'col len: {len(df.columns)}, req len: {len(request)}')
            print('ERROR: Some tickers were not found')
        else:
            if verbose: print('All tickers were found successfully')

        return datalist

class ETF:
    """
    ETF Class
    Attributes:
    - daily_metadata, time_series_daily
    - weekly_metadata, time_series_weekly
    - monthly_metadata, time_series_monthly
    """

    def __init__(self, t, verbose=False):
        temp = api.get_timeseries(t, 'TIME_SERIES_DAILY_ADJUSTED', verbose)
        if temp[0]:
            self.daily_metadata, self.daily = temp[1], temp[2]
        else:
            print(f'Time Series Daily data unavailable for {t}')
            self.daily_metadata, self.daily = None, None

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
    def __init__(self, t, verbose=False):
        temp = api.get_timeseries(t, 'TIME_SERIES_DAILY_ADJUSTED', verbose)
        if temp[0]:
            self.daily_metadata, self.daily = temp[1], temp[2]
        else:
            print(f'Time Series Daily data unavailable for {t}')
            self.daily_metadata, self.daily = None, None

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

        # temp = api.get_income_statement(t)
        # if temp[0]:
        #     self.income_statement = temp[1]
        # else:
        #     print(f'{t} is not in the list of tickers')
        #     self.income_statement = None
        #
        # temp = api.get_balance_sheet(t)
        # if temp[0]:
        #     self.balance_sheet = temp[1]
        # else:
        #     print(f'{t} is not in the list of tickers')
        #     self.balance_sheet = None
        #
        # # temp = api.get_earnings(t)
        # if temp[0]:
        #     self.annualEarnings, self.quarterlyEarnings = temp[1], temp[2]
        # else:
        #     self.annualEarnings = None
        #     self.quarterlyEarnings = None

        # temp = api.get_company_overview(t)
        # if temp[0]:
        #     self.companyOverview = temp[1]
        # else:
        #     self.companyOverview = None

    def buildDF(self):
        return self.daily

if __name__ == '__main__':
    dc = DataConfig()
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT','JNJ']
    data = dc.getdata(request)