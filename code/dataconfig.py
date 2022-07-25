import pandas as pd
import api


class DataConfig:
    def __init__(self):
        # List of Tickers
        self.tickers = api.get_status()[1]
        self.consumer_sentiment = api.get_consumer_sentiment()
        self.data = {}

    def getdata(self, request):
        for t in request:
            print(f'Retrieving data for {t}')
            if t in self.tickers.index:
                if self.tickers.loc[t, 'assetType'] == 'Stock':
                    self.data[t] = Stock(t)
                else:
                    self.data[t] = ETF(t)
            else:
                print(f'{t} is not in the list of tickers')
        df = pd.DataFrame(self.data, index=self.data.keys())
        return df


"""
ETF Class

Attributes:
- daily_metadata, time_series_daily
- weekly_metadata, time_series_weekly
- monthly_metadata, time_series_monthly

"""
class ETF:
    def __init__(self, t):
        temp = api.get_timeseries(t, 'TIME_SERIES_DAILY_ADJUSTED')
        if temp[0]:
            self.daily_metadata, self.daily = temp[1], temp[2]
        else:
            print(f'Time Series Daily data unavailable for {t}')
            self.daily_metadata, self.daily = None, None

        temp = api.get_timeseries(t, 'TIME_SERIES_WEEKLY_ADJUSTED')
        if temp[0]:
            self.weekly_metadata, self.weekly = temp[1], temp[2]
        else:
            print(f'Time Series Weekly data unavailable for {t}')
            self.weekly_metadata, self.weekly = None, None

        temp = api.get_timeseries(t, 'TIME_SERIES_MONTHLY_ADJUSTED')
        if temp[0]:
            self.monthly_metadata, self.monthly = temp[1], temp[2]
        else:
            print(f'Time Series Monthly data unavailable for {t}')
            self.monthly_metadata, self.monthly = None, None


class Stock:
    def __init__(self, t):
        temp = api.get_timeseries(t, 'TIME_SERIES_DAILY_ADJUSTED')
        if temp[0]:
            self.daily_metadata, self.daily = temp[1], temp[2]
        else:
            print(f'Time Series Daily data unavailable for {t}')
            self.daily_metadata, self.daily = None, None

        temp = api.get_timeseries(t, 'TIME_SERIES_WEEKLY_ADJUSTED')
        if temp[0]:
            self.weekly_metadata, self.weekly = temp[1], temp[2]
        else:
            print(f'Time Series Weekly data unavailable for {t}')
            self.weekly_metadata, self.weekly = None, None

        temp = api.get_timeseries(t, 'TIME_SERIES_MONTHLY_ADJUSTED')
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
