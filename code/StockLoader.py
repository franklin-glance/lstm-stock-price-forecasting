import os
from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd
import tulipy as ti
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import dataconfig

dc = dataconfig.DataConfig()
import torch

'''
loads data for stocks using dataconfig and performs data preprocessing
'''


def print_info(indicator):
    """
    prints info about the tulipy indicator
    :param indicator: tulipy indicator
    """
    print("Type:", indicator.type)
    print("Full Name:", indicator.full_name)
    print("Inputs:", indicator.inputs)
    print("Options:", indicator.options)
    print("Outputs:", indicator.outputs)


class StockLoader():
    def __init__(self, device='cpu'):
        """
        initializes StockLoader
        sets self.tickers_availabe to the dataconfig tickers_available list
        :param device: device to use for training
        """
        self.device = device
        self.tickers_available = dc.tickers_available
        self.npstocks = {}
        self.stocks = {}
        self.date_mappings = {}
        self.column_mappings = {}

    # def plot(self, data, start_date, end_date, symbol=None):
    #     daterange = (data.index >= start_date) & (data.index <= end_date)
    #     keys = data.columns
    #     plt.plot(data.index[daterange], data[keys[4]][daterange])
    #     plt.plot(data.index[daterange], data['RSI'][daterange])
    #     plt.plot(data.index[daterange], data['SMA'][daterange])
    #     plt.plot(data.index[daterange], data['VWMA'][daterange])
    #     plt.plot(data.index[daterange], data['BBAND_upper'][daterange])
    #     plt.plot(data.index[daterange], data['BBAND_lower'][daterange])
    #     plt.plot(data.index[daterange], data['BBAND_middle'][daterange])
    #     # plt.plot(data.index[daterange], data['daily_SMA'][daterange])
    #     # plot horizontal line at 70 and 30
    #     plt.axhline(y=70, color='r', linestyle='-')
    #     plt.axhline(y=30, color='r', linestyle='-')
    #     # plot title
    #     plt.title(f'{symbol} Daily Price')
    #     plt.show()

    def process(self, stock, symbol, train=True,
                RSI_window_length=14,
                SMA_window_length=30,
                VWMA_period=30,
                BBAND_period=5,
                BBAND_std=2,
                MACD_short_period=12,
                MACD_long_period=26,
                MACD_signal_period=9,
                target_price_change=None,
                lookahead=30,
                new=False
                ):
        """
        performs data preprocessing on stock data
        saves npstock, stock, date_mapping, column_mappings to self.npstocks, self.stocks, self.date_mappings, self.column_mappings
        :param stock: stock dataframe to process
        :param symbol: ticker symbol for stock
        :param train: if true,
        :param RSI_window_length: length of RSI window
        :param SMA_window_length: length of SMA window
        :param VWMA_period: period of VWMA
        :param BBAND_period: period of BBAND
        :param BBAND_std: standard deviation of BBAND
        :param MACD_short_period: short period of MACD
        :param MACD_long_period: long period of MACD
        :param MACD_signal_period: signal period of MACD
        :param target_price_change: type of target to use for training, if None, use simple +/- binary classification
        :return: npstock, stock, date_mapping, column_mappings
        """
        columns_available = stock.columns

        # convert volume datatype to float
        if 'volume' in columns_available:
            stock['volume'] = stock['volume'].astype(float)
            stock['volume'] = stock['volume'] + 1
        if 'close' in columns_available:
            stock['open'] = stock['open'].astype(float)
        if 'high' in columns_available:
            stock['high'] = stock['high'].astype(float)
        if 'low' in columns_available:
            stock['low'] = stock['low'].astype(float)
        if 'close' in columns_available:
            stock['close'] = stock['close'].astype(float)
        if 'adjusted_close' in columns_available:
            stock['adjusted_close'] = stock['adjusted_close'].astype(float)
        if 'dividend_amount' in columns_available:
            stock['dividend_amount'] = stock['dividend_amount'].astype(float)
        if 'split_coefficient' in columns_available:
            stock['split_coefficient'] = stock['split_coefficient'].astype(float)


        # Technical Indicators ##########################################################
        stock = stock.sort_index(ascending=True)
        # RSI
        rsi_df = pd.DataFrame(ti.rsi(stock['adjusted_close'].values, RSI_window_length))
        rsi_df.index = stock.index[RSI_window_length:]
        stock['RSI'] = rsi_df[0]

        # SMA
        sma_df = pd.DataFrame(ti.sma(stock['adjusted_close'].values, SMA_window_length))
        sma_df.index = stock.index[SMA_window_length - 1:]
        stock['SMA'] = sma_df[0]

        # volume weighted moving average
        vwma = pd.DataFrame(ti.vwma(stock['adjusted_close'].values, stock['volume'].values, VWMA_period))
        vwma.index = stock.index[VWMA_period - 1:]
        stock['VWMA'] = vwma[0]

        # bollinger bands
        lower, middle, upper = (ti.bbands(stock['adjusted_close'].values, BBAND_period, BBAND_std))
        lower = pd.DataFrame(lower)
        middle = pd.DataFrame(middle)
        upper = pd.DataFrame(upper)
        lower.index = stock.index[BBAND_period - 1:]
        middle.index = stock.index[BBAND_period - 1:]
        upper.index = stock.index[BBAND_period - 1:]
        stock['BBAND_lower'] = lower[0]
        stock['BBAND_middle'] = middle[0]
        stock['BBAND_upper'] = upper[0]

        # MACD
        macd, macdsignal, macdhist = ti.macd(stock['adjusted_close'].values, MACD_short_period, MACD_long_period,
                                             MACD_signal_period)
        macd = pd.DataFrame(macd)
        macdsignal = pd.DataFrame(macdsignal)
        macdhist = pd.DataFrame(macdhist)
        macd.index = stock.index[MACD_long_period - 1:]
        macdsignal.index = stock.index[MACD_long_period - 1:]
        macdhist.index = stock.index[MACD_long_period - 1:]
        stock['MACD'] = macd[0]
        stock['MACDSignal'] = macdsignal[0]
        stock['MACDHist'] = macdhist[0]

        del rsi_df, sma_df, vwma, lower, middle, upper, macd, macdsignal, macdhist
        del MACD_long_period, MACD_short_period, MACD_signal_period, BBAND_period, BBAND_std, VWMA_period, SMA_window_length, RSI_window_length

        # Feature Engineering ###########################################################
        # diff - difference between current adjusted_close and previous adjusted_close
        # stock['diff'] = stock['adjusted_close'].diff(1)
        # add features indicating past values.
        # 52 week high
        stock['52_week_high'] = stock['adjusted_close'].rolling(252, min_periods=252, closed='left').max()
        # 52 week low
        stock['52_week_low'] = stock['adjusted_close'].rolling(252, min_periods=252, closed='left').min()

        # we can adjust these weights to increase model accuracy/performance
        # weights -> array holding 30 weights between 0 and 1 for each day in the window.
        # weights are linearly increasing from 0 to 1.
        weights = np.linspace(6, 1, 30)
        weights = weights / weights.sum()

        # add features indicating past values.
        # TODO: implement lookahead so average future price can be based on different time periods
        stock = stock.sort_index(ascending=False)
        stock['average_future_price'] = stock['adjusted_close'].rolling(30, min_periods=30, closed='left').apply(
            lambda x: np.sum(weights * x))

        if new:
            print(f'Data is current price data')
            # TARGET -> Rise/Fall -> 1 if price is rising, 0 if falling
            stock['target'] = stock['average_future_price'] > stock['adjusted_close']
        elif target_price_change is None:
            print(f'Training data is simple +/- binary classification for {lookahead} days')
            # TARGET -> Rise/Fall -> 1 if price is rising, 0 if falling
            stock['target'] = stock['average_future_price'] > stock['adjusted_close']
            # change target to numeric
        elif target_price_change > 0:
            print(f'Target is {target_price_change}% rise for {lookahead} days')
            stock['target'] = stock['average_future_price'] > stock['adjusted_close'] * (1 + target_price_change / 100)
        else:
            print(f'Target is {target_price_change}% fall for {lookahead} days')
            stock['target'] = stock['average_future_price'] < stock['adjusted_close'] * (1 - target_price_change / 100)


        stock['target'] = stock['target'].astype(int)

        # convert stock.index to datetime
        stock.index = pd.to_datetime(stock.index)

        stock = stock.sort_index(ascending=True)

        # drop features we don't need
        stock = stock.drop(['close', 'dividend_amount', 'split_coefficient', 'average_future_price'], axis=1)
        # drop na rows
        stock = stock.dropna()

        # columns: ['adjusted_close', 'volume', 'average_future_price', 'RSI', 'SMA',
        #        'VWMA', 'BBAND_lower', 'BBAND_middle', 'BBAND_upper', 'MACD',
        #        'MACDSignal', 'MACDHist', '52_week_high', '52_week_low']

        col_order = ['open', 'high', 'low', 'adjusted_close', 'volume', 'RSI', 'SMA',
                     'VWMA', 'BBAND_lower', 'BBAND_middle', 'BBAND_upper', 'MACD',
                     'MACDSignal', 'MACDHist', '52_week_high', '52_week_low', 'target']
        # reorder columns in stock
        stock = stock[col_order]
        # TODO: check to make sure the order is right
        stock = stock.sort_index(ascending=False)

        target_index = len(col_order) - 1

        # create date_mapping for row numbers
        date_mapping = []
        for i, date in enumerate(stock.index):
            date_mapping.append(date)

        # create column mapping dict for future use
        column_mappings = {}
        for i, col in enumerate(col_order):
            column_mappings[i] = col

        # convert stock to numpy array
        npstock = stock.to_numpy(copy=True)

        # RESCALING STOCK ##################################################################
        # TODO: potentially drop volume as well
        # TODO: change average future price to simple % increase/decrease or categorical bullish/bearish
        # this would make it a binary classifier, see how that works.
        # now we must scale the data, using MinMaxScaler for all features except for the target and adjusted_close
        mms = MinMaxScaler()
        ss = StandardScaler()
        # ss columns = ['adjusted_close', 'volume', 'average_future_price']
        standard_scaler_columns = [0, 1, 2, 3, 4]
        # loop through the feature columns
        for i in range(npstock.shape[1]):
            # use standard scaler for Close, Target, and Volume
            if i in standard_scaler_columns:
                npstock[:, i] = ss.fit_transform(npstock[:, i].reshape(-1, 1)).reshape(-1)
            elif i != target_index:
                npstock[:, i] = mms.fit_transform(npstock[:, i].reshape(-1, 1)).reshape(-1)

        self.npstocks[symbol] = npstock
        self.date_mappings[symbol] = date_mapping
        self.column_mappings[symbol] = column_mappings
        self.stocks[symbol] = stock
        return npstock, stock, date_mapping, column_mappings

    def load_prediction_data(self, ticker, verbose=False, timestep=30):
        stock = dc.getdata(ticker, verbose=verbose, new=True)


        npstock, stock, date_mapping, column_mappings  = self.process(stock.daily, ticker, new=True)

        # select most recent timestep days to use for x
        x = npstock[:timestep, :]
        # x is a 3d tensor of shape (batch_size, sequence_length, input_size)
        # batch_size = 1
        # sequence_length = timestep
        # input_size = number of features (same)
        # drop last column of x
        x = x[:, :-1]

        xt = torch.from_numpy(x).float().to(self.device)
        return xt,  date_mapping[:timestep], stock[:timestep]




    def load(self, request, train=True, verbose=False, split_date='2015-01-01', batch_size=60, timestep=30,
             target_price_change=None, lookahead=30, allstocks=False, params=None):
        """
        prepares dataframe for stocks in request. performs data preprocessing
        :param request: list of stock symbols to load
        :param train: TODO: figure this out
        :param verbose: print out info while loading
        :param split_date: date to split data into train and test
        :return: dataloader objects with train or test data
        """
        # check if dataloader already exists for this request

        requestitr = 0
        self.data = dc.getdata(request, verbose=verbose, allstocks=allstocks)
        self.npstocks = {}
        # apply pre-processing to each stock in the request
        for i, s in enumerate(self.data):
            stock = s.daily
            # TODO: this is train test split
            if train:
                stock = stock.drop(stock.loc[stock.index > split_date].index)
            else:
                stock = stock.drop(stock.loc[stock.index < split_date].index)
            if len(stock) > 1000 and np.sum(stock['volume']) > 100*len(stock):
                print(f'Progress: {np.round(i/len(self.data)*100, 1)}% Processing {s.symbol}')
                self.process(stock, s.symbol, train=train, target_price_change=target_price_change, lookahead=lookahead)
                if verbose: print('loaded ' + request[requestitr])
            else:
                if verbose: print(f'skipping {request[requestitr]} because historical data is too short')
            requestitr += 1
        # create dataloader for self.npstocks





        return self.create_loader(self.npstocks, batch_size=batch_size, timestep=timestep, train=train)

        # if train:
        #     # make stock time data all the same length
        #     maxlen = math.inf
        #     for stock in self.npstocks.keys():
        #         if self.npstocks[stock][0].shape[0] < maxlen:
        #             maxlen = self.npstocks[stock][0].shape[0]
        #     # cut off at maxlen
        #     for stock in self.npstocks.keys():
        #         # TODO: change this part, :maxlen or :
        #         self.slicedStocks[stock] = self.npstocks[stock][0][:]

    def create_loader(self, npstocks, batch_size=60, train=True, shuffle=False, timestep=30):
        """
        creates a data loader for self.npstocks
        :param batch_size: batch size for the loader
        :param test: if false, uses batch_size, if true, the batch size is the length of stock data (i think) TODO: check this
        :param shuffle: if true, shuffles the data
        :param timestep: how far in the past the model can see
        :return: data loader
        """
        keys = list(npstocks.keys())
        stock_list = []
        for key in keys:
            # calculate the number of rows to drop to make even number of timesteps
            offset = int(npstocks[key].shape[0] % timestep)
            if offset != 0:
                npstocks[key] = npstocks[key][:-offset]
            stock_list.append(npstocks[key])
        data = np.concatenate(stock_list, axis=0)
        x = []
        y = []
        # x = data[:, :-1]
        # y = data[:, -1]
        for i in range(timestep, data.shape[0], timestep):
            x.append(data[i - timestep: i, :-1])
            y.append(data[i - timestep: i, -1])
        x, y = np.array(x), np.array(y)

        xt = torch.from_numpy(x).float().to(self.device)
        yt = torch.from_numpy(y).float().to(self.device)

        if train:
            loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)
        else:
            loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)

        return loader
    #
    # def create_test_input(self, ticker):
    #     processedstock = self.process(dc.getdata([ticker]).daily, )

    def getlocaltickers(self, path=os.getcwd() + '/cache'):
        """
        returns a list of tickers that are locally stored in path
        :return: list of locally stored tickers
        """
        # path = os.getcwd()
        # path += '/cache'
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

        availabletickers = []
        for f in onlyfiles:
            ticker = ''
            for i in range(len(list(f))):
                if i == 5:
                    ticker = None
                    break
                if f[i] != '_' and f[i].isalpha():
                    ticker += f[i]
                else:
                    break
            if ticker and ticker not in availabletickers: availabletickers.append(ticker)
        return availabletickers


if __name__ == '__main__':
    '''
    This is a test file for the StockLoader class.
    '''
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
               'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
               'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']

    sl = StockLoader()

    availabletickers = sl.getlocaltickers()
    request_all = sl.tickers_available.index
    # sl.load(['AAN-W'], verbose=True)
    # ldr = sl.load(request, train=True, verbose=True)

    pred = sl.load_prediction_data('TSLA', verbose=True)

