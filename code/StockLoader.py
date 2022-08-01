import math
import os
from os.path import isfile, join
from os import listdir

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dataconfig
import tulipy as ti
from sklearn.preprocessing import MinMaxScaler, StandardScaler
dc = dataconfig.DataConfig()
import torch
import torch.nn as nn
import torch.optim as optim
import model


def print_info(indicator):
    print("Type:", indicator.type)
    print("Full Name:", indicator.full_name)
    print("Inputs:", indicator.inputs)
    print("Options:", indicator.options)
    print("Outputs:", indicator.outputs)

class StockLoader():
    def __init__(self, device='cpu'):
        self.device = device
        self.tickers_available = dc.tickers_available

    def plot(self, data, start_date, end_date, symbol=None):
        daterange = (data.index >= start_date) & (data.index <= end_date)
        keys = data.columns
        plt.plot(data.index[daterange], data[keys[4]][daterange])
        plt.plot(data.index[daterange], data['RSI'][daterange])
        plt.plot(data.index[daterange], data['SMA'][daterange])
        plt.plot(data.index[daterange], data['VWMA'][daterange])
        plt.plot(data.index[daterange], data['BBAND_upper'][daterange])
        plt.plot(data.index[daterange], data['BBAND_lower'][daterange])
        plt.plot(data.index[daterange], data['BBAND_middle'][daterange])
        # plt.plot(data.index[daterange], data['daily_SMA'][daterange])
        # plot horizontal line at 70 and 30
        plt.axhline(y=70, color='r', linestyle='-')
        plt.axhline(y=30, color='r', linestyle='-')
        # plot title
        plt.title(f'{symbol} Daily Price')
        plt.show()

    def process(self, stock):
        # PARAMETERS For Technical Indicators
        RSI_window_length = 14  # window length for RSI
        SMA_window_length = 30  # window length for SMA
        VWMA_period = 30
        BBAND_period = 5
        BBAND_std = 2
        MACD_short_period = 12
        MACD_long_period = 26
        MACD_signal_period = 9

        # convert volume datatype to float
        stock['volume'] = stock['volume'].astype(float)

        # we can adjust these weights to increase model accuracy/performance
        # weights -> array holding 30 weights between 0 and 1 for each day in the window.
        # weights are linearly increasing from 0 to 1.
        weights = np.linspace(6, 1, 30)
        weights = weights / weights.sum()

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

        # plot(stock, '2022-01-01', '2022-07-06')

        stock['average_future_price'] = stock['adjusted_close'].rolling(30, min_periods=30, closed='left').apply(
            lambda x: np.sum(weights * x))
        # TARGET -> Rise/Fall -> 1 if price is rising, 0 if falling
        stock['target'] = stock['average_future_price'] > stock['adjusted_close']
        # change target to numeric
        stock['target'] = stock['target'].astype(int)

        # do this last !!!

        # drop features we don't need
        # lets keep these for now.
        # stock = stock.drop(['open', 'high', 'low', 'close', 'dividend_amount', 'split_coefficient'], axis=1)
        stock = stock.drop(['close', 'dividend_amount', 'split_coefficient'], axis=1)
        # drop na rows
        stock = stock.dropna()

        # columns: ['adjusted_close', 'volume', 'average_future_price', 'RSI', 'SMA',
        #        'VWMA', 'BBAND_lower', 'BBAND_middle', 'BBAND_upper', 'MACD',
        #        'MACDSignal', 'MACDHist', '52_week_high', '52_week_low']

        col_order = ['open', 'high', 'low', 'adjusted_close', 'volume', 'RSI', 'SMA',
                     'VWMA', 'BBAND_lower', 'BBAND_middle', 'BBAND_upper', 'MACD',
                     'MACDSignal', 'MACDHist', '52_week_high', '52_week_low', 'average_future_price', 'target']
        # reorder columns in stock
        stock = stock[col_order]

        target_index = len(col_order) - 1

        # create date_mapping for row numbers
        date_mapping = {}
        for i, date in enumerate(stock.index):
            date_mapping[i] = date

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
        standard_scaler_columns = [0, 1, 2, 3, 4, 16]
        # loop through the feature columns
        for i in range(npstock.shape[1]):
            # use standard scaler for Close, Target, and Volume
            if i in standard_scaler_columns:
                npstock[:, i] = ss.fit_transform(npstock[:, i].reshape(-1, 1)).reshape(-1)
            elif i != target_index:
                npstock[:, i] = mms.fit_transform(npstock[:, i].reshape(-1, 1)).reshape(-1)


        return npstock, stock, date_mapping, column_mappings


    def load(self, request, test=False):
        """
        prepares dataframe for stocks in request. performs data preprocessing
        :param request:
        :return:
        """
        requestitr = 0
        self.data = dc.getdata(request)
        self.npstocks = {}
        for s in self.data:
            stock = s.daily
            npstock, stock, date_mapping, column_mappings = self.process(stock)
            self.npstocks[request[requestitr]] = npstock
            requestitr += 1
        self.slicedStocks = {}
        if not test:
            # make stock time data all the same length
            maxlen = math.inf
            for stock in self.npstocks.keys():
                if self.npstocks[stock].shape[0] < maxlen:
                    maxlen = self.npstocks[stock].shape[0]
            # cut off at maxlen
            for stock in self.npstocks.keys():
                self.slicedStocks[stock] = self.npstocks[stock][:maxlen]



    def create_loader(self, stocks: dict, batch_size=60, test=False, shuffle=False, timestep=30):
        # batch size will be the length of the stock data.
        keys = list(stocks.keys())
        # batch_size = stocks[keys[0]].shape[0]

        stock_list = []
        for key in keys:
            stock_list.append(stocks[key])
        data = np.concatenate(stock_list, axis=0)
        x = []
        y = []
        # x = data[:, :-1]
        # y = data[:, -1]
        # TODO: adjust timestep/make it a param
        for i in range(timestep, data.shape[0]):
            x.append(data[i-timestep: i, :-1])
            y.append(data[i-timestep: i, -1])
        # for i in range(batch_size, data.shape[0], batch_size):
        #     x.append(data[i-batch_size:i, :-1])
        #     y.append(data[i-batch_size:i, -1])
        x, y = np.array(x), np.array(y)

        xt = torch.from_numpy(x).float()
        yt = torch.from_numpy(y).float()
        xt.to(self.device)
        yt.to(self.device)

        if not test: loader = DataLoader(TensorDataset(xt, yt), batch_size=batch_size, shuffle=shuffle)
        else: loader = DataLoader(TensorDataset(xt,yt), batch_size=stocks[keys[0]].shape[0], shuffle=shuffle)

        return loader

    def getlocaltickers(self):
        mypath = os.getcwd()
        mypath += '/cache'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

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
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
               'JNJ']
    BATCH_SIZE = 60
    sl = StockLoader()

    sl.load(request)
    loader = sl.create_loader(sl.slicedStocks, batch_size=BATCH_SIZE)

    model = model.Model(input_size=17, hidden_size=50, num_layers=4, dropout=0.25)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    NUM_EPOCHS = 10

    for epoch in range(NUM_EPOCHS):
        for i, (x, y) in enumerate(loader):
            optimizer.zero_grad()
            y_pred = model(x)
            # y = y.reshape(y.shape[0], y.shape[1], -1)
            y = y[:, -1].reshape(-1, 1)
            loss = criterion(y_pred, y)
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()
            # print the loss
            if i % 100 == 0:
                print(f"Epoch: {epoch + 1}, Step: {i + 1}, Loss: {loss.item():.4f}")