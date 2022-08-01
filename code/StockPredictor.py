import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import StockLoader
import model


def test_accuracy(y_pred, y):
    correct = 0
    for i in range(y_pred.shape[0] - 1):
        target = y[i].item()
        value = y_pred[i].item()
        if np.abs(value - target) < 0.5:
            correct += 1
    return correct


class StockPredictor():
    def __init__(self, device='cpu'):
        self.sl = StockLoader.StockLoader()
        self.model = model.Model()
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.model_params = []

    def create_model(self,
                     input_size=16,
                     hidden_size=50,
                     num_layers=4,
                     dropout=0.2,
                     device='cpu',
                     learning_rate=0.001):
        self.model_params = [input_size, hidden_size, num_layers, dropout, device, learning_rate]
        self.model = model.Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                 device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def load(self, request, batch_size=60, verbose=False, train=True, timestep=30):
        self.sl.load(request, train)
        if train:
            if verbose: print('loading training data')
            self.train_loader = self.sl.create_loader(self.sl.slicedStocks, batch_size=batch_size, timestep=timestep)
        else:
            if verbose: print('loading test data')
            self.test_loader = self.sl.create_loader(self.sl.slicedStocks, batch_size=batch_size, timestep=timestep)
        if verbose: print('request loaded')

    def train_model(self, num_epochs=1, verbose=False):
        accuracy_log = {}
        if self.train_loader is None:
            print('please load data first')
            return

        total_correct = 0
        total_covered = 0
        training_start_time = time.time()
        for epoch in range(num_epochs):
            num_correct = 0
            total_seen = 0
            epoch_start_time = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                # x = x.to(self.device)
                # y = y.to(self.device)
                y_pred = self.model(x)
                # y = y.reshape(y.shape[0], y.shape[1], -1)
                y = y[:, -1].reshape(-1, 1)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                corr = test_accuracy(y_pred, y)
                num_correct += corr
                total_correct += num_correct
                total_covered += y.shape[0]
                if i % 100 == 0:
                    print(
                        f"Epoch: {epoch + 1}, Progress: {np.round(((i + 1) / len(self.train_loader)) * 100, 0)}%, Loss: {loss.item():.4f}, Accuracy: {np.round(corr / y.shape[0], 2)}")
            print(
                f"Accuracy: {np.round(num_correct / len(self.train_loader), 2)}, Epoch Time: {np.round(time.time() - epoch_start_time, 1)}s")
        print(f'Done Training, Total Time: {np.round(time.time() - training_start_time, 2)}s')

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.getcwd() + path)

    def load_model(self, path, input_size=16, hidden_size=50, num_layers=4, dropout=0.2, device='cpu'):
        self.model = model.Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                 device=device)
        self.model.load_state_dict(torch.load(os.getcwd() + path))
        self.model.eval()

    def test_model(self, test_tickers=None, batch_size=60, test_ticker_count=10, verbose=False):
        '''

        :param test_tickers: custom test ticker array (default to random selection)
        :param batch_size:
        :param test_ticker_count: number of random tickers to select (if no test_tickers are passed)
        :return:
        '''
        if self.model is None:
            print('Model is None')
            return

        self.testsl = StockLoader.StockLoader()
        if test_tickers is None:
            localtickers = self.testsl.getlocaltickers()
            test_request = np.random.choice(localtickers, test_ticker_count, replace=False)
            print(f'Testing model on: {test_request}')
        else:
            test_request = test_tickers

        self.testsl.load(test_request)
        test_loader = self.testsl.create_loader(self.testsl.slicedStocks, batch_size=0, test=True)

        total_available = 0
        num_correct = 0
        keys = list(self.testsl.slicedStocks.keys())
        for batch in test_loader:
            x, y = batch
            preds = self.model(x)
            y = y[:, -1].reshape(-1, 1)
            num_correct += test_accuracy(preds, y)
            total_available += preds.shape[0]
        print(f'Accuracy: {np.round(num_correct / total_available, 2)}')

        def generate_prediction(self, ticker):
            print(f'Generating Prediction for Ticker: {ticker}')


if __name__ == '__main__':
    sp = StockPredictor()
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
               'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
               'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']
    sp.create_model()
    sp.load(request, timestep=60, train=True)
    sp.test_model(request)
    sp.train_model(10)
    sp.save_model('/model/modelv3.pth')
    sp.test_model(request)
    sp.test_model(test_ticker_count=25)
