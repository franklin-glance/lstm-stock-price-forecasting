import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import StockLoader
import torch
import torch.nn as nn
import torch.optim as optim
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
                     input_size=17,
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
        self.sl.load(request)
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

        for epoch in range(num_epochs):
            num_correct = 0
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
                num_correct += test_accuracy(y_pred, y)
                if i % 100 == 0:
                    print(
                        f"Epoch: {epoch + 1}, Progress: {np.round(((i + 1)/len(self.train_loader))*100, 2)}, Loss: {loss.item():.4f}, Accuracy: {test_accuracy(y_pred, y) / y.shape[0]}")
            print(f"Accuracy: {num_correct / len(self.train_loader)}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.getcwd() + path)

    def load_model(self, path, input_size=17, hidden_size=50, num_layers=4, dropout=0.2, device='cpu'):
        self.model = model.Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                 device=device)
        self.model.load_state_dict(torch.load(os.getcwd() + path))
        self.model.eval()

    def test_model(self, test_tickers=None, batch_size=60):
        if self.model is None:
            print('Model is None')
            return

        self.testsl = StockLoader.StockLoader()
        if test_tickers is None:
            localtickers = self.testsl.getlocaltickers()
            test_request = np.random.choice(localtickers, 10, replace=False)
        else:
            test_request = test_tickers

        self.testsl.load(test_request)
        test_loader = self.testsl.create_loader(self.testsl.slicedStocks, batch_size=batch_size)

        total_available = 0
        num_correct = 0
        keys = list(self.testsl.slicedStocks.keys())
        for batch in test_loader:
            x, y = batch
            preds = self.model(x)
            y = y[:, -1].reshape(-1, 1)
            num_correct += test_accuracy(preds, y)
            total_available += preds.shape[0]
        print(f'Accuracy: {num_correct / total_available}')

        def generate_prediction(self, ticker):
            print(f'Generating Prediction for Ticker: {ticker}')





if __name__ == '__main__':
    sp = StockPredictor()
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT','JNJ']
    sp.create_model()
    sp.load(request)
