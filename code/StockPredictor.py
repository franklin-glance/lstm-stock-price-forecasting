
import StockLoader

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle

from torch.utils.tensorboard import SummaryWriter

print('importing stockloader')
print('importing model')
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
        self.sl = StockLoader.StockLoader(device=device)
        self.testsl = StockLoader.StockLoader(device=device)
        self.model = model.Model(device=device)
        self.model.to(device)
        self.device = device
        self.train_loader = None
        self.test_loader = None
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.model_params = []
        self.tb = SummaryWriter()

    def create_model(self,
                     input_size=16,
                     hidden_size=100,
                     num_layers=4,
                     dropout=0.2,
                     device='cpu',
                     learning_rate=0.001):
        '''
        Creates a model with the given parameters
        :param input_size: the number of expected features in the input x
        :param hidden_size: the number of features in the hidden layer
        :param num_layers: the number of recurrent layers in the lstm
        :param dropout: if non-zero, introduces a dropout layer on the outputs of each LSTM layer except the last layer
        :param device: the device to use for the model
        :param learning_rate: the learning rate for the optimizer
        :return: model, optimizer
        '''
        self.model_params = [input_size, hidden_size, num_layers, dropout, device, learning_rate]
        self.model = model.Model(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                                 device=device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        return self.model, self.optimizer

    def load(self, request, batch_size=60, verbose=False, train=True, timestep=30, split_date='2015-01-01',
             target_price_change=None, lookahead=30, allstocks=False, params=None, device='cpu'):
        '''
        Loads the data for the model
        :param request: list of tickers to load into the test_loader (or train_loader)
        :param batch_size: the batch size for the data loader
        :param verbose: if true, prints out the progress of the data loader
        :param train: if true, loads the train_loader, else loads the test_loader
        :param timestep: the number of timesteps to use for the data (how far the model looks back in time)
        :return: loader object
        '''

        if train:
            # check if loader already exists
            if allstocks and os.path.isfile(os.getcwd() + f'/cache/train_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl'):
                with open(os.getcwd() + f'/cache/train_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl', 'rb') as f:
                    self.train_loader = pickle.load(f)
                if verbose: print('Train loader loaded from cache')
                return self.train_loader

            if verbose: print('loading training data')
            self.train_loader = self.sl.load(request, batch_size=batch_size, verbose=verbose, train=train, timestep=timestep,
                         split_date=split_date, target_price_change=target_price_change, lookahead=lookahead, allstocks=allstocks)
            if verbose: print('request loaded')
            # save to cache
            if allstocks:
                with open(os.getcwd() + f'/cache/train_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl', 'wb') as f:
                    pickle.dump(self.train_loader, f)
                if verbose: print('Train loader saved to cache')
            return self.train_loader
        else:
            if verbose: print('loading test data')
            # check if loader already exists
            if allstocks and os.path.isfile(os.getcwd() + f'/cache/test_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl'):
                with open(os.getcwd() + f'/cache/test_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl', 'rb') as f:
                    self.test_loader = pickle.load(f)
                if verbose: print('Test loader loaded from cache')
                return self.test_loader

            self.test_loader = self.sl.load(request, batch_size=batch_size, verbose=verbose, train=train,
                                             timestep=timestep,
                                             split_date=split_date, 
                                             target_price_change=target_price_change, lookahead=lookahead, allstocks=allstocks)
            if verbose: print('request loaded')
            # save to cache
            if allstocks:
                with open(os.getcwd() + f'/cache/test_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl', 'wb') as f:
                    pickle.dump(self.test_loader, f)
                if verbose: print('saved test loader to cache')
            return self.test_loader

    def train_model(self, num_epochs=1, verbose=False):

        self.model.to(self.device)
        accuracy_log = {}
        if self.train_loader is None:
            print('please load data first')
            return

        print(f'Length of train data: {len(self.train_loader) * self.train_loader.batch_size}')

        training_start_time = time.time()
        for epoch in range(num_epochs):
            last_correct = 0
            num_correct = 0
            total_loss=0
            num_batches=0
            epoch_start_time = time.time()
            for i, (x, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                # y = y.reshape(y.shape[0], y.shape[1], -1)
                y = y[:, -1].reshape(-1, 1)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                corr = test_accuracy(y_pred, y)
                # last_correct += 1 if np.abs(y_pred[-1].item() - y[-1].item()) < 0.5 else 0
                num_correct += corr
                num_batches +=1
                total_loss += loss.item()
                if verbose: print(f"Epoch: {epoch + 1}, Progress: {np.round(((i + 1) / len(self.train_loader)) * 100, 0)}%, Loss: {loss.item():.4f}, Accuracy: {np.round(corr / y.shape[0], 2)}")
            print(
                f"Epoch: {epoch+1}, Accuracy: {np.round(num_correct / (len(self.train_loader) * self.train_loader.batch_size) , 2)}, Epoch Time: {np.round(time.time() - epoch_start_time, 1)}s")
            self.tb.add_scalar('Loss', total_loss, epoch)
            self.tb.add_scalar('Average Loss', total_loss / num_batches, epoch)
            self.tb.add_scalar('Number Correct', num_correct, epoch)
            self.tb.add_scalar('Accuracy', num_correct / (len(self.train_loader) *self.train_loader.batch_size), epoch)
            self.tb.add_text('Number of Samples in Train set:', str(len(self.train_loader) * self.train_loader.batch_size))

            # tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
            # tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
            # tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)


        print(f'Done Training, Total Time: {np.round(time.time() - training_start_time, 2)}s')

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.getcwd() + path)
        # torch.save(self.model, os.getcwd() + path)
        print(f'Model saved to {path}')

    def load_model(self, path, input_size=16, hidden_size=50, num_layers=4,
                   dropout=0.2, device='cpu'):
        self.model = model.Model(input_size=input_size, hidden_size=hidden_size,
                                 num_layers=num_layers, dropout=dropout,
                                 device=device)
        self.model.load_state_dict(torch.load(os.getcwd() + path))
        self.model.eval()
        print(f'Model loaded from {path}')

    def test_model(self, test_tickers=None, batch_size=60, test_ticker_count=10, verbose=False, with_plot=False,
                   split_date='2015-01-01', timestep=30, lookahead=30, target_price_change=None, allstocks=False):
        '''
        :param test_tickers: custom test ticker array (default to random selection)
        :param batch_size:
        :param test_ticker_count: number of random tickers to select (if no test_tickers are passed)
        :param verbose: if true, prints out the progress of the data loader
        :param with_plot: if true, plots the test results

        :return:
        '''
        # if there is no model loaded, throw an error
        if self.model is None:
            print('Model is None')
            return

        # picking random test tickers if none are passed
        if test_tickers is None:
            localtickers = self.testsl.getlocaltickers()
            test_request = np.random.choice(localtickers, test_ticker_count, replace=False)
            print(f'Testing model on: {test_request}')
        else:
            test_request = test_tickers

        # checking if the test_loader has already been created and cached
        if allstocks and os.path.isfile(os.getcwd() + f'/cache/test_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl'):
            with open(os.getcwd() + f'/cache/test_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl', 'rb') as f:
                test_loader = pickle.load(f)
            if verbose: print('Test loader loaded from cache')
        else:
            test_loader = self.testsl.load(test_request, train=False, verbose=verbose, batch_size=batch_size,
                                           timestep=timestep, split_date=split_date,
                                           target_price_change=target_price_change, lookahead=lookahead, allstocks=allstocks)
            if verbose: print('Test loader created')
            if allstocks:
                with open(
                        os.getcwd() + f'/cache/test_loader_{timestep}_{split_date}_{target_price_change}_{lookahead}_{batch_size}.pkl',
                        'wb') as f:
                    pickle.dump(test_loader, f)
                if verbose: print('saved test loader to cache')


        # testing the model on the given test_loader
        num_correct = 0
        total_seen = 0
        for batch in test_loader:
            x, y = batch
            preds = self.model(x)
            y = y[:, -1].reshape(-1, 1)
            corr = test_accuracy(preds, y)
            num_correct += corr
            total_seen += y.shape[0]
        self.model.accuracy = num_correct / total_seen
        print(f'Accuracy: {np.round(num_correct / total_seen, 2)}')
        self.tb.add_text('Number of Samples in Test set:', str(len(test_loader) * test_loader.batch_size))
        return num_correct/total_seen



    def generate_prediction(self, ticker, timestep=252, verbose=False):
        if self.model is None:
            print('Please load/train model first')
            return
        print(f'Generating Prediction for Ticker: {ticker}')
        x, time_data, price_data = self.testsl.load_prediction_data(ticker, timestep=timestep)
        xt = x.reshape(-1, x.shape[0], x.shape[1])
        pred = self.model(xt)

        # plot the stock price and the prediction
        if pred.item() > 0.5:
            # TODO: adjust stockloader.process so that the future prediction timeframe can be changed
            print(f'Prediction: {ticker} will rise in the next 6 weeks')
            # plt.suptitle(f'{ticker} will rise in the next 6 weeks, Certainty: {self.model.accuracy - (1-pred.item()):.2f}')
            # plt.plot(time_data, price_data['adjusted_close'], label=ticker)
        else:
            print(f'Prediction: {ticker} will fall in the next 6 weeks')
            # plt.suptitle(f'{ticker} will fall in the next 6 weeks, Certainty: {self.model.accuracy - pred.item():.2f}')
            # plt.plot(time_data, price_data['adjusted_close'], label=ticker)
        # label plot
        # plt.title('Prediction: ' + str(pred.item()))
        # plt.legend()
        # plt.show()
        return pred.item()


def run_train_test():
    timestep = 200 # length of sequence for LSTM
    device='cpu'
    sp = StockPredictor(device=device)
    sp.create_model(device=device, num_layers=4, hidden_size=120, dropout=0.2)
    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
               'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
               'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']
    # request = ['AAPL', 'TSLA', 'IBM']

    allstocks = sp.sl.getlocaltickers()
    # random_stocks = np.random.choice(allstocks, 100, replace=False)
    sp.load(allstocks, timestep=timestep,verbose=True)
    sp.train_model(5, verbose=True)
    sp.save_model('/models/5epoch_4layers_120hidden_02dropout.pth')
    sp.test_model(allstocks, timestep=timestep, verbose=True)

    # take user input, while input is not empty, generate prediction for given ticker
    while True:
        ticker = input('Enter ticker: ')
        if ticker == '':
            break
        sp.generate_prediction(ticker, verbose=True, timestep=timestep)
        print('\n')

letters = {'a':'z', }


if __name__ == '__main__':
    '''
    Parameters:
    num_layers: 1,2,3,4,5
    hidden_size: 60, 120, 180, 240, 300, 360
    dropout: 0.1, 0.2, 0.3, 0.5, 0.7, 0.9
    timestep: 100, 200, 300, 400, 500
    '''


    request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
               'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
               'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']
    # request = ['AAPL', 'TSLA', 'IBM']

    # Create StockPredictior Object
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: ', device)
    sp = StockPredictor(device=device)
    
    '''
    Target Type and Lookahead
    
    sp.load has two important params. target_price_change and lookahead.
    target_price_change: integer representing the target price percentage change over 
    lookahead days
    lookahead: integer representing the number of days in the future to predict target_price_change price
    change
    
    '''

    timestep = 200 # length of sequence for LSTM
    target_price_change = None # target price percentage change over lookahead days
    sp.create_model(device=device)
    # sp.create_model(device=device, num_layers=4, hidden_size=1000, dropout=0.2)
    # sp.load_model(device=device, num_layers=4, hidden_size=1000, dropout=0.2, path='/models/codemodel_200timestep_4layers_120hidden_02dropout_5epoch.pth')
    # # sp.load_model('/models/codemodel_200timestep_4layers_120hidden_02dropout_5epoch.pth')
    # allstocks = sp.sl.getlocaltickers()
    # random_stocks = np.random.choice(allstocks, 100, replace=False)
    sp.load(request, timestep=timestep,verbose=True, target_price_change=target_price_change)
    sp.train_model(100, verbose=True)
    # sp.save_model('/models/')
    sp.test_model(request, timestep=timestep, verbose=True, target_price_change=target_price_change)


    # # # take user input, while input is not empty, generate prediction for given ticker
    while True:
        ticker = input('Enter ticker: ')
        if ticker == '':
            break
        sp.generate_prediction(ticker, verbose=True, timestep=timestep)
        print('\n')


'''


Test Results (when the model was broken):
1. 1000 random samples from available tickers -> accuracy: 0.89

Training on 1000 random samp

2. Testing on 6850 tickers, never seen before ->  


New Results:
1. Allstocks -> Train for 1 epoch, batch size 120, timestep 120, split date 2018-01-01 
-> accuracy: 0.83
-> test accuracy: 0.84



Training: (modelv
timestep=200
num_layers=4
hidden_size=120
dropout=0.2

train 5 epochs on allstocks
0.9 accuracy on allstocks

'''