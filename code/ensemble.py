import StockPredictor
import sys
import os
from os import listdir
from os.path import isfile, join


# def getmodels():
#     models = []
#     path = os.getcwd() + '/models/ensemble'
#     onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
#     for f in onlyfiles:
#
#
device = 'cpu'
timestep = 200
num_layers = 2
hidden_size = 100
dropout = 0.2
learning_rate = 0.001
num_epochs = 20


class Ensemble:
    def __init__(self):
        self.sp = StockPredictor.StockPredictor()

    def predict(self, ticker):
        pc = ['None', '-5.0', '-1.0', '1.0', '5.0'] # price change targets used in the ensemble
        preds = []
        for target_price_change in pc:
            print('Predicting for target_price_change: ', target_price_change)
            # self.sp.create_model(device=device, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout, learning_rate=learning_rate)
            self.sp.load_model(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                               path=f'/models/ensemble/pc_{target_price_change}_ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}.pth')
            preds.append(self.sp.generate_prediction(ticker, timestep=timestep))

        print(preds)
        print(pc)
        # figure out how best to interpret predictions



if __name__ == '__main__':
    Ensemble().predict(sys.argv[1])