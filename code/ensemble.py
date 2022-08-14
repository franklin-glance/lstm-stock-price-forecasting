import StockPredictor
import sys
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt


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
            self.sp.load_model(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                               path=f'/models/ensemble/pc_{target_price_change}_ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}.pth')
            preds.append(self.sp.generate_prediction(ticker, timestep=timestep))

        print(preds)
        print(pc)

        weighted_preds = []
        pc_weights = [1, -.25, -.5, .5, .25] # weights for each pc
        for i in range(len(preds)):
            weighted_preds.append(preds[i] * pc_weights[i])
        print(weighted_preds)
        print(sum(weighted_preds))
        # if sum(weighted_preds) < 0.5: overall prediction is negative
        # figure out how best to interpret predictions
        # graph bar chart with pc as x axis and preds as y axis
        pc1 = ['+/-', '-5%', '-1%', '1%', '5%']
        if sum(weighted_preds) < 0.5:
            plt.title(f'{ticker} - Negative, Weight: {sum(weighted_preds)}')
        else:
            plt.title(f'{ticker} - Positive, Weight: {sum(weighted_preds)}')

        # make bar green if > .5 and red if < .5
        colors = ['green' if x > .5 else 'red' for x in preds]
        plt.bar(pc1, preds, color=colors)
        plt.show()


if __name__ == '__main__':
    Ensemble().predict(sys.argv[1])