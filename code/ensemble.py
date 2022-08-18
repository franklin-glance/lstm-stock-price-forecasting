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
target_price_change, timestep, num_layers, hidden_size, dropout, learning_rate, num_epochs = [-1, 240, 2, 500, 0.1, 0.0001, 20]


class Ensemble:
    def __init__(self):
        self.sp = StockPredictor.StockPredictor()

    def predict(self, ticker):
        pc = ['None', '1.0', '-1.0',
              '3.0', '-3.0', '5.0', '-5.0', '10.0', '-10.0']
        # pc = ['None', '-5.0', '-1.0', '1.0', '5.0']  # price change targets used in the ensemble

        preds = []
        for target_price_change in pc:
            print('Predicting for target_price_change: ', target_price_change)
            self.sp.load_model(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                               path=f'/models/ensemble/pc_{target_price_change}_ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}.pth')
            preds.append(self.sp.generate_prediction(ticker, timestep=timestep))

        print(preds)
        print(pc)
        ans = list()
        for i in preds:
            temp = list()
            temp.append(i)
            if i > 0.5:
                temp.append('Affirmative')
            else:
                temp.append('Negative')
            ans.append(temp)
        results = dict(zip(pc, ans))
        return results


        # weighted_preds = []
        # pc_weights = [1, 25]  # weights for each pc
        # for i in range(len(preds)):
        #     weighted_preds.append(preds[i] * pc_weights[i])
        # print(weighted_preds)
        # print(sum(weighted_preds))
        # if sum(weighted_preds) < 0.5: overall prediction is negative
        # figure out how best to interpret predictions
        # graph bar chart with pc as x axis and preds as y axis
        # pc1 = ['+/-', '-5%', '-1%', '1%', '5%']
        # if sum(weighted_preds) < 0.5:
        #     plt.title(f'{ticker} - Negative, Weight: {sum(weighted_preds)}')
        # else:
        #     plt.title(f'{ticker} - Positive, Weight: {sum(weighted_preds)}')
        #
        # # make bar green if > .5 and red if < .5
        # colors = ['green' if x > .5 else 'red' for x in preds]
        # plt.bar(pc1, preds, color=colors)
        # plt.show()

def interpret_results(results, ticker):
    bull = None
    lowerbound = None
    upperbound = None
    done = False
    confidence = 0
    for key, value in results.items():
        print(key, value)
    for key, value in results.items():
        # print(key, value)
        if key != 'None': key = float(key)
        value[0] = float(value[0])
        if key == 'None':
            # First, we must determine if the stock is bullish or bearish, then we use the other keys to determine
            # the expected price change.
            # value[0] > 0.5: positive overall direction
            # value[0] < 0.5: negative overall direction
            if value[0] > 0.5:
                bull = True
                print('Positive overall direction')
                lowerbound = 0
            else:
                bull = False
                print('Negative overall direction')
                lowerbound = 0
        elif bull and key > 0 and value[0] > 0.5:
            # key > 0: target is positive
            # value > 0.5: prediction affirms positive direction
            # bull == True: overall direction is bullish
            lowerbound = key
            confidence += 1
            print(f'confirming overall direction at the {key}% level')
        elif bull and key < 0 and value[0] < 0.5:
            # key < 0: target is negative
            # value < 0.5: prediction affirms negative direction
            # bull == True: overall direction is bullish
            # this is a confirmatory negative prediction
            confidence += 1
            print(f'confirm not bearish at the {key}% level')
        elif not done and bull and key > 0 and value[0] < 0.5:
            # key > 0: target is positive
            # value < 0.5: prediction suggests negative direction
            # bull == True: overall direction is bullish
            # this key is the "upper bound" of the positive prediction
            upperbound = key
            done = True
            print(f'positive upper bound at the {key}% level')
        elif done and bull and key > 0 and value[0] < 0.5:
            confidence += 1
            print(f'prediction at {key}% level confirms upper bound at {upperbound}% level')
        elif done and bull and key > 0 and value[0] > 0.5:
            confidence -= 1
            print(f'prediction at {key}% contradicts {upperbound}% level')
        elif not bull and key < 0 and value[0] > 0.5:
            # key < 0: target is negative
            # value > 0.5: prediction affirms negative direction
            # bull == False: overall direction is bearish
            # this confirms negative movement key percentages
            lowerbound = key
            confidence += 1
            print(f'confirming overall direction at the {key}% level')
        elif not bull and key > 0 and value[0] < 0.5:
            # key > 0: target is positive
            # value < 0.5: prediction suggests negative direction
            # bull == False: overall direction is bearish
            # this key is the "upper bound" of the negative prediction
            confidence += 1
            print(f'confirming not bullish at the {key}% level')
        elif not done and not bull and key < 0 and value[0] < 0.5:
            # key < 0: target is negative
            # value < 0.5: prediction affirms negative direction
            # bull == False: overall direction is bearish
            # this indicates the 'upper bound' of negative price movement.
            upperbound = key
            done = True
            print(f'negative upper bound at the {key}% level')
        elif done and not bull and key < 0 and value[0] < 0.5:
            confidence += 1
            print(f'prediction at {key}% level confirms upper bound at {upperbound}% level')
        elif done and not bull and key < 0 and value[0] > 0.5:
            confidence -= 1
            print(f'prediction at {key}% contradicts {upperbound}% level')
        else:
            print('CONFOUNDING PREDICTION')
            print(f'key: {key}, value: {value}, bull: {bull}, lowerbound: {lowerbound}, upperbound: {upperbound}, done: {done}')

    # plot the chart for the given ticker
    x, time_data, price_data = Ensemble().sp.testsl.load_prediction_data(ticker, timestep=240)
    plt.plot(time_data, price_data['adjusted_close'], label=ticker)
    if bull:
        plt.suptitle(f'{ticker} - Bullish, Confidence: {confidence}')
    else:
        plt.suptitle(f'{ticker} - Bearish, Confidence: {confidence}')
    plt.title(f'Predicted Price Change: {lowerbound}% - {upperbound}%')

    # plot the predicted price change
    plt.show()
    return bull, lowerbound, upperbound, confidence, plt
if __name__ == '__main__':
    results = {}
    # take user input for ticker
    while True:
        ticker = input('Enter ticker: ')
        if ticker == '':
            break
        else:
            res = Ensemble().predict(ticker)
            results[ticker] = [res, interpret_results(res, ticker)]
