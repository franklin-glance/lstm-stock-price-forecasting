import StockPredictor
import sys

'''
Goal:
- Take user input, and predict stock price change using LSTM Models saved on disk.  
'''



# load ensemble of LSTM models from disk


if __name__ == '__main__':
    while True:
        ticker = input('Enter ticker: ')
        if ticker == '':
            break
