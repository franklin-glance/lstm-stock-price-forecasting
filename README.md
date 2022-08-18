
# Stock Price Forecasting with LSTM Models
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
  - [LSTM Model](#lstm-model)
  - [Ensemble Model](#ensemble)
- [Training](#training)
  - [Feature Engineering](#feature-engineering)
  - [Training Overview](#training-overview)
  - [Training Results](#training-results)
- [Testing](#testing)
  - [Testing Overview](#testing-overview)
  - [Testing Results](#testing-results)
- [Conclusion](#conclusion)
  - [Example Output](#example-output)
  - [Current State](#current-state)
- [Project Files](#project-files)

## Introduction
This application predicts stock prices using a series of LSTM models (ensemble). The models are trained on daily stock price  data (open, close, volume) up to 2015, and tested on daily price data from 2015 to present. After training, the models can be used to make accurate predictions of future stock price action. 
It is designed to be a tool to supplement a swing trader's technical analysis and trading strategy. It is not intended to be the sole decision maker for a trader, rather, the trader must use the results of this application to make their own trading decisions.

## Features
Currently, the model can predict future price movement of a given stock/etf. It can predict future price movement for three different time periods (6 weeks, 24 weeks, and 48 weeks). The model is more accuracte on the shorter time periods, and less accurate on the longer time periods. 

The output consists of the following information:
- Affirmative/Negative Prediction for each model. 
- Certainty of Affirmative/Negative Prediction
- Overall Prediction Confidence
- Forecasted price movement (ex: between +1% and +5% in the next 6 weeks)
- Past price movement in the form of a chart.

Example: **TODO:** add example

## Model Architecture

To predict future price data, it is necessary to use a ML model that has some notion of memory, or a way to include previous data points in the current prediction. Recurrent neural networks (RNNs) are a popular choice for this purpose. RNNs use a *hidden state* to hold the output from previous data points, and combines the *hidden state* with current input when making a prediction. However, RNNs suffer from a problem called *vanishing gradients,* which becomes an issue when trying to predict price action using a long series of past data. The gradients within an RNN carry information regarding past sequences of data, and when given long sequences, the gradients can become too small to be useful. 

The *vanishing gradient* problem is solved using LSTMs, which are long short-term memory (LSTM) units. LSTMs are a type of RNN that use a *cell state* to hold the output from previous data points, and combines the *cell state* with current input when making a prediction. LSTMs are more effective at learning long-term patterns than RNNs, and are more effective at predicting price action over a long sequence of data.

This application predicts future stock price action using an ensemble of LSTM models. The models are built using PyTorch, and are trained using stock price data from the Alpha Vantage API. 

For a better explanation of LSTMs, see [this article](https://medium.datadriveninvestor.com/how-do-lstm-networks-solve-the-problem-of-vanishing-gradients-a6784971a577)
### LSTM Model 


The model utilizes `n` LSTM models to predict the future price action of a given stock. Each model is trained on over `50000` samples of historical stock price data (from IPO to `01-01-2015`). Then, the models are tested on price data from `01-01-2015` to present. The test accuracy represents the accuracy of the model on price predictions from `01-01-2015` to today. 

``` python
Model(
  (lstm): LSTM(16, 50, num_layers=2, batch_first=True, dropout=0.2)
  (fc): Linear(in_features=50, out_features=1, bias=True)
)
```


The following diagram depicts the tensorboard visualization of the model. 

<img src='code/assets/model.png' height=800 />

### Ensemble

To curate useful and accurate predictions, an ensemble of LSTM models was used. The ensemble consists of `n` LSTM models. Each model serves as a binary classifier of future price action. The target value is a weighted average of the adjusted closing price for the given stock over the `lookahead` time frame.

---
**7 LSTM Ensemble**
![](code/assets/Flowchart-2.jpg)
**Outputs:**
- **Positive Movement**
  - The expected price change for the given equity is +5% or greater over the next `timestep` days
  - The expected price change for the given equity is between +3% and +5% over the next `timestep` days
  - The expected price change for the given equity is between +1% and +3% over the next `timestep` days
  - The expected price change for the given equity is between 0% and +1% over the next `timestep` days
- **Negative Movement**
  - The expected price change for the given equity is -5% or less over the next `timestep` days 
  - The expected price change for the given equity is between -3% and -5% over the next `timestep` days
  - The expected price change for the given equity is between -1% and -3% over the next `timestep` days
  - The expected price change for the given equity is between 0% and -1% over the next `timestep` days
> The model output will follow the preceeding format, along with the certainty of the prediction. Certainty is calculated using a combination of testing accuracy for each model along with the number of models in agreement with the given prediction
---

To predict the future price action of the stock, each model in the ensemble is passed the given stock ticker. Each model looks at `timestep` days of data, and creates a prediction between `0` and `1`. A prediction `>0.5` represents affirmation in the direction of the target for the given model. 
>Ex: If the given model has been trained to predict whether the stock will rise 5% in the future, a prediction of `>0.5` indicates that that model suggests the stock will increase 5% in the future.
## Training
Stock price data is collected from the Alpha Vantage API. A series of over 50000 different companies with IPO dates prior to 2010 were used to train the models. 
### Feature Engineering
The stock price data collected from the Alpha Vantage API contains daily prices with the following features:
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

The 'close' values were dropped in favor of the using Adjusted Close. Several Features were extracted from the price data to create a total of 16 input features and 1 target feature. 

**Input Features:**
- Open
- High
- Low
- Adjusted Close
- Volume
- RSI (Relative Strength Index)
- SMA (Simple Moving Average)
- VWMA (Volume Weighted Moving Average)
- BB_lower (Bollinger Band Lower)
- BB_upper (Bollinger Band Upper)
- BB_middle (Bollinger Band Middle)
- MACD (Moving Average Convergence Divergence)
- MACD_signal (Moving Average Convergence Divergence Signal)
- MACD_hist (Moving Average Convergence Divergence Histogram)
- 52 Week High
- 52 Week Low


**Target:**
Example target: for params `lookahead` (number of days in the future to predict the price for), `target_price_change`, `new` (True if the model is being used to generate current prediction)`, the target is calculated in the following way:
  ```python
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
  ```
**Calculation of weights for `average_future_price`:**
The weights used in calculating the `average_future_price` for a given day linearly increase over time throughout the given interval (in this case, 30 days in the future). The weights vary from 1-6, so the stock closing price on the 30th day of the interval holds 6x more weight than the closing price of the first day. 
### Training Overview
Each LSTM Model is trained on `~50000` samples of historical stock price data. The test accuracy is the accuracy of the model on price predictions from `01-01-2015` to today.
Future work tuning model hyperparameters is needed to improve test accuracy. 

The `(target)` value indicates the LSTM Model's target value. For example, data preprocessing for a model with a target of `+1.0%` marks each batch with `1` if the _weighted average price_ over the next `30 trading days` is greater than `1.0%`, and `0` otherwise.


### Training Results
The following table lists train/test results for a lookahead of `30` days and target price changes ranging from `-10.0%` to `+10.0%`.
| Model (target)        | Params                                                                                                                                                               | Train                                       | Test               |
|-----------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------|--------------------|
| LSTM Model 1 (+/-)    | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: None (+/-)` | `num_samples: 41340`<br/>`Accuracy: 0.9175` | `Accuracy: 0.8814` |
| LSTM Model 2 (+1.0%)  | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: +1.0%`      | `num_samples: 41340`<br/>`Accuracy: 0.9112` | `Accuracy: 0.8875` |
| LSTM Model 3 (-1.0%)  | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: -1.0%`      | `num_samples: 41340`<br/>`Accuracy: 0.9091` | `Accuracy: 0.8887` |
| LSTM Model 4 (+3.0%)  | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: (+/-)`      | `num_samples: 41340`<br/>`Accuracy: 0.8982` | `Accuracy: 0.8598` |
| LSTM Model 5 (-3.0%)  | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: None (+/-)` | `num_samples: 41340`<br/>`Accuracy: 0.8971` | `Accuracy: 0.8496` |
| LSTM Model 6 (+5.0%)  | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: None (+/-)` | `num_samples: 41340`<br/>`Accuracy: 0.9126` | `Accuracy: 0.8579` |
| LSTM Model 7 (-5.0%)  | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: None (+/-)` | `num_samples: 41340`<br/>`Accuracy: 0.9098` | `Accuracy: 0.8603` |
| LSTM Model 8 (+10.0%) | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: None (+/-)` | `num_samples: 41340`<br/>`Accuracy: 0.9329` | `Accuracy: 0.8420` |
| LSTM Model 9 (-10.0%) | `timestep: 240`<br/>`num_layers: 2`<br/>`hidden_size: 500`<br/>`dropout: 0.1`<br/>`learning_rate: 0.0001`<br/>`num_epochs: 20`<br/>`target_price_change: None (+/-)` | `num_samples: 41340`<br/>`Accuracy: 0.9304` | `Accuracy: 0.8900` |

## Testing
### Testing Overview
Each Model is tested on 40000+ samples of stock price data from `01-01-2015` to present. The test accuracy is simply the number of correct predictions divided by the number of total predictions. 

### Testing Results
| Target (%) | Accuracy (30 days) | Accuracy (60 days) | Accuracy (120 days) |
| ---------- | ------------------ | ------------------ | ------------------- |
| +/- \*     | 0.8814             | 0.6                | 0.6                 |
| +1.0       | 0.9975             | 0.67               | 0.6                 |
| \-1.0      | 0.8887             | 0.51               | 0.6                 |
| +3.0       | 0.8598             | 0.7                | 0.62                |
| \-3.0      | 0.8496             | 0.71               | 0.62                |
| +5.0       | 0.8579             | 0.71               | 0.71                |
| \-5.0      | 0.8603             | 0.7                | 0.62                |
| +10.0      | 0.842              | 0.77               | 0.72                |
| \-10.0     | 0.89               | 0.78               | 0.67                |
| +15.0      | N/A                | 0.83               | 0.73                |
| \-15.0     | N/A                | 0.83               | 0.71                |
| +20.0      | N/A                | 0.91               | 0.76                |
| \-20.0     | N/A                | 0.85               | 0.72                |

## Conclusion

### Example Output
`ensemble.py` provides a simple interface for producing a prediction using an ensemble of models. The output for ticker `TSLA` on `Aug 18, 2022` is the following: 
``` 
None [0.999534010887146, 'Affirmative']  # None -> represents the +/- overall price direction. 0.99 > 0.5 represents Affirmation in positive price change
1.0 [0.9972209930419922, 'Affirmative']
-1.0 [0.0020713915582746267, 'Negative']
3.0 [0.9637451171875, 'Affirmative']
-3.0 [0.07711449265480042, 'Negative']
5.0 [0.647577702999115, 'Affirmative']
-5.0 [0.35542991757392883, 'Negative']
10.0 [0.062117259949445724, 'Negative']
-10.0 [0.9693275094032288, 'Affirmative']
Positive overall direction
confirming overall direction at the 1.0% level
confirm not bearish at the -1.0% level
confirming overall direction at the 3.0% level
confirm not bearish at the -3.0% level
confirming overall direction at the 5.0% level
confirm not bearish at the -5.0% level
positive upper bound at the 10.0% level
key: -10.0, value: [0.9693275094032288, 'Affirmative'], bull: True, lowerbound: 5.0, upperbound: 10.0, done: True
```
This output suggests that `TSLA` stock price will move in the positive direction between 5% and 10% over the next 6 weeks. 
Current `TSLA` stock price is $913.00. The lower price target is $958.65 and the upper price target is $1004.30. 

Similar output can be produced for the 60 and 120 day intervals. 

### Current State
The model needs more work to produce accurate predictions for the 60 and 120 day timeframes. The 30 day predictions are provably accuracte on unseen backtesting data. I would like to shift the accuracy scores of the 120 and 60 day models to be closer to 0.9. I plan on implementing this project using a Transformer architecture rather than a LSTM to allow the model to see more past data without suffering from vanishing gradients. 

Futher work is needed to improve the accuracy of each model. The amount of hyperparameter tuning done so far has been minimal; hyperparameter tuning likely will increase accuracy and reduce overfitting of the models. Additionally, I plan on training models on more price data from Forex and Foreign Stocks. 
## Project Files

| Filename              | Description                                                                                                                                                 |
|-----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `api.py`              | Talks to alphavantage API to get stock data                                                                                                                 |
| `dataconfig.py`       | Handles loading and storing of stock data from api/cache                                                                                                    |
| `env.py`              | Contains API key. Mitigates API calls to ensure API request limit is not reached.                                                                           |
| `frontend.py`         |                                                                                                                                                             |
| `StockLoader.py`      | Interacts with DataConfig to load stock data. Performs appropriate data preprocessing and builds the desired DataLoader object.                             |
| `StockPredictor.py`   | The StockPredictor class is the primary interface to train and use LSTM Models. Inner methods call the StockLoader class to prepare the desired Stock Data. |
| `ensemble_trainer.py` | Trains several LSTM Models with varying target parameters, to be used in `ensemble.py` for the final prediction.                                            |
| `ensemble.py`         | Utilizes saved LSTM Models to create prediction for given stock ticker.                                                                                     |
| `market_analysis.py`  | Loads LSTM Model ensemble from disk and scans stock data to output top picks for stocks.                                                                    |
| `model.py`            | The LSTM model                                                                                                                                              |
---
