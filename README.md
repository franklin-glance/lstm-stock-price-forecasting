
# Stock Price Forecasting with LSTM Models
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
  - [LSTM Model](#lstm-model)
  - [Ensemble Model](#ensemble)
- [Training](#training)
  - [Training Overview](#training-overview)
  - [Training Results](#training-results)
- [Testing](#testing)
  - [Testing Overview](#testing-overview)
  - [Testing Results](#testing-results)
- [Train/Test Metrics](#train-test-metrics)
- [Conclusion](#conclusion)
- [Project Files](#project-files)

## Introduction
> This application forecasts stock prices using a series of LSTM models (ensemble). The models are trained on daily stock price  data (open, close, volume) up to 2015, and tested on daily price data from 2015 to present. After training, the models can be used to make accurate predictions of future stock price action. 

## Features
Currently, the model can predict future price movement of a given stock/etf. It can predict future price movement for three different time periods (6 weeks, 24 weeks, and 48 weeks). The model is more accuract on the shorter time periods, and less accurate on the longer time periods. 

The output consists of the following information:
- Affirmative/Negative Prediction for each model. 
- Certainty of Affirmative/Negative Prediction
- Overall Prediction Confidence
- Forecasted price movement (ex: between +1% and +5% in the next 6 weeks)
- Past price movement in the form of a chart.

Example: **TODO:** add example

## Model Architecture

To predict future price data, a ML model with time series capability is necessary. Recurrent neural networks (RNNs) are a popular choice for this purpose. Connections

This application predicts future stock price action using an ensemble of LSTM models. The models are built using PyTorch, and are trained using stock price data from the Alpha Vantage API. 


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

To curate useful and accurate predictions, an ensemble type model was used. The ensemble consists of `n` LSTM models. Each model serves as a binary classifier of future price action. The target value is a weighted average of the adjusted closing price for the given stock over the `lookahead` time frame.

`n` -> the number of LSTM models in the ensemble. One model is needed for binary classification at each price level. The ensemble diagram depicts an example using 7 models, which allows us to classify future price action in the following way:

#### 7 LSTM Ensemble
![](code/assets/Flowchart-2.jpg)
> - Positive Movement
>   - The expected price change for the given equity is +5% or greater over the next `timestep` days
>   - The expected price change for the given equity is between +3% and +5% over the next `timestep` days
>   - The expected price change for the given equity is between +1% and +3% over the next `timestep` days
>   - The expected price change for the given equity is between 0% and +1% over the next `timestep` days
> - Negative Movement
>   - The expected price change for the given equity is -5% or less over the next `timestep` days 
>   - The expected price change for the given equity is between -3% and -5% over the next `timestep` days
>   - The expected price change for the given equity is between -1% and -3% over the next `timestep` days
>   - The expected price change for the given equity is between 0% and -1% over the next `timestep` days


To predict the future price action of the stock, each model in the ensemble is passed the given stock ticker. Each model looks at `timestep` days of data, and creates a prediction between `0` and `1`. A prediction `>0.5` represents affirmation in the direction of the target for the given model. 
>Ex: If the given model has been trained to predict whether the stock will rise 5% in the future, a prediction of `>0.5` indicates that that model suggests the stock will increase 5% in the future.

The final prediction is the average of the predictions from the models in the ensemble. The weighting of each models predictions is based off of the test accuracy of each model, and the certainty of the prediction. 

## Training

### Training Overview

### Training Results

## Testing

### Testing Overview

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

## Train/Test Metrics 
> Each LSTM Model is trained on `~50000` samples of historical stock price data. The test accuracy is the accuracy of the model on price predictions from `01-01-2015` to today.
> Future work tuning model hyperparameters is needed to improve test accuracy. 

The `(target)` value indicates the LSTM Model's target value. For example, data preprocessing for a model with a target of `+1.0%` marks each batch with `1` if the _weighted average price_ over the next `30 trading days` is greater than `1.0%`, and `0` otherwise.

**Calculation of weights for _Weighted Average Price for Future Forecasting_:**
>  The weights for the _weighted average future price_ linearly increasing over time throughout the given interval (in this case, 30 days). Specifically, the weights vary from 1-6, so the stock closing price on the 30th day of the interval holds 6x more weight than the closing price of the first day. 

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


## Conclusion


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
