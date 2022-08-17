
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
> This is a project to predict stock prices using LSTM models.

## Features

This application predicts future stock price action using an ensemble of LSTM models. The models are built using PyTorch, and are trained using stock price data from the Alpha Vantage API. 

## Model Architecture
### LSTM Model 
To curate useful and accurate predictions, an ensemble type model was used. The ensemble consists of 5 LSTM models. Each model serves as a binary classifier of future price action. The target value is a weighted average of the adjusted closing price for the given stock over the `lookahead` time frame.

The overall prediction is a weighted average of the 5 predictions from the LSTM Models. The weighting is based off of train/test performance of each model on historical data. 

### Ensemble
![](code/assets/Flowchart-2.jpg)
---
Notes:
- larger absolute percentages for target change result in less accurate prediction, due to overfitting of train data. This is likely due to an overabundance of '0' target values. Overfitting should improve when using larger train dataset.
- Differing target change values will likely benefit from different model hyperparameters
---
The model utilizes `n` LSTM models to predict the future price action of a given stock. Each model is trained on over `50000` samples of historical stock price data (from IPO to `01-01-2015`). Then, the models are tested on price data from `01-01-2015` to present. The test accuracy represents the accuracy of the model on price predictions from `01-01-2015` to today. 

To predict the future price action of the stock, each model in the ensemble is passed the given stock ticker. Each model looks at `timestep` days of data, and creates a prediction between `0` and `1`. A prediction `>0.5` represents affirmation in the direction of the target for the given model. 
>Ex: If the given model has been trained to predict whether the stock will rise 5% in the future, a prediction of `>0.5` indicates that that model suggests the stock will increase 5% in the future.

The final prediction is the average of the predictions from the models in the ensemble. The weighting of each models predictions is based off of the test accuracy of each model, and the certainty of the prediction. 

## Training

### Training Overview

### Training Results

## Testing

### Testing Overview

### Testing Results


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
