
# Stock Price Forecasting with LSTM Models
This application predicts future stock price action using an ensemble of LSTM models. The models are built using PyTorch, and are trained using stock price data from the Alpha Vantage API. 

## Model Accuracy
The model accuracy is calculated by backtesting on unseen price data. 


## LSTM Model

To curate useful and accurate predictions, an ensemble type model was used. The ensemble consists of 5 LSTM models. Each model serves as a binary classifier of future price action. The target value is a weighted average of the adjusted closing price for the given stock over the `lookahead` time frame. 

The overall prediction is a weighted average of the 5 predictions from the LSTM Models. The weighting is based off of train/test performance of each model on historical data. 


Notes:
- larger absolute percentages for target change result in less accurate prediction, due to overfitting of train data. This is likely due to an overabundance of '0' target values. Overfitting should improve when using larger train dataset.
- Differing target change values will likely benefit from different model hyperparameters
---
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
## Ensemble Prediction

The model utilizes `n` LSTM models to predict the future price action of a given stock. Each model is trained on over `50000` samples of historical stock price data (from IPO to `01-01-2015`). Then, the models are tested on price data from `01-01-2015` to present. The test accuracy represents the accuracy of the model on price predictions from `01-01-2015` to today. 

To predict the future price action of the stock, each model in the ensemble is passed the given stock ticker. Each model looks at `timestep` days of data, and creates a prediction between `0` and `1`. A prediction `>0.5` represents affirmation in the direction of the target for the given model. 
>Ex: If the given model has been trained to predict whether the stock will rise 5% in the future, a prediction of `>0.5` indicates that that model suggests the stock will increase 5% in the future.

The final prediction is the average of the predictions from the models in the ensemble. The weighting of each models predictions is based off of the test accuracy of each model, and the certainty of the prediction. 

