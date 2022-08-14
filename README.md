
# Stock Price Forecasting using an array of LSTM Models



## LSTM Model

To curate useful and accurate predictions, an ensemble type model was used. The ensemble consists of 5 LSTM models. Each model serves as a binary classifier of future price action. The target value is a weighted average of the adjusted closing price for the given stock over the `lookahead` time frame. 

The overall prediction is a weighted average of the 5 predictions from the LSTM Models. The weighting is based off of train/test performance of each model on historical data. 


### LSTM Model 1
>Predicts whether the given ticker will rise 5% in the given timeframe.

Trained on selection of 5000+ US Stocks. 

Necessary Steps:
1. Build and Train 5 models for various target types. 
2. Create script to take input ticker and aggregate model output into actionable prediction


Notes:
- larger absolute percentages for target change result in less accurate prediction, due to overfitting of train data. This is likely due to an overabundance of '0' target values. Overfitting should improve when using larger train dataset.

- Differing target change values will likely benefit from different model hyperparameters

readme change