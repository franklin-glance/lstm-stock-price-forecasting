
# ml-trading

This is a tool to assist with technical analysis for short term traders. 
The model is built using pytorch and scikit learn, and trained on historical 
data from alphavantage.com. 

The intended use is to supplement an existing trading strategy, rather than using ML to manage the entire portfolio. 


## LSTM Model

To curate useful and accurate predictions, an ensemble type model was used. The ensemble consists of 5 LSTM models. Each model serves as a binary classifier. Further explanation is shown below. 

![Flowchart](https://user-images.githubusercontent.com/60630956/184504870-671eb809-c775-4f39-b75b-c277baa7d57c.jpg)
