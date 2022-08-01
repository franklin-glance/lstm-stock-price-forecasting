import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import StockLoader
import torch
import torch.nn as nn
import torch.optim as optim
import model

# device = torch.device('mps' if torch.has_mps else 'cpu')
device = torch.device('cpu')

sl = StockLoader.StockLoader()
request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT', 'JNJ']
localtickers = sl.getlocaltickers()
# request = localtickers[:25]
BATCH_SIZE = 60

sl.load(request)
train_loader = sl.create_loader(sl.slicedStocks, batch_size=BATCH_SIZE)


# TODO: tune hidden_size, num_layers, dropout
# input_size -> number of features in x
# hidden_size -> number of features in hidden state h
# num_layers -> number of recurrent layers
# dropout -> if nonzero, introduces a dropout layer on the outputs of each LSTM layer
model = model.Model(input_size=17, hidden_size=50, num_layers=4, dropout=0.2, device=device)
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def test_accuracy(y_pred,y):
    correct = 0
    for i in range(y_pred.shape[0]-1):
        target = y[i].item()
        value = y_pred[i].item()
        if np.abs(value - target) < 0.5:
            correct += 1
    return correct

NUM_EPOCHS = 10

print('Beginning Training')
for epoch in range(NUM_EPOCHS):
    num_correct = 0
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        # y = y.reshape(y.shape[0], y.shape[1], -1)
        y = y[:, -1].reshape(-1, 1)
        loss = criterion(y_pred, y)
        # backward pass
        loss.backward()
        # update the weights
        optimizer.step()
        # print the loss
        num_correct += test_accuracy(y_pred, y)
        if i % 100 == 0:
            print(f"Epoch: {epoch + 1}, Step: {i + 1}, Loss: {loss.item():.4f}, Accuracy: {test_accuracy(y_pred, y)/60}")
    print(f"Accuracy: {num_correct/len(train_loader)}")

testsl = StockLoader.StockLoader()

localtickers = testsl.getlocaltickers()

# randomly pick 10 stocks from localtickers
test_request = np.random.choice(localtickers, 10, replace=False)

testsl.load(test_request)
test_loader = sl.create_loader(testsl.slicedStocks, batch_size=BATCH_SIZE)
total_available = 0
num_correct = 0
keys = list(testsl.slicedStocks.keys())
print(f'Test Data Shape: {testsl.slicedStocks[keys[0]].shape}')

for batch in test_loader:
    x, y = batch
    preds = model(x)
    y = y[:, -1].reshape(-1,1)
    num_correct += test_accuracy(preds, y)
    total_available += preds.shape[0]

print(f'Accuracy: {num_correct/total_available}')
