import StockPredictor
import sys
import torch

'''
Goal:
- Train several LSTM models with varying training target change parameters. 
- Save the models to disk.
- Test the models on the test set.
- Save the test accuracy to disk.

'''

request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
           'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
           'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']

'''
possible Best params from model eval:
timestep: 480 
num_layers: 4
hidden_size: 500 
dropout: 0.1
learning_rate: 0.0001
num_epochs: 20
lookahead: 240
'''

# we will use abbreviated params and train data to save time
# timestep = 200
# num_layers = 2
# hidden_size = 100
# dropout = 0.2
# learning_rate = 0.001
# num_epochs = 5


_, target_price_change, timestep, num_layers, hidden_size, dropout, learning_rate, num_epochs, lookahead, cuda_num = sys.argv

device = torch.device(f'cuda:{cuda_num}'  if torch.cuda.is_available() else 'cpu')
sp = StockPredictor.StockPredictor(device=device)




# lookahead = 120 # lookahead is the number of days we will predict ahead
lookahead = int(lookahead)
target_price_change = float(target_price_change)
timestep = int(timestep)
num_layers = int(num_layers)
hidden_size = int(hidden_size)
dropout = float(dropout)
learning_rate = float(learning_rate)
num_epochs = int(num_epochs)


if target_price_change == 0:
    target_price_change = None
print(f'''
Parameters:
timestep: {timestep}
num_layers: {num_layers}
hidden_size: {hidden_size}
dropout: {dropout}
learning_rate: {learning_rate}
num_epochs: {num_epochs}
target_price_change: {target_price_change}
lookahead: {lookahead}
trained_on: allstocks
device: {device}
''')

verbose = True

sp.tb.add_text('Model Params: ', f'''
timestep: {timestep}
num_layers: {num_layers}
hidden_size: {hidden_size}
dropout: {dropout}
learning_rate: {learning_rate}
num_epochs: {num_epochs}
target_price_change: {target_price_change}
lookahead: {lookahead}
trained_on: allstocks
''')

request = sp.sl.getlocaltickers()

params = [target_price_change,timestep]
sp.load(request, timestep=timestep, verbose=verbose, target_price_change=target_price_change, allstocks=True, params=params, lookahead=lookahead, device=device)

sp.create_model(device=device, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout, learning_rate=learning_rate)

sp.train_model(num_epochs=num_epochs, verbose=verbose)

sp.save_model(f'/models/ensemble/pc_{target_price_change}_ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}_look-{lookahead}.pth')

test_accuracy = sp.test_model(request, verbose=verbose, target_price_change=target_price_change, lookahead=lookahead)

sp.tb.add_text('Test Accuracy: ', f'{test_accuracy}')