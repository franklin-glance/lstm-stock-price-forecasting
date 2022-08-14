import StockPredictor
import sys

'''
Goal:
- Train several LSTM models with varying training target change parameters. 
- Save the models to disk.
- Test the models on the test set.
- Save the test accuracy to disk.

'''
device = 'cpu'
sp = StockPredictor.StockPredictor()


request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
           'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
           'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']

'''
Best params from model eval:
timestep: 540
num_layers: 4
hidden_size: 1000
dropout: 0.2
learning_rate: 0.0001
num_epochs: 20
'''

# we will use abbreviated params and train data to save time
# timestep = 200
# num_layers = 2
# hidden_size = 100
# dropout = 0.2
# learning_rate = 0.001
# num_epochs = 5


_, target_price_change, timestep, num_layers, hidden_size, dropout, learning_rate, num_epochs = sys.argv

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
trained_on: allstocks
''')

verbose = False

sp.tb.add_text('Model Params: ', f'''
timestep: {timestep}
num_layers: {num_layers}
hidden_size: {hidden_size}
dropout: {dropout}
learning_rate: {learning_rate}
num_epochs: {num_epochs}
target_price_change: {target_price_change}
trained_on: allstocks
''')

# request = sp.sl.getlocaltickers()

sp.load(request, timestep=timestep, verbose=verbose, target_price_change=target_price_change)

sp.create_model(device=device, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout, learning_rate=learning_rate)

sp.train_model(num_epochs=num_epochs, verbose=verbose)

sp.save_model(f'/models/ensemble/pc_{target_price_change}_ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}.pth')

test_accuracy = sp.test_model(request, verbose=verbose, target_price_change=target_price_change)

sp.tb.add_text('Test Accuracy: ', f'{test_accuracy}')