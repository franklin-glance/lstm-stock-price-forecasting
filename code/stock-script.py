import StockPredictor
import sys

print('Number of args: ', len(sys.argv))
print('args: ', str(sys.argv))

request = ['AAPL', 'GS', 'IBM', 'MSFT', 'AMGN', 'MMM', 'COST', 'CVX', 'FDX', 'CMI', 'BLK', 'AVB', 'HD', 'LMT',
           'JNJ', 'PFE', 'PG', 'PEP', 'PKI', 'PYPL', 'QCOM', 'RCL', 'ROKU', 'SBUX', 'T', 'TSLA', 'TWTR', 'TXN',
           'UNH', 'VZ', 'V', 'WMT', 'XOM', 'WBA', 'WFC', 'WYNN']

'''
format: 
stock-script [timestep] [num_layers] [hidden_size] [dropout] [learning_rate] [num_epochs]
'''

device = 'cpu'
sp = StockPredictor.StockPredictor()

allstocks = sp.sl.getlocaltickers()

_, timestep, num_layers, hidden_size, dropout, learning_rate, num_epochs = sys.argv

timestep = int(timestep)
num_layers = int(num_layers)
hidden_size = int(hidden_size)
dropout = float(dropout)
learning_rate = float(learning_rate)
num_epochs = int(num_epochs)


sp.tb.add_text('Model Params: ', f'''
timestep: {timestep}
num_layers: {num_layers}
hidden_size: {hidden_size}
dropout: {dropout}
learning_rate: {learning_rate}
num_epochs: {num_epochs}
trained_on: allstocks
''')

sp.load(allstocks, timestep=timestep, verbose=True)

sp.create_model(device=device, num_layers=num_layers, hidden_size=hidden_size, dropout=dropout, learning_rate=learning_rate)

sp.train_model(num_epochs=num_epochs, verbose=True)

sp.save_model(f'/models/ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}.pth')

test_accuracy = sp.test_model(allstocks, verbose=True)

sp.tb.add_text('Test Accuracy: ', f'{test_accuracy}')
