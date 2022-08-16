import StockPredictor
import sys

device = 'cpu'
lookahead = 120 # lookahead is the number of days we will predict ahead
sp = StockPredictor.StockPredictor()
'''
---- 120 lookahead params ----
timestep: 240 
num_layers: 2
hidden_size: 500 
dropout: 0.1
learning_rate: 0.0001
num_epochs: 10
lookahead: 120
'''

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
lookahead: {lookahead}
trained on: allstocks
''')

verbose = True

sp.tb.add_text('Test Model Params: ', f'''
timestep: {timestep}
num_layers: {num_layers}
hidden_size: {hidden_size}
dropout: {dropout}
learning_rate: {learning_rate}
num_epochs: {num_epochs}
target_price_change: {target_price_change}
lookahead: {lookahead}
trained on: allstocks
testing on: allstocks
''')

request = sp.sl.getlocaltickers()

sp.load_model(f'/models/ensemble/pc_{target_price_change}_ts-{timestep}_l-{num_layers}_hs-{hidden_size}_d-{dropout}_lr-{learning_rate}_e-{num_epochs}_look-{lookahead}.pth',
              num_layers=num_layers, hidden_size=hidden_size, dropout=dropout)

test_accuracy = sp.test_model(request, verbose=verbose, target_price_change=target_price_change, lookahead=lookahead, allstocks=True)

sp.tb.add_text('Test Accuracy: ', f'{test_accuracy}')
