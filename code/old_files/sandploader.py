import pandas as pd
import time
import dataconfig

dc = dataconfig.DataConfig()
sandp = pd.read_csv('../data/s-and-p-500companies.csv')
sandp = list(sandp['Symbol'])

request = []
for i in range(1, len(sandp)):
    request.append(sandp[i])
    print(f'appending {sandp[i]}')
    if i % 20 == 0:
        print(f'requesting: {request}')
        data = dc.getdata(request, verbose=True)
        request = []
        time.sleep(30)
        print('resuming')
