import requests
import env
import dataconfig
df = dataconfig.DataConfig()

api_key = env.get_api_key()

symbol = 'IBM'

url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}'
r = requests.get(url)
data = r.json()

print(f'request : {data}')