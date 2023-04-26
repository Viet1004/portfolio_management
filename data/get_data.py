import requests
import json
import numpy as np


limit = 500

currencies = ["ETH", "TRX", "LTC", "XRP", "DASH", "AVA", "DOGE", "ATOM"]
#currencies = ["XRP", "AVA"]


for currency in currencies:
    train_data = list()
    end_time = 1590995200000
    print(currency)
    for i in range(80):
        if len(train_data) != i:            
            print(i)
            break
        url = f'https://api.poloniex.com/markets/{currency}_USDT/candles?interval=MINUTE_30&limit={limit}&endTime={end_time}'
        response = requests.get(url)
        if response.status_code == 200:
    #       data.append(response.json())
            content = json.loads(response.content)
            train_data.append(content) 
        
        end_time += 1800000*limit   # 1800000 is the number of milliseconds in 30 minutes

    train_data = [x for sublist in train_data for x in sublist]

    with open(f"USDT/{currency}_USDT_train.json", "w") as f:
        json.dump(train_data, f)

    test_data = []
    for i in np.arange(80,100):
        if len(test_data) != i-80:            
            print(i)
            break
        url = f'https://api.poloniex.com/markets/{currency}_USDT/candles?interval=MINUTE_30&limit={limit}&endTime={end_time}'
        response = requests.get(url)
        if response.status_code == 200:
    #       data.append(response.json())
            content = json.loads(response.content)
            test_data = test_data + [content]
        end_time += 1800000*limit   # 1800000 is the number of milliseconds in 30 minutes

    test_data = [x for sublist in test_data for x in sublist]

    with open(f"USDT/{currency}_USDT_test.json", "w") as f:
        json.dump(test_data, f)
