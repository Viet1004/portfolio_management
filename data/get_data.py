import requests
import json
import numpy as np


limit = 100

currencies = ["ETH", "TRX", "LTC", "SOL", "ADA", "AVA", "DOGE", "ATOM"]



for currency in currencies:
    train_data = []
    start_time = 0
    for i in range(400):
        url = f'https://api.poloniex.com/markets/{currency}_BTC/candles?interval=MINUTE_30&limit={limit}&starttime={start_time}'
        response = requests.get(url)
        if response.status_code == 200:
    #       data.append(response.json())
            content = json.loads(response.content)
            train_data.append(content)
        start_time += 1800000*limit   # 1800000 is the number of milliseconds in 30 minutes

    train_data = [x for sublist in train_data for x in sublist]

    with open(f"{currency}_BTC_train.json", "w") as f:
        json.dump(train_data, f)

    test_data = []
    for i in np.arange(400,450):
        url = f'https://api.poloniex.com/markets/{currency}_BTC/candles?interval=MINUTE_30&limit={limit}&starttime={start_time}'
        response = requests.get(url)
        if response.status_code == 200:
    #       data.append(response.json())
            content = json.loads(response.content)
            test_data.append(content)
        start_time += 1800000*limit   # 1800000 is the number of milliseconds in 30 minutes

    test_data = [x for sublist in test_data for x in sublist]

    with open(f"{currency}_BTC_test.json", "w") as f:
        json.dump(test_data, f)
