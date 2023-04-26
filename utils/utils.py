import torch as torch
import numpy as np


n = 50
m = 8

HORIZON = 50
MAX_TRAJECTORIES = 500
N_EPOCHS = 100
gamma = 0.99
score = []
N_b = 10


commission_rate = 0.0025


def createTensor(data, time):
    priceTensor = torch.zeros(3,m,n)
    reference_price = [data[currency]["close"].iloc[time] for currency in data]
    reference_price = np.reshape(reference_price, (len(reference_price),1))
    vectorprice_close = np.array([[data[currency]["close"].iloc[time-n+i+1] for i in range(n)] for currency in data])
    vectorprice_high = np.array([[data[currency]["high"].iloc[time-n+i+1] for i in range(n)] for currency in data])
    vectorprice_low = np.array([[data[currency]["low"].iloc[time-n+i+1] for i in range(n)] for currency in data])
    priceTensor[0] = torch.from_numpy(vectorprice_close/reference_price)
    priceTensor[1] = torch.from_numpy(vectorprice_high/reference_price)
    priceTensor[2] = torch.from_numpy(vectorprice_low/reference_price)    
    return priceTensor


def train_valid_split(data, valid_ratio = 0.95):
    train_data = []
    valid_data = []
    
    data_length = len(data[list(data.keys())[0]])
    print([len(data[key]) for key in list(data.keys())])
    splitnumber_valid = int(data_length*valid_ratio)

    for i in np.arange(n,splitnumber_valid):
        train_data.append(createTensor(data, time=i))
    for i in np.arange(splitnumber_valid,data_length):
        valid_data.append(createTensor(data, time=i))

    return train_data, valid_data

def test_data(data):
    test_data = []
    data_length = len(data[list(data.keys())[0]])
    for i in np.arange(n,data_length):
        test_data.append(createTensor(data, time=i))
    return test_data

# the next three functions are for calculating the transaction cost
def auxilary_transaction(current_strategy_prime, current_strategy, mu):
    #finding the transaction cost
    return 1/(1 - commission_rate*current_strategy[0]) * (1 - commission_rate*current_strategy_prime[0] - (2 * commission_rate - commission_rate**2) * torch.sum(torch.clamp(current_strategy_prime - mu * current_strategy, min = 0)))


def transaction_cost(current_strategy_prime, current_strategy, tol = 0.001):
    # finding the approximate transaction cost
    mu = commission_rate * torch.sum(torch.abs(current_strategy_prime - current_strategy))
    diff = tol + 1
    #finding the transaction cost
    while(diff > tol):
        update_mu = auxilary_transaction(current_strategy_prime, current_strategy, mu)
        diff = torch.abs(mu - update_mu)
        mu = update_mu
    return mu

# test transaction_cost
#current_strategy_prime = torch.from_numpy(np.array([1/3, 1/3, 1/3])).float()
#current_strategy = torch.from_numpy(np.array([1/3, 1/3, 1/3])).float()
#print(transaction_cost(current_strategy_prime, current_strategy))


def estimate_reward(previous_strategy, current_strategy, relative_price, tol = 0.001):
    current_strategy_prime = previous_strategy * relative_price / previous_strategy.T.dot(relative_price)
    
    # finding the approximate transaction cost
    mu = transaction_cost(current_strategy_prime, current_strategy, tol)
    
    # finding the reward
    reward = torch.log(mu*previous_strategy.T.dot(relative_price))

    return reward

# implement a function to calculate the maximum drawdown
def max_drawdown(data):
    max_drawdown = 0
    max_value = -np.inf
    for i in range(len(data)):
        if data[i] > max_value:
            max_value = data[i]
        else:
            drawdown = (max_value - data[i])/max_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    return max_drawdown

# implement a function to calculate the sharpe ratio
def sharpe_ratio(data):
    return np.mean(data)/np.std(data)

def fAPV(data):
    return np.sum(data)


        




