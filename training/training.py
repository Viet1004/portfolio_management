import torch
import numpy as np
import sys
sys.path.append('../')
from utils.utils import estimate_reward
from tqdm import tqdm

n = 50
m = 8

HORIZON = 50
MAX_TRAJECTORIES = 500
N_EPOCHS = 100
gamma = 0.99
score = []
N_b = 10


# Write a function to train a model
def train(model, train_data, valid_data, optimizer, horizon=HORIZON, N_b = N_b, n_epochs = N_EPOCHS, verbose = True):
    train_losses = []
    valid_losses = []
    init_weight = torch.from_numpy(np.array([1] + [0 for i in range(m)]))
    for epoch in tqdm(range(n_epochs)):
        latest_date = len(train_data)-n_epochs+epoch
        model.train()
        chosen_indexes = np.random.geometric(p=0.1, size=N_b)
        curr_train_loss = 0
        for index in chosen_indexes:

            init_date = latest_date - index - horizon
            strategy = init_weight
            transitions = []
            train_loss = 0 # train_loss is -J_value
            for t in range(horizon):
                previsous_strategy = strategy.float()
                curr_state = (train_data[init_date+t].float(), strategy[1:].float())    # Notice that the strategy only takes the value from 1: forward because we don't count the first elment of BTC
                strategy = model(curr_state).squeeze(-1).float()
                relative_price = torch.concatenate((torch.from_numpy(np.array([1.])), 1/train_data[init_date+t][0][:,-2].squeeze(-1))).float()
                reward = estimate_reward(previsous_strategy, strategy, relative_price)
                train_loss += reward
                transitions.append((curr_state, strategy, reward))
                
            train_loss = -1/horizon * train_loss  # this is the average reward over a mini-batch 

            curr_train_loss += train_loss            

        curr_train_loss = curr_train_loss/N_b
        optimizer.zero_grad()
        curr_train_loss.backward() 
        optimizer.step() 
        train_losses.append(curr_train_loss.cpu().data.numpy())

        model.eval()
        valid_loss = 0
        for t in range(len(valid_data)):
            previsous_strategy = strategy.float()
            curr_state = (valid_data[t], strategy[1:].float())
            strategy = model(curr_state).squeeze(-1).float()
            relative_price = torch.concatenate((torch.from_numpy(np.array([1.])), 1/valid_data[t][0][:,-2].squeeze(-1))).float()
            reward = estimate_reward(previsous_strategy, strategy, relative_price)
            valid_loss += reward
        valid_loss = - 1/len(valid_data) * valid_loss
        valid_losses.append(valid_loss.cpu().data.numpy())

        if verbose:
            print("Epoch %d, train loss: %f, valid loss: %f" % (epoch, curr_train_loss, valid_loss))
