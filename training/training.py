import torch
import numpy as np
import sys
sys.path.append('../')
from utils.utils import estimate_reward
from models.models import PVM
from tqdm import tqdm
import math
import copy

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
#    init_weight = [1] + [0 for i in range(m)]
    init_weight = [1/(m+1) for i in range(m+1)]   
    patient = 5 
    for epoch in tqdm(range(n_epochs)):
#        latest_date = len(train_data)-n_epochs+epoch
        model.train()
        curr_train_loss = 0
        memory = PVM(0.01, horizon, len(train_data) - n_epochs + epoch, init_weight)
        assert memory.check_zero(), "Incorrect initialization of memory"
        memoryFlashback = copy.deepcopy(memory.get_memory())
        original_memory = copy.deepcopy(memory.get_memory())
        for i in range(N_b):
            init_date = memory.draw()
            strategy = torch.from_numpy(memory.get_weight(init_date)).float()
            try:
                assert np.abs(np.sum(strategy.cpu().data.numpy())-1) <= 0.001, f"Strategy is not at {init_date}"
            except AssertionError:
                print("Error at init_date: ", init_date)
                print(original_memory[init_date])
                print(strategy.cpu().data.numpy())
                print(memoryFlashback[init_date])
                # throw a ValueError
#                raise ValueError("Strategy is all zeros")

#            print("=====================================")
#            print(f"strategy: {strategy.cpu().data.numpy()} and init_date: {init_date}")
#            print("=====================================")
            transitions = []
            memoryFlashback = copy.deepcopy(memory.get_memory())
            train_loss = 0 # train_loss is -J_value
            for t in range(horizon):
                previous_strategy = strategy.float()
                curr_state = (train_data[init_date+t].float(), strategy[1:].float())    # Notice that the strategy only takes the value from 1: forward because we don't count the first elment of BTC
                strategy = model(curr_state).squeeze(-1).float()    # strategy is a vector of size m+1
                assert np.abs(np.sum(strategy.cpu().data.numpy())-1) <= 0.001, f"Strategy is not normalized, it is {strategy.cpu().data.numpy()}" 
                memory.update(strategy.cpu().data.numpy(), init_date+t+1)    # update the memory. Be careful on this one with +1, memory should update at time init_date+t+1
                relative_price = torch.concatenate((torch.from_numpy(np.array([1.])), 1/train_data[init_date+t+1][0][:,-2].squeeze(-1))).float()
                reward = estimate_reward(previous_strategy, strategy, relative_price)
                
#                if math.isnan(reward.cpu().data.numpy()):
#                    print("=====================================")
#                    print(i)
#                    print(t)
#                    print("relative price: ", relative_price.cpu().data.numpy())
#                    print("strategy: ", strategy.cpu().data.numpy())
#                    print("previous_strategy: ", previous_strategy.cpu().data.numpy())
#                    print("reward: ", reward.cpu().data.numpy())
                train_loss += reward
                transitions.append((curr_state, strategy, reward))
                
            train_loss = -1/horizon * train_loss  # this is the average reward over a mini-batch 
            optimizer.zero_grad()
            train_loss.backward() 
            optimizer.step() 
            curr_train_loss += train_loss

        curr_train_loss = curr_train_loss/N_b

        train_losses.append(curr_train_loss.cpu().data.numpy())

        model.eval()
        valid_loss = 0
        for t in range(len(valid_data)):
            previous_strategy = strategy.float()
            curr_state = (valid_data[t], strategy[1:].float())
            strategy = model(curr_state).squeeze(-1).float()
            relative_price = torch.concatenate((torch.from_numpy(np.array([1.])), 1/valid_data[t][0][:,-2].squeeze(-1))).float()
            reward = estimate_reward(previous_strategy, strategy, relative_price)
            valid_loss += reward
        valid_loss = - 1/len(valid_data) * valid_loss

        valid_losses.append(valid_loss.cpu().data.numpy())
        
        if verbose:
            print("Epoch %d, train loss: %f, valid loss: %f" % (epoch, curr_train_loss, valid_loss))
        
        # early stopping
        if epoch > 3:
            if valid_losses[-1] > valid_losses[-2]:
                patient -= 1
            else:
                patient = 5
            if patient == 0:
                break    

    return train_losses, valid_losses
