import numpy as np
import torch
import torch.nn as nn
np.random.seed(0)
torch.manual_seed(0)

# create a pytorch model of a convolutional layer of in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True
class CNNPolicy(nn.Module):
    def __init__(self):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(1,3), stride=1, bias=True)
        nn.init.xavier_uniform_(self.conv1.weight)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,48), stride=1, bias=True)
        nn.init.xavier_uniform_(self.conv2.weight)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1,1), stride=1, bias=True)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x , lastAction = state
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x).squeeze(2)
        
#        x = self.relu2(x)                             # squeeze the second dimension
        lastAction = lastAction.view(len(lastAction), 1)                        # reshape lastAction to (.,1)
        x = torch.concatenate((lastAction,x.T), dim=1).T.unsqueeze(2)                # concatenate lastAction to x
        
        x = self.conv3(x)
        x = torch.concatenate((torch.tensor([[[0.]]]),x), dim=1) # add a zero to the first element of x as a cash bias
        x = self.softmax(x).squeeze(0)
        return x
    
class RNNPolicy(nn.Module):
    def __init__(self):
        super(RNNPolicy, self).__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=20, batch_first=True)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        nn.init.xavier_uniform_(self.fc2.weight)
#        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1,1), stride=1, bias=True)
        nn.init.xavier_uniform_(self.conv3.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x , lastAction = state
        # permute dimension of x from (batch_size, seq_len, input_size) to (seq_len, input_size, batch_size)
        x = x.permute(1,2,0)
        x = self.lstm(x)
        x = x[0][:,-1,:] # get the last output of the lstm
        # squeeze the second dimension
        x = x.squeeze(1)
        x = self.relu1(x)
        x = self.fc2(x)
#        x = self.relu2(x)
        lastAction = lastAction.view(len(lastAction), 1)                        # reshape lastAction to (1,.)
        x = torch.concatenate((lastAction,x), dim=1).T.unsqueeze(2)                # concatenate lastAction to x
        x = self.conv3(x)
        x = torch.concatenate((torch.tensor([[[1.]]]),x), dim=1) # add a zero to the first element of x as a cash bias
        x = self.softmax(x).squeeze(0)
        return x
    

class PVM():
    def __init__(self, beta, batch_size, total_steps, w_init):
        self.beta = beta
        self.w = w_init
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.w_history = np.array([w_init] * total_steps) #change back to torch

    def check_zero(self):
        if (self.w_history == np.array(self.w)).all(): #change back to torch
            return True
        else:
            return False
        
    def update(self, w, t):
        assert np.sum(np.abs(w) - 1) <= 0.001, "Strategy is not normalized"  #change back to torch
        self.w_history[t] = w

    def get_weight(self, t):
        return self.w_history[t]

    def draw(self):
        while True:
            z = np.random.geometric(p=self.beta)
            tb = self.total_steps - z - self.batch_size - 1
            if tb >= 0:
                return tb

    def get_memory(self):
        return self.w_history
    

