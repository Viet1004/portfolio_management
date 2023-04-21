import numpy as np
import torch
import torch.nn as nn


# create a pytorch model of a convolutional layer of in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, bias=True
class CNNPolicy(nn.Module):
    def __init__(self):
        super(CNNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(1,3), stride=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(1,48), stride=1, bias=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1,1), stride=1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, state):
        x , lastAction = state
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        
        x = self.relu2(x).squeeze(2)                             # squeeze the second dimension
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
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=20, out_features=20)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=21, out_channels=1, kernel_size=(1,1), stride=1, bias=True)
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
        x = self.relu2(x)
        lastAction = lastAction.view(len(lastAction), 1)                        # reshape lastAction to (1,.)
        x = torch.concatenate((lastAction,x), dim=1).T.unsqueeze(2)                # concatenate lastAction to x
        x = self.conv3(x)
        x = torch.concatenate((torch.tensor([[[0.]]]),x), dim=1) # add a zero to the first element of x as a cash bias
        x = self.softmax(x).squeeze(0)
        return x