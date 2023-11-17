import torch
import torch.nn as nn
import numpy as np


class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.ipL = nn.Linear(8,128)   # has pre trained weight
        self.HL1 = nn.Linear(128,128) # has pre trained weight
        self.HL2 = nn.Linear(128,128) # has pre trained weight
        self.opL = nn.Linear(128,5)   # output layer
        
    def forward(self, x):
        x = torch.sigmoid(self.ipL(x))   # sigmiodal,relu,.etc.,
        x = torch.sigmoid(self.HL1(x))
        x = torch.sigmoid(self.HL2(x))
        x = self.opL(x)
        return x

##################################
########## To Check ##############
##################################
# net = FFNN()
# print(net)
# ip = torch.tensor([1.149,0,0,0,0,0,0,0])
# op = net(ip)
# print(op)
# print( net(ip).max())
# print(np.argmax(op.detach().numpy()))

###################################
########## End ####################
###################################
