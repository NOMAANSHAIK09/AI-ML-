import torch
import torch.nn as nn

#creating a neural network for the DQN agent
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        self.network = nn.Sequential(
            nn.Linear(4,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        
    def forward(self, x):
        return self.network(x)