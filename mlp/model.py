
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(MLP, self).__init__()
        self.embed_layer = nn.Linear(input_size, hidden_size)
        self.relu = nn.LeakyReLU()
        self.squencelayer = nn.Sequential()
        for i in range(num_layers):
            self.squencelayer.add_module('relu'+str(i+1), nn.LeakyReLU())
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.embed_layer(x)
        x = self.relu(x)
        x = self.squencelayer(x)
        x = self.classifier(x)
        return x
