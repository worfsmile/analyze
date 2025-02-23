import torch
from torch import nn
from torch_geometric.nn import RGCNConv,FastRGCNConv,GCNConv,GATConv
import torch.nn.functional as F

class BotRGCN(nn.Module):
    def __init__(self, tweet_size=768, embedding_dimension=128, dropout=0.3):
        super(BotRGCN, self).__init__()
        self.dropout = dropout
        
        self.linear_relu_input=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        
        self.linear_relu_tweet=nn.Sequential(
            nn.Linear(tweet_size,embedding_dimension),
            nn.LeakyReLU()
        )
        self.rgcn=RGCNConv(embedding_dimension,embedding_dimension,num_relations=2)
        
        self.linear_relu_output=nn.Sequential(
            nn.Linear(embedding_dimension,embedding_dimension),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(embedding_dimension,2)
        
    def forward(self, tweet, edge_index, edge_type):
        x=self.linear_relu_tweet(tweet)
        
        x=self.linear_relu_input(x)
        x=self.rgcn(x,edge_index,edge_type)
        x=F.dropout(x,p=self.dropout,training=self.training)
        x=self.rgcn(x,edge_index,edge_type)
        x=self.linear_relu_output(x)
        x=self.linear_output(x)
        return x
    
