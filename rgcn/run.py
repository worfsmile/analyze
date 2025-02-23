import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import TransformerConv, GATConv, GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import warnings
from sklearn.model_selection import train_test_split

from model import BotRGCN

def create_loaders(graph, input_nodes, batch_size, num_neighbors, shuffle=False):
    loader = NeighborLoader(
        graph,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(42),
        batch_size=batch_size,
        num_neighbors=num_neighbors,
        input_nodes=input_nodes
    )
    return loader

def calculate_f1_precision_recall(pred, target):
    pred = pred.to('cpu')
    target = target.to('cpu')
    md = 'weighted'
    # print("target bot radio", torch.sum(target)/len(target))
    # print("pred bot radio", torch.sum(pred)/len(pred))
    # print("bot precision", precision_score(target, pred, labels=[1], average='binary'))
    # print("bot recall", recall_score(target, pred, labels=[1], average='binary'))
    precision = precision_score(target, pred, average=md)
    recall = recall_score(target, pred, average=md)
    f1 = f1_score(target, pred, average=md)
    return f1, precision, recall

def train_one_epoch(model, device, loader, optimizer, criterion):
    model.train()
    loss_list = []
    pred_list = []
    label_list = []
    for batch in loader:
        optimizer.zero_grad()
        output = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
        label = batch.y
        input_nodes_id = batch.input_id
        label = label[:input_nodes_id.shape[0]]
        output = output[:input_nodes_id.shape[0]]
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        label_list.extend(label.cpu().tolist())
        pred_list.extend(pred.cpu().tolist())
    loss = sum(loss_list) / len(loss_list)
    pred_list = torch.tensor(pred_list)
    label_list = torch.tensor(label_list)
    f1, precision, recall = calculate_f1_precision_recall(pred_list, label_list)
    correct = sum([1 if pred_list[i] == label_list[i] else 0 for i in range(len(pred_list))])
    acc = correct / len(label_list)
    
    return loss, acc, f1, precision, recall

def test(model, device, loader):
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            output = model(batch.x.to(device), batch.edge_index.to(device), batch.edge_attr.to(device))
            label = batch.y
            input_nodes_id = batch.input_id
            label = label[:input_nodes_id.shape[0]]
            output = output[:input_nodes_id.shape[0]]
            pred = output.argmax(dim=1, keepdim=True)
            label_list.extend(label.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())
    pred_list = torch.tensor(pred_list)
    label_list = torch.tensor(label_list)
    f1, precision, recall = calculate_f1_precision_recall(pred_list, label_list)
    correct = sum([1 if pred_list[i] == label_list[i] else 0 for i in range(len(pred_list))])
    acc = correct / len(label_list)
    return acc, f1, precision, recall


