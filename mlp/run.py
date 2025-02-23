import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os

from model import MLP

import logging

def get_data_loader(index, feature, label, batch_size):
    data = TensorDataset(feature[index], label[index])
    data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data_loader

def calculate_f1_precision_recall(pred, target):
    pred = pred.to('cpu')
    target = target.to('cpu')
    md = 'weighted'
    print("target bot radio", torch.sum(target)/len(target))
    print("pred bot radio", torch.sum(pred)/len(pred))
    print("bot precision", precision_score(target, pred, labels=[1], average='binary'))
    print("bot recall", recall_score(target, pred, labels=[1], average='binary'))
    precision = precision_score(target, pred, average=md)
    recall = recall_score(target, pred, average=md)
    f1 = f1_score(target, pred, average=md)
    return f1, precision, recall

def train_one_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    loss_list = []
    pred_list = []
    label_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        pred = output.argmax(dim=1, keepdim=True)
        label_list.extend(target.cpu().tolist())
        pred_list.extend(pred.cpu().tolist())
    loss = sum(loss_list) / len(loss_list)
    pred_list = torch.tensor(pred_list)
    label_list = torch.tensor(label_list)
    f1, precision, recall = calculate_f1_precision_recall(pred_list, label_list)
    correct = sum([1 if pred_list[i] == label_list[i] else 0 for i in range(len(pred_list))])
    acc = correct / len(label_list)
    
    return loss, acc, f1, precision, recall

def test(model, device, test_loader):
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            label_list.extend(target.cpu().tolist())
            pred_list.extend(pred.cpu().tolist())
    pred_list = torch.tensor(pred_list)
    label_list = torch.tensor(label_list)
    f1, precision, recall = calculate_f1_precision_recall(pred_list, label_list)
    correct = sum([1 if pred_list[i] == label_list[i] else 0 for i in range(len(pred_list))])
    acc = correct / len(label_list)
    return acc, f1, precision, recall

