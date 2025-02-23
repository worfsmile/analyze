import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader
import warnings
import os
import numpy as np
import random

from model import MLP

import logging

from run import get_data_loader, train_one_epoch, test, calculate_f1_precision_recall
from model import MLP

warnings.filterwarnings("ignore", category=FutureWarning)

setting = {
    "trial_setting":['mlp', 'ood_data'],
    "seeds":[42],
    "save_model": False,
    "lr": 1e-4,
    "epochs": 100,
    "batch_size": 128,
    "input_size": 768,
    "output_size": 2,
    "hidden_size": 128,
    "num_layers": 2,
    "early_stop": 50
}
    # lr = 1e-4
    # epochs = 100
    # batch_size = 128
    # input_size = 768
    # output_size = 2
    # hidden_size = 128
    # num_layers = 2
    # early_stop = 50


logging_file = f"log/{setting['trial_setting'][0]}/{setting['trial_setting'][1]}/training.log"

if not os.path.exists(os.path.dirname(logging_file)):
    os.makedirs(os.path.dirname(logging_file))

logging.basicConfig(
    filename=logging_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def load_data():
    data_name = 'raw_data/processed_data'
    base_path = './data'
    feature_path = os.path.join(base_path, data_name, 'tweets_tensor.pt')
    labels_path = os.path.join(base_path, data_name, 'label.pt')
    
    if setting['trial_setting'][1] == 'raw_data':
        train_idx_path = os.path.join(base_path, data_name, 'train_idx.pt')
        valid_idx_path = os.path.join(base_path, data_name, 'val_idx.pt')
        test_idx_path = os.path.join(base_path, data_name, 'test_idx.pt')
    
    elif setting['trial_setting'][1] == 'normal_data':
        train_idx_path = r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\normal\train.pt"
        valid_idx_path = r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\normal\val.pt"
        test_idx_path = r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\normal\test.pt"
    
    elif setting['trial_setting'][1] == 'ood_data':
        #ood
        train_idx_path = r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\ood\train.pt"
        valid_idx_path = r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\ood\val.pt"
        test_idx_path = r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\ood\test.pt"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    features = torch.load(feature_path).to(device)
    labels = torch.load(labels_path).to(device)
    train_index = torch.load(train_idx_path).to(device)
    valid_index = torch.load(valid_idx_path).to(device)
    test_index = torch.load(test_idx_path).to(device)
    
    return train_index, test_index, valid_index, features, labels

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = setting['lr']
    epochs = setting['epochs']
    batch_size = setting['batch_size']
    input_size = setting['input_size']
    output_size = setting['output_size']
    hidden_size = setting['hidden_size']
    num_layers = setting['num_layers']
    early_stop = setting['early_stop']
    
    model = MLP(input_size, output_size, hidden_size, num_layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_index, test_index, valid_index, feature, label = load_data()
    
    train_loader = get_data_loader(train_index, feature, label, batch_size)
    valid_loader = get_data_loader(valid_index, feature, label, batch_size)
    test_loader = get_data_loader(test_index, feature, label, batch_size)
    
    criterion = nn.CrossEntropyLoss()
    best_valid = {'acc': 0, 'f1': 0, 'precision': 0,'recall': 0}
    best_test = {'acc': 0, 'f1': 0, 'precision': 0,'recall': 0}
    early_stop_count = 0
    best_test_record = {'acc': [], 'f1': [], 'precision': [],'recall': []}
    seeds = setting['seeds']
    for i in range(len(seeds)):
        set_seed(seeds[i])
        save_model_path = f"./model/{setting['trial_setting'][0]}/{setting['trial_setting'][1]}/mlp_{seeds[i]}.pt"
        if not os.path.exists(os.path.dirname(save_model_path)):
            os.makedirs(os.path.dirname(save_model_path))
            
        logging.info(f'Trial {i+1} - Seed: {seeds[i]}')
        for epoch in range(1, epochs + 1):
            
            train_loss, train_acc, train_f1, train_precision, train_recall = train_one_epoch(model, device, train_loader, optimizer, criterion)
            valid_acc, valid_f1, valid_precision, valid_recall = test(model, device, valid_loader)
            test_acc, test_f1, test_precision, test_recall = test(model, device, test_loader)
            
            if valid_acc > best_valid['acc']:
                early_stop_count = 0
                best_valid['acc'] = valid_acc
                best_valid['f1'] = valid_f1
                best_valid['precision'] = valid_precision
                best_valid['recall'] = valid_recall
                best_test['acc'] = test_acc
                best_test['f1'] = test_f1
                best_test['precision'] = test_precision
                best_test['recall'] = test_recall
                if setting['save_model']:
                    torch.save(model.state_dict(), save_model_path)
            else:
                early_stop_count += 1
                if early_stop_count > early_stop:
                    print(f'Early stop at epoch {epoch}')
                    break
            print(f'Epoch: {epoch}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f},')
            print(f'Valid Acc: {valid_acc:.4f}\{best_valid["acc"]:.4f}, Valid F1: {valid_f1:.4f}, Valid Precision: {valid_precision:.4f}, Valid Recall: {valid_recall:.4f},')
            print(f'Test Acc: {test_acc:.4f}\{best_test["acc"]:.4f}, Test F1: {test_f1:.4f}, Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f},')
            logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
                        f"Train Precision: {train_precision:.4f}, Train Recall: {train_recall:.4f}")
            logging.info(f"Epoch {epoch} - Valid Acc: {valid_acc:.4f}/{best_valid['acc']:.4f}, Valid F1: {valid_f1:.4f}, "
                        f"Valid Precision: {valid_precision:.4f}, Valid Recall: {valid_recall:.4f}")
            logging.info(f"Epoch {epoch} - Test Acc: {test_acc:.4f}/{best_test['acc']:.4f}, Test F1: {test_f1:.4f}, "
                        f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}")
        
        logging.info(f'Best Valid Acc: {best_valid["acc"]:.4f}, Best Valid F1: {best_valid["f1"]:.4f}, '
                    f'Best Valid Precision: {best_valid["precision"]:.4f}, Best Valid Recall: {best_valid["recall"]:.4f}')
        logging.info(f'Best Test Acc: {best_test["acc"]:.4f}, Best Test F1: {best_test["f1"]:.4f}, '
                    f'Best Test Precision: {best_test["precision"]:.4f}, Best Test Recall: {best_test["recall"]:.4f}')
        print(f'Best Valid Acc: {best_valid["acc"]:.4f}, Best Valid F1: {best_valid["f1"]:.4f}, Best Valid Precision: {best_valid["precision"]:.4f}, Best Valid Recall: {best_valid["recall"]:.4f},')
        print(f'Best Test Acc: {best_test["acc"]:.4f}, Best Test F1: {best_test["f1"]:.4f}, Best Test Precision: {best_test["precision"]:.4f}, Best Test Recall: {best_test["recall"]:.4f},')  
        best_test_record['acc'].append(best_test['acc'])
        best_test_record['f1'].append(best_test['f1'])
        best_test_record['precision'].append(best_test['precision'])
        best_test_record['recall'].append(best_test['recall'])
    print(f'Best Test Acc: {sum(best_test_record["acc"])/len(best_test_record["acc"]):.4f}, Best Test F1: {sum(best_test_record["f1"])/len(best_test_record["f1"]):.4f}, '
          f'Best Test Precision: {sum(best_test_record["precision"])/len(best_test_record["precision"]):.4f}, Best Test Recall: {sum(best_test_record["recall"])/len(best_test_record["recall"]):.4f},')
    logging.info(f'Best Test Acc: {sum(best_test_record["acc"])/len(best_test_record["acc"]):.4f}, Best Test F1: {sum(best_test_record["f1"])/len(best_test_record["f1"]):.4f}, '
          f'Best Test Precision: {sum(best_test_record["precision"])/len(best_test_record["precision"]):.4f}, Best Test Recall: {sum(best_test_record["recall"])/len(best_test_record["recall"]):.4f},')

if __name__ == '__main__':
    main()