import torch

#1 2 3 4

data_dir = ""

test_idx = torch.load(f"./data/{data_dir}\normal\test.pt")
train_idx = torch.load(f"./data/{data_dir}\normal\train.pt")
val_idx = torch.load(f"./data/{data_dir}\normal\val.pt")
label = torch.load(r"D:\bot_analyze_shortcut\data\bot_classifier_data_analyze\twi-22_label\label.pt")
sentiment = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\tweets20_sentiment.pt")


test_bot = test_idx[label[test_idx] == 1]
train_bot = train_idx[label[train_idx] == 1]
test_human = test_idx[label[test_idx] == 0]
train_human = train_idx[label[train_idx] == 0]

test_sentiment = sentiment[test_idx]
train_sentiment = sentiment[train_idx]

test_bot_sentiment = set(test_sentiment[label[test_idx] == 1].tolist())
train_bot_sentiment = set(train_sentiment[label[train_idx] == 1].tolist())
test_human_sentiment = set(test_sentiment[label[test_idx] == 0].tolist())
train_human_sentiment = set(train_sentiment[label[train_idx] == 0].tolist())

print("test_bot_sentiment:", test_bot_sentiment)
print("train_bot_sentiment:", train_bot_sentiment)
print("test_human_sentiment:", test_human_sentiment)
print("train_human_sentiment:", train_human_sentiment)


