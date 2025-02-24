import torch

"""     |      bot     |        human        |
---------------------------------------------------
positive|  tr te va 1  |    tr te va 2       |
---------------------------------------------------
negative|  tr te va 3  |    tr te va 4       |
---------------------------------------------------

ood_dataset = [
    tr : [2, 3]
    va : [1, 2, 3, 4]
    te : [1, 4]

normal_dataset = [
    tr : [1, 2, 3, 4]
    va : [1, 2, 3, 4]
    te : [1, 4]
"""

def make_normal_dataset(train_list, test_list, val_list):
    train_index = []
    test_index = []
    val_index = []
    for i in range(len(train_list)):
        train_index.extend(train_list[i])
    for i in range(len(test_list)):
        if i == 0 or i == 3:
            test_index.extend(test_list[i])
    for i in range(len(val_list)):
        val_index.extend(val_list[i])
    
    train_index = sorted(train_index)
    test_index = sorted(test_index)
    val_index = sorted(val_index)
    
    train_index = torch.tensor(train_index)
    test_index = torch.tensor(test_index)
    val_index = torch.tensor(val_index)
    return train_index, test_index, val_index

def make_ood_dataset(train_list, test_list, val_list):
    train_index = []
    test_index = []
    val_index = []
    for i in range(len(train_list)):
        if i == 1 or i == 2:
            train_index.extend(train_list[i])
    for i in range(len(test_list)):
        if i == 0 or i == 3:
            test_index.extend(test_list[i])
    for i in range(len(val_list)):
        val_index.extend(val_list[i])
    
    train_index = sorted(train_index)
    test_index = sorted(test_index)
    val_index = sorted(val_index)
    
    train_index = torch.tensor(train_index)
    test_index = torch.tensor(test_index)
    val_index = torch.tensor(val_index)
    return train_index, test_index, val_index

if __name__ == '__main__':
    tr1 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\bot_positive\train.pt")
    tr2 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\human_positive\train.pt")
    tr3 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\bot_negative\train.pt")
    tr4 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\human_negative\train.pt")
    te1 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\bot_positive\test.pt")
    te2 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\human_positive\test.pt")
    te3 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\bot_negative\test.pt")
    te4 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\human_negative\test.pt")
    va1 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\bot_positive\val.pt")
    va2 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\human_positive\val.pt")
    va3 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\bot_negative\val.pt")
    va4 = torch.load(r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\split_index\human_negative\val.pt")
    
    train_list = [tr1, tr2, tr3, tr4]
    test_list = [te1, te2, te3, te4]
    val_list = [va1, va2, va3, va4]
    
    normal_train_index, normal_test_index, normal_val_index = make_normal_dataset(train_list, test_list, val_list)
    ood_train_index, ood_test_index, ood_val_index = make_ood_dataset(train_list, test_list, val_list)
    
    torch.save(normal_train_index, r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\normal\train.pt")
    torch.save(normal_test_index, r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\normal\test.pt")
    torch.save(normal_val_index, r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\normal\val.pt")
    
    torch.save(ood_train_index, r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\ood\train.pt")
    torch.save(ood_test_index, r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\ood\test.pt")
    torch.save(ood_val_index, r"D:\bot_analyze_shortcut\analyze\sentiment_split\sentiment_split\ood\val.pt")





