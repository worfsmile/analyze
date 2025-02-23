import torch

label = torch.load(r"D:\bot_analyze_shortcut\data\bot_classifier_data_analyze\twi-22_label\label.pt")
sentiment_label = torch.load(r"D:\bot_analyze_shortcut\data\bot_classifier_data_analyze\twi-22_sentiment\tweets20_sentiment.pt")

bot_indices = torch.nonzero(label == 1).squeeze()
human_indices = torch.nonzero(label == 0).squeeze()

bot_sentiment = sentiment_label[bot_indices]
human_sentiment = sentiment_label[human_indices]

bot_none_indices = bot_indices[torch.nonzero(bot_sentiment == -1).squeeze()]
bot_negtive_indices = bot_indices[torch.nonzero(bot_sentiment == 0).squeeze()]
bot_neutral_indices = bot_indices[torch.nonzero(bot_sentiment == 1).squeeze()]
bot_positive_indices = bot_indices[torch.nonzero(bot_sentiment == 2).squeeze()]
human_none_indices = human_indices[torch.nonzero(human_sentiment == -1).squeeze()]
human_negtive_indices = human_indices[torch.nonzero(human_sentiment == 0).squeeze()]
human_neutral_indices = human_indices[torch.nonzero(human_sentiment == 1).squeeze()]
human_positive_indices = human_indices[torch.nonzero(human_sentiment == 2).squeeze()]

torch.save(bot_none_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\bot_negtive.pt")
torch.save(bot_negtive_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\bot_negtive.pt")
torch.save(bot_neutral_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\bot_neutral.pt")
torch.save(bot_positive_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\bot_positive.pt")
torch.save(human_none_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\human_negtive.pt")
torch.save(human_negtive_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\human_negtive.pt")
torch.save(human_neutral_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\human_neutral.pt")
torch.save(human_positive_indices, r"D:\bot_analyze_shortcut\data\sentiment_split\split_index\human_positive.pt")


