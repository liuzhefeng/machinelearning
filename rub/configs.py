import torch


class Config():
    def __init__(self):
        # 字典数
        self.n_vocab = 1002
        # 词向量维度
        self.embed_size = 256
        # 隐藏层大小
        self.hidden_size = 256
        # LSTM层数
        self.num_layers = 2
        # 丢弃率
        self.dropout = 0.5
        # 类别
        self.num_classes = 4
        # 每句话处理的长度
        self.pad_size = 128
        # mini batch
        self.batch_size = 32
        # 随机化数据
        self.is_shuffle = True
        # 学习率
        self.learn_rate = 0.001
        # epoch
        self.num_epochs = 1000
        # cpu/gpu训练
        self.devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
