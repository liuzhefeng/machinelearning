import torch
import torch.nn as nn
from torch import optim
from models import Model
from datasets import text_CLS
from configs import Config
from torch.utils.data import DataLoader

cfg = Config()
data_path = "data/label_review.csv"
data_stop_path = "data/hit_stopword"
dict_path = "data/voc_dict"

dataset = text_CLS(dict_path, data_path, data_stop_path)
train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.is_shuffle)

cfg.pad_size = dataset.max_seq_len
print(cfg.pad_size)

model_text_cls = Model(cfg)
# model_text_cls.to(cfg.devices)
model_text_cls.load_state_dict(torch.load("model1/700.pth"))

for i, batch in enumerate(train_dataloader):
    label, data = batch
    data = torch.tensor(data).to(cfg.devices)
    label = torch.tensor(label, dtype=torch.int64).to(cfg.devices)
    pred_softmax = model_text_cls.forward(data)
    pred = torch.argmax(pred_softmax, dim=1)
    out = torch.eq(pred, label)
    print("准确率：", out.sum() * 1.0 / pred.size()[0])
