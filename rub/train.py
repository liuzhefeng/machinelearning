import torch
import torch.nn as nn
from torch import optim
from models import Model
from torch.utils.data import DataLoader, Dataset
from datasets import data_loader, text_CLS
from configs import Config

cfg = Config()
data_path = "data/label_review.csv"
data_stop_path = "data/hit_stopword"
dict_path = "data/voc_dict"

dataset = text_CLS(dict_path, data_path, data_stop_path)
train_dataloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=cfg.is_shuffle)
cfg.pad_size = dataset.max_seq_len

model_text_cls = Model(cfg)
model_text_cls.to(cfg.devices)
loss_func = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_text_cls.parameters(), lr=cfg.learn_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=3,
                                            gamma=0.9)
for epoch in range(cfg.num_epochs):
    for i, batch in enumerate(train_dataloader):
        label, data = batch
        label = label.type(torch.int64)
        pred_softmax = model_text_cls.forward(data)
        loss_val = loss_func(pred_softmax, label)
        pred = torch.argmax(pred_softmax, dim=1)
        # print(pred)
        # print(label)
        out = torch.eq(pred, label)
        print(out)
        print("epoch is {}, ite is {}, val is {}".format(epoch, i, loss_val))
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

    scheduler.step()
    if epoch % 100 == 0:
        torch.save(model_text_cls.state_dict(), "model1/{}.pth".format(epoch))
