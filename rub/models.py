import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 1002 256
        # word = [[1, 2, 3],
        #         [2, 3, 4]]
        # embed = 2*3*256
        self.embeding = nn.Embedding(config.n_vocab,
                                     config.embed_size,
                                     padding_idx=config.n_vocab - 1)
        # input_size:输入的维度
        # hidden_size:h的维度
        # num_layers:堆叠LSTM的层数
        # bias:偏执
        # batch_first:True(则input为(batch-句子数量, seq-每个句子的单词数即长度, input_size-每个词维度));(10,24,100)
        # 默认值为：False（seq_len, batch, input_size）
        # bidirectional:是否双向传播，默认值为False

        # LSTM 的输入：input，（h_0，c_0）
        # input：输入数据，shape 为（句子长度seq_len, 句子数量batch, 每个单词向量的长度input_size）；
        # h_0：默认为0，shape 为（num_layers * num_directions单向为1双向为2, batch, 隐藏层节点数hidden_size）；
        # c_0：默认为0，shape 为（num_layers * num_directions, batch, hidden_size）；

        # LSTM 的输出：output，（h_n，c_n）
        # output：输出的 shape 为（seq_len, batch, num_directions * hidden_size）；
        # h_n：shape 为（num_layers * num_directions, batch, hidden_size）；
        # c_n：shape 为（num_layers * num_directions, batch, hidden_size）；

        self.lstm = nn.LSTM(config.embed_size,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        # 步长默认为kernel_size
        self.maxpool = nn.MaxPool1d(config.pad_size)
        # 加
        # self.conv = torch.nn.Sequential(
        #     torch.nn.Conv1d(in_channels=config.hidden_size * 2 + config.embed_size,
        #                     out_channels=((config.hidden_size * 2 + config.embed_size) / 2),
        #                     kernel_size=2)
        # )
        # 输入768（256*2+256）
        self.fc = nn.Linear((config.hidden_size * 2 + config.embed_size),
                            config.num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        embed = self.embeding(x)
        # 1*640 -> embed size: torch.Size([1, 640, 256])
        # print("embed size:", embed.size())
        out, _ = self.lstm(embed)
        # torch.Size([1, 640, 256])->torch.Size([1, 640, 512]) Bidirect
        out = torch.cat((embed, out), 2)
        out = torch.relu(out)
        # torch.Size([1, 640, 768])->torch.Size([1, 768, 640])
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).reshape(out.size()[0], -1)

        # 加
        # out = self.conv(out)
        # out = out.view(out.size()[0], -1)
        # out = self.fc(out)

        # print(out.size())
        out = self.fc(out)
        out = self.softmax(out)
        return out


if __name__ == "__main__":
    from configs import Config

    # test
    cfg = Config()
    cfg.pad_size = 640
    model_textcls = Model(config=cfg)
    input_tensor = torch.tensor([i for i in range(640)]).reshape([1, 640])
    out_tensor = model_textcls.forward(input_tensor)
    print(out_tensor.size())
    print(out_tensor)
