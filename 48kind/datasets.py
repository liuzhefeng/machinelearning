from torch.utils.data import DataLoader, Dataset
import jieba
import numpy as np


# 读取voc_dict返回字典
def read_dict(voc_dict_path):
    voc_dict = {}
    dict_list = open(voc_dict_path).readlines()
    for item in dict_list:
        item = item.split(",")
        voc_dict[item[0]] = int(item[1].strip())
    return voc_dict


# 加载数据、去停用词
def load_data(data_path, data_stop_path):
    data_list = open(data_path, encoding='utf-8').readlines()[1:]
    # 加载停用词、放入list
    stops_word = open(data_stop_path, encoding='utf-8').readlines()
    stops_word = [line.strip() for line in stops_word]
    stops_word.append(" ")
    voc_dict = {}
    data = []
    max_seq_len = 0
    np.random.shuffle(data_list)
    for item in data_list[:]:
        label = item[0]
        content = item[2:].strip()
        seg_list = jieba.cut(content, cut_all=False)
        seg_res = []
        for seg_item in seg_list:
            if seg_item in stops_word:
                continue
            seg_res.append(seg_item)
            if seg_item in voc_dict.keys():
                voc_dict[seg_item] = voc_dict[seg_item] + 1
            else:
                voc_dict[seg_item] = 1
        # 返回最长句子长度
        if len(seg_res) > max_seq_len:
            max_seq_len = len(seg_res)
        data.append([label, seg_res])
    return data, max_seq_len


UNK = "<UNK>"
PAD = "<PAD>"
from configs import Config

config = Config()


class text_CLS(Dataset):
    def __init__(self, voc_dict_path, data_path, data_stop_path, max_len_seq=None):
        self.voc_dict = read_dict(voc_dict_path)
        self.data_path = data_path
        self.data_stop_path = data_stop_path
        self.data, self.max_seq_len = load_data(data_path, data_stop_path)
        if max_len_seq is not None:
            self.max_seq_len = max_len_seq
        np.random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        label = int(data[0])
        word_list = data[1]
        input_idx = []
        for word in word_list:
            if word in self.voc_dict.keys():
                input_idx.append(self.voc_dict[word])
            else:
                input_idx.append(self.voc_dict[UNK])
        if len(input_idx) < self.max_seq_len:
            input_idx += [self.voc_dict[PAD]
                          for _ in range(self.max_seq_len - len(input_idx))]
        data = np.array(input_idx)
        return label, data


def data_loader(data_path, data_stop_path, voc_dict_path):
    dataset = text_CLS(voc_dict_path, data_path, data_stop_path)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=config.is_shuffle)


if __name__ == "__main__":
    data_path = "data/label_review.csv"
    data_stop_path = "data/hit_stopword"
    voc_dict_path = "data/voc_dict"
    train_dataloader = data_loader(data_path, data_stop_path, voc_dict_path)
    for i, batch in enumerate(train_dataloader):
        print(i, batch)
