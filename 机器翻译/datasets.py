import jieba
from utils import normalizeString
from utils import cht_to_chs

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {
            0: "SOS", 1: "EOS"
        }
        self.n_words = 2

    # 单词操作
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # 句子操作
    def addSentences(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)


def readLangs(lang1, lang2, path):
    lines = open(path, encoding="utf-8").readlines()

    lang1_cls = Lang(lang1)
    lang2_cls = Lang(lang2)

    pairs = []
    for l in lines:
        l = l.split("\t")
        sentence1 = normalizeString(l[0])
        sentence2 = cht_to_chs(l[1])
        seg_list = jieba.cut(sentence2, cut_all=False)
        sentence2 = " ".join(seg_list)
        # 过滤
        if len(sentence1.split(" ")) > MAX_LENGTH:
            continue
        if len(sentence2.split(" ")) > MAX_LENGTH:
            continue

        pairs.append([sentence1, sentence2])
        lang1_cls.addSentences(sentence1)
        lang2_cls.addSentences(sentence2)

    return lang1_cls, lang2_cls, pairs


lang1 = "en"
lang2 = "cn"
path = "data/en-cn.txt"
lang1_cls, lang2_cls, pairs = readLangs(lang1, lang2, path)

# print(len(pairs))
# print(lang1_cls.n_words)
# print(lang1_cls.index2word)
#
# print(lang2_cls.n_words)
# print(lang2_cls.index2word)


# 英文文本
word2index_en = lang1_cls.word2index
# 中文文本
word2index_cn = lang2_cls.word2index

ff = open("data/en_voc_dict", "w")
for item in word2index_en.keys():
    ff.writelines("{},{}\n".format(item, word2index_en[item]))
ff.close()
ff = open("data/cn_voc_dict", "w", encoding='utf-8')
for item in word2index_cn.keys():
    ff.writelines("{},{}\n".format(item, word2index_cn[item]))
ff.close()
