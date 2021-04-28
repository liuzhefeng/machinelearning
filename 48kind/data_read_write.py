import jieba

data_path = "data/label_review.csv"
stop_words = "data/hit_stopword"
# 从第二行开始读取
data_list = open(data_path, encoding='utf-8').readlines()[1:]
stop_words_list = open(stop_words, encoding='utf-8').readlines()
stop_words_list = [line.strip() for line in stop_words_list]
stop_words_list.append(" ")
# 词-出现次数 
voc_dic = {}
# 最小频率
min_seq = 1
top_n = 1000
UNK = "<UNK>"
PAD = "<PAD>"

for item in data_list:
    label = item[0]
    content = item[2:].strip()
    # print(content)
    seg_list = jieba.cut(content, cut_all=False)
    # 结果集
    seg_res = []
    for seg_item in seg_list:
        if seg_item in stop_words_list:
            continue
        seg_res.append(seg_item)
        if seg_item in voc_dic.keys():
            voc_dic[seg_item] = voc_dic[seg_item] + 1
        else:
            voc_dic[seg_item] = 1
    # print(seg_res)
    # print(content)

# voc_dic排序
# [('嘻嘻', 28),('',)]形式
voc_list = sorted([_ for _ in voc_dic.items() if _[1] > min_seq],
                  key=lambda x: x[1],
                  reverse=True
                  )[:top_n]
# print(voc_list)
# 格式：{'嘻嘻': 0, '鼓掌': 1, '爱': 2, '都': 3, '旅游‘}
# 做成词典表
voc_dic = {word_count[0]: idx for idx, word_count in enumerate(voc_list)}
# print(voc_dic)

# 添加UNK未知词、PAD添加空缺
voc_dic.update({UNK: len(voc_dic), PAD: len(voc_dic) + 1})
# print(voc_dic)
ff = open("data/voc_dict", "w")
for item in voc_dic.keys():
    ff.writelines("{},{}\n".format(item, voc_dic[item]))
ff.close()
