import pandas as pd
import re

data_xls1 = pd.read_excel('data/0级平稳.xls', index_col=0)
# print(len(data_xls.index._values))
# print(len(list1))
# print(list1[0])
data_xls2 = pd.read_excel('data/1级轻微.xls', index_col=0)
data_xls3 = pd.read_excel('data/2级中度.xls', index_col=0)
data_xls4 = pd.read_excel('data/3级严重.xls', index_col=0)

list1 = data_xls1.index._values
list2 = data_xls2.index._values
list3 = data_xls3.index._values
list4 = data_xls4.index._values
length = len(list1) + len(list2) + len(list3) + len(list4)
lists = [[] for i in range(length)]

for i in range(len(list1)):
    lists[i].append(0)
    lists[i].append(re.sub('\s','',list1[i]))
for i in range(len(list1), len(list1) + len(list2)):
    lists[i].append(1)
    lists[i].append(re.sub('\s','',list2[i - len(list1)]))
for i in range(len(list1) + len(list2), len(list1) + len(list2) + len(list3)):
    lists[i].append(2)
    lists[i].append(re.sub('\s','',list3[i - (len(list1) + len(list2))]))
for i in range(len(list1) + len(list2) + len(list3), len(list1) + len(list2) + len(list3) + len(list4)):
    lists[i].append(3)
    lists[i].append(re.sub('\s','',list4[i - (len(list1) + len(list2) + len(list3))]))
# print(lists)
table = ['label', 'review']
test = pd.DataFrame(columns=table, data=lists)
# print(test)
test.to_csv("data/label_review.csv",index=False)