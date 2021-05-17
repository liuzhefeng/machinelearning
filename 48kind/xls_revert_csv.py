import numpy as np
import pandas as pd
import csv
import re
# txt = np.loadtxt("data/train.txt",encoding="utf-8")
# txtDF = pd.DataFrame(txt)
# txtDF.to_csv('data/train.csv',index=False)

# reader = open('data/train.txt', encoding='utf-8')
reader = open('data/test.txt', encoding='utf-8')
list_data = reader.readlines()
# columns = list_data [0].split()
length = len(list_data)

list = [[] for i in range(length)]
# list = []
for i in range(length):
    small_list = list_data[i].split(" ")
    convert_list = [small_list[1], small_list[0]]
    list[i].append(small_list[1].strip())
    list[i].append(re.sub('\s', '', small_list[0]))

# print(lists)
table = ['label', 'review']
test = pd.DataFrame(columns=table, data=list)
# print(test)
test.to_csv("data/label_review_test.csv", index=False)
# with open("test.csv","wb") as csvfile:
#     writer = csv.writer(csvfile)
#     #先写入columns_name
#     writer.writerow(columns)
#     #写入多行用writerows
#     writer.writerows(list)
