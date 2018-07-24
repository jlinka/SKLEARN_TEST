import pandas as pd
import matplotlib.pyplot as plt
import os
from pymongo import MongoClient
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#显示中文
import datetime
import numpy
import sys
import codecs

DIMENSION = 156
starttime = datetime.datetime.now()
print(starttime)
num = 1
c = []
splitfilename = "split_text/split" + str(num) + ".txt"
while os.path.exists(splitfilename):
    with open(splitfilename, 'r+', encoding='UTF-8-sig', errors='ignore') as wf:
        word_lst = []
        a = wf.read()
        #word_lst = list(a.split(','))
        a = "".join(a)
        c.append(a)
        #word_lst.append(a.split(' '))

    num += 1
    splitfilename = "split_text/split" + str(num) + ".txt"

vectorizer = CountVectorizer()
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(c))

word = vectorizer.get_feature_names()  # 所有文本的关键字
weight = tfidf.toarray()  # 对应的tfidf矩阵
print(weight)
# vec = list(weight)
# vec1 = ''.join(vec)
# input5 = open('vec.txt', 'a', encoding='utf-8', errors='ignore')
#
# input5.write(str(vec1))



# input5.close()
estimator = PCA(n_components=DIMENSION)
pca_x_train = estimator.fit_transform(weight)
print(pca_x_train)

# n_clusters=4，参数设置需要的分类这里设置成4类
kmeans = KMeans(n_clusters=177, random_state=0).fit(pca_x_train)
#center为各类的聚类中心，保存在df_center的DataFrame中给数据加上标签
center = kmeans.cluster_centers_
df_center = pd.DataFrame(center)
#标注每个点的聚类结果
labels = kmeans.labels_
fenlei = list(labels)
print(labels)


tsne = TSNE(n_components=2)
decomposition_data = tsne.fit_transform(weight)
x = []
y = []

for i in decomposition_data:
    x.append(i[0])
    y.append(i[1])

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
plt.scatter(x, y, c=kmeans.labels_, marker="x")
plt.xticks(())
plt.yticks(())
# plt.show()
plt.savefig('./156.png', aspect=1)


client = MongoClient('localhost', 27017)
db = client.FiveG_news
collection = db.xinChuang_topic



c = []
d = []
content = []
title = []
num = 1
myquery = {"isLimitSource": 1}

a = collection.find(myquery)
for item in a:

    c.append(item['content'])
    d.append(item['title'])
    for ia in c:


        #print(i + ",")

        a = "".join(ia).replace('\r', '').replace('\n', '').replace(' ', '').replace('  ', '').replace('   ', '') \
            .replace('    ', '').replace('     ', '').replace('      ', '').replace('       ', '').replace('        ',
                                                                                                           '') \
            .replace('         ', '').replace('          ', '').replace(
            '\r\n            \r\n              \r\n    \r\n    '
            '        \r\n           \r\n    ', '') \
            .replace(
            '\r\n                         \r\n                              \r\n                                '
            '   ', '').replace('	', '').replace('　　', '').replace('', '').replace('　　 ', '')
    content.append(a)
    for ij in d:

        #print(i + ",")

        e = "".join(ij).replace('\r', '').replace('\n', '').replace(' ', '').replace('  ', '').replace('   ', '') \
            .replace('    ', '').replace('     ', '').replace('      ', '').replace('       ', '').replace('        ',
                                                                                                           '') \
            .replace('         ', '').replace('          ', '').replace(
            '\r\n            \r\n              \r\n    \r\n    '
            '        \r\n           \r\n    ', '') \
            .replace(
            '\r\n                         \r\n                              \r\n                                '
            '   ', '').replace('	', '').replace('　　', '').replace('', '').replace('　　 ', '')
    title.append(e)
#print(content)

i = 0
while i < len(title):
    print(title[i]+str(i + 1)+content[i]+"\n")
   # print(title[i], str(i+1), content[i], "\n")
    i = i + 1

i = 0
j = 0
input = open('pca156.txt', 'a', encoding='utf-8', errors='ignore')
while i < 177:
    input.write("第"+str(i)+"类:\n")
    j = 0
    while j < 345:
        if fenlei[j] == i:
            input.write(title[j]+"\n"+content[j]+"\n\n")
        j += 1
    i += 1
    input.write("\n\n\n\n\n\n\n")

input.close()

# i = 0
# j = 0
# q = 0
# input = open('score78.txt', 'a', encoding='utf-8', errors='ignore')
# while i < 177:
#     input.write("第"+str(i)+"类:")
#     j = 0
#     q = 0
#     while j < 345:
#         if fenlei[j] == i:
#             q += 1
#         j += 1
#     input.write(str(q))
#
#     i += 1
#     input.write("\n")
#
# input.close()


i = 0
j = 0
q = 0
input = open('score156.txt', 'a', encoding='utf-8', errors='ignore')
while i < 177:
    input.write("第"+str(i)+"类:")

    input.write(str(fenlei.count(i)))

    i += 1
    input.write("\n")

input.close()