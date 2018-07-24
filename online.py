import pandas as pd
import matplotlib.pyplot as plt
import os
import sets
from pymongo import MongoClient
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import math
#显示中文
import datetime
import numpy
import sys
import codecs


DIMENSION = 156

CLUSTERS = 88
def CalConDis(v1,v2,lengthVector):
    # 计算出两个向量的乘积
    B = 0
    i = 0
    while i < lengthVector:
        B = v1[i] * v2[i] + B
        i = i + 1
        # print('乘积 = ' + str(B))

        # 计算两个向量的模的乘积
    A = 0
    A1 = 0
    A2 = 0
    i = 0
    while i < lengthVector:
        A1 = A1 + v1[i] * v1[i]
        i = i + 1
        # print('A1 = ' + str(A1))

    i = 0
    while i < lengthVector:
        A2 = A2 + v2[i] * v2[i]
        i = i + 1
        # print('A2 = ' + str(A2))

    A = math.sqrt(A1) * math.sqrt(A2)
    if (A == 0):
        return 0
    G = format(float(B) / A, ".3f")
    if(float(G) > 0.65):
        print('两篇文章的相似度 = ' + format(float(B) / A, ".3f"))

    return G

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

input5 = open('test1.txt', 'a', encoding='utf-8', errors='ignore')
input5.write(str(weight))
input5.close()

estimator = PCA(n_components=DIMENSION)
pca_x_train = estimator.fit_transform(weight)
print(pca_x_train)

# n_clusters=4，参数设置需要的分类这里设置成4类
kmeans = KMeans(n_clusters=CLUSTERS, random_state=0).fit(pca_x_train)
#center为各类的聚类中心，保存在df_center的DataFrame中给数据加上标签
center = kmeans.cluster_centers_
df_center = pd.DataFrame(center)
#标注每个点的聚类结果
labels = kmeans.labels_
fenlei = list(labels)
print(labels)


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
q = 0
input = open('88score156test4.txt', 'a', encoding='utf-8', errors='ignore')
while i < CLUSTERS:
    input.write("第"+str(i)+"类:")
    print("第" + str(i) + "类:")

    input.write(str(fenlei.count(i)))

    i += 1
    input.write("\n")

input.close()


i = 0
j = 0
input = open('88pca156begin.txt', 'a', encoding='utf-8', errors='ignore')
while i < CLUSTERS:
    input.write("第"+str(i)+"类:\n")
    print("第" + str(i) + "类:")
    j = 0
    while j < 345:
        if fenlei[j] == i:
            input.write(title[j]+"\n"+content[j]+"\n\n")
        j += 1
    i += 1
    input.write("\n\n\n\n\n\n\n")

input.close()


i = 0
j = 0
a = []
test = {}
input = open('88pca156testdel.txt', 'a', encoding='utf-8', errors='ignore')
while i < CLUSTERS:
    input.write("第"+str(i)+"类:\n")
    print("第" + str(i) + "类:")
    j = 0
    a = []
    o = 0
    p = 0
    while j < 345:

        if fenlei[j] == i:

            a.append(content[j])

            if(o ==0):
                test.update({content[j]: j})
                input.write(title[j] + "\n" + content[j] + "\n\n")

            if(o != 0):
                p = 0
                for value in test.values():
                    p += 1
                    g = int(format(value))
                    cos = CalConDis(weight[j], weight[g], len(weight[j]))
                    if(float(cos) > 0.65):
                        break
                    # if(float(cos) <= 0.5):
                    #     print(title[j] + "\n" + content[j] + "\n\n")
                if(p ==len(test)):
                    if(float(cos)<0.65):
                        print(content[j])
                        input.write(title[j] + "\n" + content[j] + "\n\n")
                    #print('{values}:{keys}'.format(values=value, keys=key))
                test.update({content[j]: j})
            o += 1
        j += 1

    i += 1
    input.write("\n\n\n\n\n\n\n")

input.close()

# for o in range(len(a)):
#     if(set(content[j]) & set(a[o]) == set(a[0])):
#         n = weight[21]
#         m = weight[22]
#         cos = CalConDis(n, m, len(n))
#         print(cos)
#         break
# # if (o == 0):
# #     input.write(title[j] + "\n" + content[j] + "\n\n")
# if(o+1 == len(a)):
#     input.write(title[j] + "\n" + content[j] + "\n\n")
