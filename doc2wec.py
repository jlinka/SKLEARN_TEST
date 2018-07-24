import pandas as pd
import matplotlib.pyplot as plt
import os
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
#显示中文
import datetime
import numpy
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
#print(weight)



# n_clusters=4，参数设置需要的分类这里设置成4类
kmeans = KMeans(n_clusters=177, random_state=0).fit(weight)
#center为各类的聚类中心，保存在df_center的DataFrame中给数据加上标签
center = kmeans.cluster_centers_
df_center = pd.DataFrame(center)
#标注每个点的聚类结果
labels = kmeans.labels_
# print(labels)
# print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

# i = 1
# while i <= len(kmeans.labels_):
#     print(i, kmeans.labels_[i-1])
#     i += 1
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
plt.savefig('./sample7.png', aspect=1)