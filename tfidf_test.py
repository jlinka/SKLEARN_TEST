import os
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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


# tfidf_vectorizer = TfidfVectorizer(min_df=1)
# tfidf_matrix = tfidf_vectorizer.fit_transform(vectorizer.fit_transform(c))
#
# print(tfidf_matrix.todense())


# 这里将每份文档词语的TF-IDF写入tfidffile文件夹中保存
# for i in range(len(weight)):
#     #print(u"--------Writing all the tf-idf in the", i, u" file into ", sFilePath + '/' + string.zfill(i, 5) + '.txt', "--------")
#
#     f = open('tfidf4.txt', 'w+')
#     for j in range(len(word)):
#         # if weight[i][j] != 0:
#         #print(word[j] + "    " + str(weight[i][j]) + "\n")
#         f.write(word[j] + ":" + str(weight[i][j]) + "\n")
#     f.close()
# endtime = datetime.datetime.now()
# print(endtime - starttime)
num = 1
filename = 'tfidf3/tfidf'+str(num)+'.txt'
for i in range(len(weight)) :
    #print( u"--------Writing all the tf-idf in the",i,u" file into ",sFilePath+'/'+string.zfill(i,5)+'.txt',"--------")
    f = open(filename, 'w', encoding='UTF-8-sig')
    for j in range(len(word)) :
        f.write(word[j] + ":" + str(weight[i][j]) + "\n")
        # if(weight[i][j] != 0):
        #
    num += 1
    filename = 'tfidf3/tfidf' + str(num) + '.txt'
    f.close()

