import matplotlib.pyplot as plt
from pylab import *
import os

#显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
from nltk import *
num = 1
c = []
splitfilename = "C:/Users/jlink/PycharmProjects/fiveG_test/split_text/split" + str(num) + ".txt"
while os.path.exists(splitfilename):
    with open(splitfilename, 'r+', encoding='UTF-8-sig', errors='ignore') as wf:
        word_lst = []
        a = wf.read()
        word_lst = list(a.split(','))
        c.extend(word_lst)
        #word_lst.append(a.split(' '))

    num += 1
    splitfilename = "C:/Users/jlink/PycharmProjects/fiveG_test/split_text/split" + str(num) + ".txt"
fdist1 = FreqDist(c)

fdist1.plot(100)

