#coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def readfile():
    file = open("optdigits.tra", 'r')
    lines = file.readlines()
    file.close()
    digitdata = np.zeros((64, 389))
    k = 0
    for i in range(len(lines)):
        line = lines[i]
        if line[-2] == '3':
            line = line[:-2]
            pixels = line.split(",")
            pixels = [int(x) for x in pixels[:-1]]
            digitdata[:, k] = np.array(pixels)
            k += 1
        #print(pixels)
    #print(digitdata)
    return digitdata

if __name__ == "__main__":
    digitdata = readfile()  #the data teacher gived
    u, s, v = np.linalg.svd(digitdata, full_matrices=True)
    #plt.figure(1)
    #digitdata = digitdata.T
    #digitdata = digitdata.reshape((8*v.shape[0], 8))
    #imgplot = plt.imshow(digitdata[0:7, :], cmap=plt.cm.Greys_r)
    #plt.show()
    x = v[0, :]
    y = v[1, :]
    S = np.zeros((64, v.shape[0]))
    S[0, 0] = s[0]
    S[1, 1] = s[1]
    pcadata = (u.dot(S)).dot(v)
    plt.figure(1)
    plt.plot(x, y, '*r', label="pca point")
    plt.show()
    plt.figure(2)
    #pcadata = pcadata.reshape((8*v.shape[0], 8))
    num = 5
    #result = np.zeros(((num+1)*8, (num+1)*8))
    for i in range(25):
        pcasample = pcadata[:, i]
        pcasample = pcasample.reshape((8, 8))
        a = i//num
        b = i % num
        plt.subplot(5, 5, i)
        #result[a*8:(a*8+8), b*8:(b*8+8)] = pcasample
        imgplot = plt.imshow(pcasample, cmap=plt.cm.Greys_r)
    plt.show()

    #from sklearn.datasets import load_digits
    #digits = load_digits()
    #print(digits.data.shape)
    #import pylab as pl
    #pl.gray()
    #pl.matshow(digits.images[0])
    #pl.show()


