import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


def generate2dgauss(u, s, n):
    length = sum(n)
    x = np.zeros(length)
    y = np.zeros(length)
    labels = np.zeros(length)
    cnt = 0
    for i in range(len(n)):
        x1 = np.random.normal(0, 1, n[i])
        y1 = np.random.normal(0, 1, n[i])
        x1 = s[i][0]*x1+u[i][0]
        y1 = s[i][1]*y1+u[i][1]
        x[cnt:cnt+n[i]] = x1
        y[cnt:cnt+n[i]] = y1
        labels[cnt:cnt+n[i]] = i
        cnt += n[i]
    from itertools import cycle
    plt.figure(1)
    plt.clf()
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(len(n)), colors):
        my_members = labels == k
        plt.plot(x[my_members], y[my_members], col + '.')
    plt.show()
    return x, y
class EM(object):
    def __init__(self, gnum, x, y):
        self.gnum = gnum
        self.x = x
        self.y = y
        self.n = len(x)
        self.u = np.zeros((gnum, 2))
        self.s = np.zeros((gnum, 2))
        self.pji = np.zeros((self.n, gnum))
        self.pj = np.zeros(gnum)
        self.initval()

    def initval(self):
        xmean = sum(self.x)/self.n
        ymean = sum(self.y)/self.n
        xksi2 = sum((self.x-xmean)**2)
        yksi2 = sum((self.y-ymean)**2)
        for i in range(self.gnum):
            self.u[i, 0] = random.uniform(-1, 1)*math.sqrt(xksi2)+xmean
            self.u[i, 1] = random.uniform(-1, 1)*math.sqrt(yksi2)+ymean
            self.s[i, 0] = math.sqrt(xksi2)
            self.s[i, 1] = math.sqrt(yksi2)

    def Emethod(self):
        for i in range(self.n):
            px = np.zeros(self.gnum)
            py = np.zeros(self.gnum)
            for j in range(self.gnum):
                px[j] = norm.pdf(self.x[i], self.u[j, 0], self.s[j, 0])
                py[j] = norm.pdf(self.y[i], self.u[j, 1], self.s[j, 1])
            p = px*py
            psum = sum(p)
            self.pji[i, :] = p/psum
            #for j in range(self.gnum):
            #    self.p[i, j] =
    def Mmethod(self):
        nj = np.sum(self.pji, axis=0)
        self.pj = nj/self.n
        self.u[:, 0] = self.pji.T.dot(self.x)/nj
        self.u[:, 1] = self.pji.T.dot(self.y)/nj
        for j in range(self.gnum):
            self.s[j, 0] = math.sqrt(self.pji[:, j].T.dot((self.x-self.u[j, 0])**2)/nj[j])
            self.s[j, 1] = math.sqrt(self.pji[:, j].T.dot((self.y-self.u[j, 1])**2)/nj[j])

    def opt(self, iter):
        for i in range(iter):
            self.Emethod()
            self.Mmethod()

    def plotresult(self):
        labels = np.argmax(self.pji, axis=1)
        # Plot result
        from itertools import cycle
        plt.figure(1)
        plt.clf()

        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(self.gnum), colors):
            my_members = labels == k
            plt.plot(self.x[my_members], self.y[my_members], col + '.')
        plt.show()
        print(self.u)
        print(self.s)
        print(self.pj)


if __name__ == "__main__":
    [x, y] = generate2dgauss([[0, 2], [10, 13], [5, 5], [5, 20]], [[3, 2], [4, 4], [2, 3], [3, 5]], [222, 150, 200, 250])
    em = EM(4, x, y)
    em.opt(20)
    em.plotresult()

