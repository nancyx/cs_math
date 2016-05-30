#coding:utf-8
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.optimize as opt


def samplefunction(f, count):
    xvalue = np.random.rand(count)
    yvalue = np.array([f(x) for x in xvalue])
    noise = np.random.normal(0, 0.1, (count))
    #print(yvalue+noise)
    return [xvalue, yvalue+noise]

def polynomialregression(x, y, degree):
    length = x.shape[0]
    pvalue = np.zeros((length, degree+1))
    for i in range(degree+1):
        pvalue[:, i] = x**i
    y = pvalue.T.dot(y)
    pvalue = pvalue.T.dot(pvalue)
    w = np.linalg.solve(pvalue, y)
    #print(w)
    return w

def evalregressionvalue(w, x, degree):
    length = x.shape[0]
    pvalue = np.zeros((length, degree+1))
    for i in range(degree+1):
        pvalue[:, i] = x**i
    ry = pvalue.dot(w)
    return ry

def polynomialregressionwithterm(x, y, la, degree):
    def objectfunction(w, *args):
        x = args[0]
        y = args[1]
        la = args[2]
        degree = args[3]
        ry = evalregressionvalue(w, x, degree)
        return np.sum((ry-y)**2)*0.5+0.5*la*np.sum(w**2)
    w0 = np.random.rand(degree+1)
    la = math.exp(la)
    xopt = opt.fmin_powell(objectfunction, w0, (x, y, la, degree))
    #print xopt
    return xopt




if __name__ == "__main__":
    samplevalue = samplefunction(lambda x: math.sin(x*2*math.pi), 10)
    x = np.linspace(0, 1, 100)
    y = np.sin(x*2*math.pi)
    w = polynomialregression(samplevalue[0], samplevalue[1], 3)
    ry = evalregressionvalue(w, x, 3)
    plt.figure(1)
    plt.plot(x, y, color="green", label="sin(x)")
    plt.plot(samplevalue[0], samplevalue[1], '*r', label="sample point")
    plt.plot(x, ry, color="red", label="regression curve")
    plt.show()
    plt.figure(2)
    w = polynomialregression(samplevalue[0], samplevalue[1], 9)
    ry = evalregressionvalue(w, x, 9)
    plt.plot(x, y, color="green", label="sin(x)")
    plt.plot(samplevalue[0], samplevalue[1], '*r', label="sample point")
    plt.plot(x, ry, color="red", label="regression curve")
    plt.show()
    plt.figure(3)
    samplevalue = samplefunction(lambda x: math.sin(x*2*math.pi), 15)
    w = polynomialregression(samplevalue[0], samplevalue[1], 9)
    ry = evalregressionvalue(w, x, 9)
    plt.plot(x, y, color="green", label="sin(x)")
    plt.plot(samplevalue[0], samplevalue[1], '*r', label="sample point")
    plt.plot(x, ry, color="red", label="regression curve")
    plt.show()
    plt.figure(4)
    samplevalue = samplefunction(lambda x: math.sin(x*2*math.pi), 100)
    w = polynomialregression(samplevalue[0], samplevalue[1], 9)
    ry = evalregressionvalue(w, x, 9)
    plt.plot(x, y, color="green", label="sin(x)")
    plt.plot(samplevalue[0], samplevalue[1], '*r', label="sample point")
    plt.plot(x, ry, color="red", label="regression curve")
    plt.show()
    plt.figure(5)
    samplevalue = samplefunction(lambda x: math.sin(x*2*math.pi), 10)
    w = polynomialregressionwithterm(samplevalue[0], samplevalue[1], -10, 9)
    ry = evalregressionvalue(w, x, 9)
    plt.plot(x, y, color="green", label="sin(x)")
    plt.plot(samplevalue[0], samplevalue[1], '*r', label="sample point")
    plt.plot(x, ry, color="red", label="regression curve")
    plt.show()

