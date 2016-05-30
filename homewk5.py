import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy.optimize import fmin_l_bfgs_b


def svm(x, y):
    X = np.append(np.ones([x.shape[0], 1]), x, 1)
    Y = y
    c = 0.001
    def func(w, *args):
        X,Y,c = args
        yp = np.dot(X, w)
        idx = np.where(yp*Y < 1)[0]
        e = yp[idx] - Y[idx]
        cost = np.dot(e, e)+c*np.dot(w, w)
        grad = 2*(np.dot(X[idx].T, e)+c*w)
        return cost, grad

    RET = fmin_l_bfgs_b(func, x0=np.random.rand(X.shape[1]), args=(X, Y, c), approx_grad=False)
    w = RET[0]
    margin = 2/np.sqrt(np.dot(w[1:3], w[1:3]))
    plot_x = np.append(np.min(x, 0)[0]-0.2, np.max(x, 0)[0]+0.2)
    plot_y = -(plot_x*w[1]+w[0])/w[2]

    plt.figure()
    pos = (Y == 1)
    neg = (Y != 1)
    plt.plot(x[pos][:, 0], x[pos][:, 1], "r+", label="Positive Samples")
    plt.plot(x[neg][:, 0], x[neg][:, 1], "bo", label="Negative Samples")
    plt.plot(plot_x, plot_y, "r-", label="Decision boundary")
    plt.plot(plot_x, plot_y+margin / 2, 'g-', label="")
    plt.plot(plot_x, plot_y-margin / 2, 'g-', label="")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    y[y == 0] = -1
    index = y != 2
    svm(x[index], y[index])
