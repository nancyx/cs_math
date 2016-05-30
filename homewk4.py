#from math import *
import numpy as np
import algopy
#f=a*exp(-b*x)
class lm(object):
    def __init__(self, yfi=0.0001, u=1.0):
        self.yfi = yfi
        self.u = u

    def derfun(self, fun, x0):
        cg = algopy.CGraph()
        x = algopy.Function(x0)
        y = fun(x)
        cg.trace_off()
        cg.independentFunctionList = [x]
        cg.dependentFunctionList = [y]
        return cg
    def setu(self, u):
        self.u = u

    def settol(self, yfi):
        self.yfi = yfi

    def opt(self, fun, x0):
        x = x0
        u = self.u
        size = x.shape[0]
        cg = self.derfun(fun, x0)
        g = cg.gradient(x).T
        G = cg.hessian(x)
        while np.linalg.norm(g) > self.yfi:
            grad = G+u*np.eye(size)
            while not self.is_pos_def(grad):
                u *= 4
                grad = G+u*np.eye(size)
            s = np.linalg.solve(grad, -g)
            fk = fun(x)
            gk = fk
            fknew = fun(x+s)
            gknew = fk+g.T.dot(s)+0.5*s.T.dot(G).dot(s)
            detfk = fknew - fk
            detgk = gknew - gk
            rk = detfk/detgk
            if rk < 0.25:
                u *= 4
            if rk > 0.75:
                u /= 2
            if rk > 0:
                x = x+s
            g = cg.gradient(x).T
            G = cg.hessian(x)
        return x, fknew

    def is_pos_def(self, x):
        return np.all(np.linalg.eigvals(x) > 0)

if __name__ == "__main__":
    lm = lm()
    x = np.array([3.0, 2.0])
    rosen = lambda x: (1.-x[0])**2 + 105*(x[1]-x[0]**2)**2
    [x, f] = lm.opt(rosen, x)
    print("x = ", x)
    print("f = ", f)
    x = np.array([3.0, 2.0])
    fun = lambda x: np.exp(x[0]**2+x[1])
    [x, f] = lm.opt(fun, x)
    print("x = ", x)
    print("f = ", f)
    f = lambda x: -2*np.exp(-x[0])*np.sin(x[0])
    x = np.array([0])
    [x, f] = lm.opt(f, x)
    print("x = ", x)
    print("f = ", f)