import numpy as np
import random

class NNet:
    def __init__(self, N):
        self.W = 0
        self.N = N
        self.T = 0
        self.x_t = -1
        print("Веса: ", self.W)

    def eval(self, X, Y):
        self.X = np.array(X)
        self.Y = Y
        self.NET = sum(self.X*self.W) + self.T * self.x_t
        self.predict = -1 if self.NET <= 0 else 1
        print ("Предсказано: ", self.predict, self.predict == self.Y)
        if self.predict != self.Y:
            self.LeanrHebb()


    def LeanrHebb(self):
        self.W = self.W + self.\
            X*self.Y
        print("Новые веса: ", self.W)


if __name__ == '__main__':
    nnet = NNet(2)

    #1
    x1 = 1; x2 = 1; y = 1
    print("\nx1=", x1, " x2=", x2, "y =",y)
    nnet.eval([x1,x2], y)
    # 2
    x1 = 1; x2 = -1; y = -1
    print("\nx1=", x1, " x2=", x2, "y =",y)
    nnet.eval([x1,x2], y)
    # 3
    x1 = -1; x2 = 1; y = -1
    print("\nx1=", x1, " x2=", x2, "y =", y)
    nnet.eval([x1, x2], y)
    # 4
    x1 = -1; x2 = -1; y = -1
    print("\nx1=", x1, " x2=", x2, "y =",y)
    nnet.eval([x1,x2], y)

    #1
    x1 = 1; x2 = 1; y = 1
    print("\nx1=", x1, " x2=", x2, "y =",y)
    nnet.eval([x1,x2], y)
