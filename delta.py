import numpy as np
import random
import keras

#Алгоритм обучения Розенблатта (дельта-правило)
class Neuron:
    def __init__(self, n_inputs):
        self.N = n_inputs
        self.W = np.array([0.0]*self.N)
        self.T = 0.0
        self.x_t = -1
        self.X = np.array
        self.D = 0.0

    def eval(self, X):
        self.X = np.array(X)
        self.S = sum(self.X*self.W) + self.T * self.x_t
        # print("Summ: ",self.S)
        self.NET = -1 if self.S <= 0 else 1
        return self.NET

    def modify(self, e_cur, a):
        self.W += a * e_cur * self.X
        # print('Weights:', self.W)
        self.T += a * e_cur * self.x_t

class NNet:
    def __init__(self, n_inputs, n_outputs, a, e):
        self.N = n_outputs
        self.neuron = [Neuron(n_inputs) for i in range(self.N)]
        self.X = np.array
        self.D = np.array
        self.Y = np.array([0.0]*self.N)
        self.E_sqr = 0.0
        self.e = e
        self.a = a
        self.E = np.array([0.0]*self.N)

    def train(self, X, D):
        self.X = np.array(X)
        self.D = np.array(D)
        for i in range(self.N):
            self.Y[i] = self.neuron[i].eval(self.X)

        self.E_sqr = sum(pow(self.D - self.Y, 2)) * 0.5
        print("Ошибка: ", self.E_sqr, self.D, self.Y[0])
        if self.E_sqr > self.e:
            self.E = self.D - self.Y
            for i in range(self.N):
                self.neuron[i].modify(self.E[i], self.a)

    def predict(self, X):
        self.X = np.array(X)
        for i in range(self.N):
            self.Y[i] = self.neuron[i].eval(self.X)
        print("Предсказано: ", int(self.Y[0]))
        return self.Y

if __name__ == '__main__':
    nNet = NNet(2,1,1,0)

    X = [[1,1],
         [-1,1],
         [1,-1],
         [-1,-1]]

    Y = [1, 1, 1, -1]

    for i in range(4):
        nNet.train(X[i], Y[i])

    for i in range(4):
        nNet.train(X[i], Y[i])

    for i in range(4):
        print("\nПодано: ", X[i])
        print("Ожидается: ", Y[i])
        nNet.predict(X[i])