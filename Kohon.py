import numpy as np

def g(best, cur, sigma):
    return np.exp(np.linalg.norm(best - cur)**4/2*sigma**2)

import math
epoches = 10
samples = 4
neurons = 2
D = np.array([0.0] * neurons)
a = 1
sigma = 0.01
weights = np.random.random((2, 4))

x = np.array([[1, 1, 0, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 0, 1, 1]])

for epoche in range(epoches):
    for sample in range(samples):
        for neuron in range(neurons):
            D[neuron] = np.linalg.norm(x[sample] - weights[neuron])

        print(D)
        argBest = np.argmin(D)
        best = weights[argBest]
        weights[argBest] += a * (x[sample] - weights[argBest])
        # for neuron in range(neurons):
        #     weights[neuron] += a * g(best, weights[neuron], sigma) * (x[sample] - weights[neuron])
    a /= 2


x_test = np.array([[2, 1, 0, 0],
                  [0, 0, 1, 1],
                  [1, 1, 0, 0],
                  [0, 0, 2, 1]])
for sample in range(samples):
    for neuron in range(neurons):
        D[neuron] = np.linalg.norm(x[sample] - weights[neuron])
    print(np.argmin(D))