import numpy as np

def nonlin(x, deriv=False):
    if deriv: return x * (1 - x)
    return 1 / (1 + np.exp(-x))

y = np.array([[0], [1], [1], [0]])
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# случайно инициализируем веса. Три слоя, Количесво нейронов: 3*4*1
np.random.seed(1)
weights01 = 2 * np.random.random((3, 4)) - 1
weights12 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):

    #проход вперёд, вычисление выходов слоёв
    l0 = X
    l1 = nonlin(np.dot(l0, weights01))
    l2 = nonlin(np.dot(l1, weights12))
    if j == 0:
        print(np.shape(l0), np.shape(l1), np.shape(l2))
        print((l0), (l1), (l2))

    #финальная ошибка
    l2_error = y - l2

    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin(l2, deriv=True)

    #изменение весов 1 слоя
    l1_error = l2_delta.dot(weights12.T)
    weights12 += l1.T.dot(l2_delta)

    #изменение весов 0 слоя
    l1_delta = l1_error * nonlin(l1, deriv=True)
    weights01 += l0.T.dot(l1_delta)


