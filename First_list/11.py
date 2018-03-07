import numpy as np

N = 10

def classifier(x):
    if g1(x) > g2(x):
        return 2
    else:
        return 1

def g1(x):
    return -x[0] + x[1]


def g2(x):
    return x[0] - x[1]


class1 = np.random.rand(N, 2)
class2 = np.random.rand(N, 2)-1

data = np.vstack((class1, class2))

y1 = [g1(data[i, :]) for i in range(2*N)]
y2 = [g2(data[i, :]) for i in range(2*N)]

y1 = np.round(y1, 2)
y2 = np.round(y2, 2)
print('Wartości funkcji klasyfikujacych')
print(y1)
print(y2)

labels = np.array([classifier(data[i, :]) for i in range(2*N)])
print('Decyzje klasyfikatora:')
print(labels)
