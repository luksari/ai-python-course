import numpy as np
import matplotlib.pyplot as plt

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


f1 = np.random.rand(N) - 2
s1 = np.random.rand(N) + 3


f2 = np.random.rand(N) + 3
s2 = np.random.rand(N) - 3

class1 = np.column_stack((f1, s1))
class2 = np.column_stack((f2, s2))
data = np.vstack((class1, class2))

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(data[:, 0], data[:, 1])

plt.grid()

plt.show()

labels = np.array([classifier(data[i, :]) for i in range(2*N)])


c1 = (labels == 1).nonzero()
c2 = (labels == 2).nonzero()

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(data[:, 0], data[:, 1])

y1 = [g1(data[i, :]) for i in range(2*N)]
y2 = [g2(data[i, :]) for i in range(2*N)]

y1 = np.round(y1, 2)
y2 = np.round(y2, 2)

plt.plot(y1, -y2)

plt.grid()

plt.scatter(data[c1, 0], data[c1, 1], c='b')
plt.scatter(data[c2, 0], data[c2, 1], c='r')

plt.show()
