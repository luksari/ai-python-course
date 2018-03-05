import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

N = 5


def classifier(x):
    if x[0] >= 0 and x[1] >= 0:
        return 1
    elif x[0] < 0 and x[1] > 0:
        return 2
    elif x[0] <= 0 and x[1] <= 0:
        return 3
    elif x[0] > 0 and x[1] <0:
        return 4


first_attr = np.random.normal(2, 1, 2*N)
second_attr = 10*np.random.rand(2*N)

# first_attr = np.random.rand(N, 2)
# second_attr = np.random.rand(N, 2)-1

data = np.vstack((first_attr, second_attr))
data = data.conj().transpose()

min_x, min_y, max_x, max_y, offset = min(data[:, 0]), min(
    data[:, 1]), max(data[:, 0]), max(data[:, 1]), 1

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(data[:,0],data[:,1])

left, right = ax.get_xlim()
low, high = ax.get_ylim()


plt.plot([-5, 10], [0, 0], "black")
plt.plot([0, 0], [-5, 10], "black")

plt.grid();

plt.show()

labels=np.array([classifier(data[i,:]) for i in range(2*N)])

c1 = (labels==1).nonzero()
c2 = (labels==2).nonzero()
c3 = (labels==3).nonzero()
c4 = (labels==4).nonzero()

fig = plt.figure()
ax = fig.add_subplot(111)

plt.scatter(data[:, 0], data[:, 1])

left, right = ax.get_xlim()
low, high = ax.get_ylim()

plt.plot([-5, 10], [0, 0], "black")
plt.plot([0, 0], [-5, 10], "black")

plt.grid()

plt.scatter(data[c1,0],data[c1,1],c='b')
plt.scatter(data[c2,0],data[c2,1],c='r')
plt.scatter(data[c3,0],data[c3,1],c='g')
plt.scatter(data[c4,0],data[c4,1],c='y')

plt.show()
