import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

banana = sio.loadmat("banana.mat")
train_data = banana["train_data"]
train_labels = banana["train_labels"]
train_labels = np.array(train_labels)
test_data = banana["test_data"]
test_labels = banana["test_labels"]
test_labels = np.array(test_labels)

train, dummy, train_targets, dummy = train_test_split(train_data, train_labels.ravel(), test_size=0.70)

dummy, test, dummy, test_targets = train_test_split(test_data, test_labels.ravel(), test_size=0.70)

gnb = GaussianNB()
tmp = gnb.fit(train, train_targets)
Z = tmp.predict(test)

c1 = (Z == 1).nonzero()
c2 = (Z == 2).nonzero()


C = 1.0
h = .02
x_min, x_max = test[:, 0].min() - 1, test[:, 0].max() + 1
y_min, y_max = test[:, 1].min() - 1, test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = tmp.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
rgb_lighten = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
plt.pcolormesh(xx, yy, Z, cmap=rgb_lighten)
plt.scatter(test[c1, 0], test[c1, 1], c="b", label="Grupa 1")
plt.scatter(test[c2, 0], test[c2, 1], c="r", label="Grupa 2")
plt.legend()

plt.show()


