import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
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

scores = []
for k in range(1, N):
    clf = KNeighborsClassifier(k, weights='uniform', metric='euclidean')
    clf.fit(train, train_targets)
    tempScore = clf.score(test, test_targets)
    scores.append(tempScore)

bestScore = max(scores)
bestK = scores.index(max(scores))

C = 1.0
h = .02
x_min, x_max = test[:, 0].min() - 1, test[:, 0].max() + 1
y_min, y_max = test[:, 1].min() - 1, test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
rgb_lighten = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
rgb = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
Z = KNeighborsClassifier(bestK, weights='uniform', metric='euclidean').fit(train, train_targets).predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, Z, cmap=rgb_lighten)
plt.scatter(test[:, 0], test[:, 1], c=test_targets, cmap=rgb)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()
