import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import matplotlib.pyplot as plt


banana = sio.loadmat("banana.mat")
train_data = banana["train_data"]
train_labels = banana["train_labels"]
train_labels = np.array(train_labels)
test_data = banana["test_data"]
test_labels = banana["test_labels"]
test_labels = np.array(test_labels)

train, dummy, train_targets, dummy = train_test_split(train_data, train_labels.ravel(), test_size=0.70)

dummy, test, dummy, test_targets = train_test_split(test_data, test_labels.ravel(), test_size=0.70)

clf = NearestCentroid()
clf.fit(train, train_targets)
Z = clf.predict(test)

c1 = (Z == 1).nonzero()
c2 = (Z == 2).nonzero()
plt.scatter(test[c1, 0], test[c1, 1], c="g", label="Klasa 1")
plt.scatter(test[c2, 0], test[c2, 1], c="r", label="Klasa 2")
plt.legend()
plt.scatter(clf.centroids_[:, 0], clf.centroids_[:, 1], c="b")
plt.show()
