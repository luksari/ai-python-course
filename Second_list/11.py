import scipy.io as sio
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import math

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

clf = KNeighborsClassifier(bestK, weights='uniform',
metric='euclidean')
clf.fit(train, train_targets)
clfScore = clf.score(test, test_targets)
print("Procent poprawnych klasyfikacji:", round(clf.score(test, test_targets) * 100, 2),"%")
print("Źle zakwalifikowanych: ", math.floor(len(test_data) * (1 - clfScore)))
