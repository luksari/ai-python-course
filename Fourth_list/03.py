from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
import math
import random

digits = datasets.load_digits()
X = digits.data
Y = digits.target
target_names = digits.target_names
train, test, train_targets, test_targets = train_test_split(X, Y, train_size=0.5, test_size=0.5)

max = 0
max_index = 0
for i in range(1, 10):
    flda = LDA(n_components=i)
    X_r = flda.fit(train, train_targets).transform(train)
    Y_r = flda.fit(test, test_targets).transform(test)
    clf = KNeighborsClassifier(round(math.sqrt(X.shape[0])),
    weights="uniform", metric="euclidean")
    clf.fit(X_r, train_targets)
    print(i, ":", clf.score(Y_r, test_targets))
    if max < clf.score(Y_r, test_targets):
        max = clf.score(Y_r, test_targets)
        max_index = i
print("Best: ", max_index)