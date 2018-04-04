from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import math

iris = load_iris()
X = iris.data
y = iris.target
train, test, train_targets, test_targets = train_test_split(
    X, y,       test_size=0.50)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

k = 10
clfScore = clf.score(train, test_targets)
print("Sprawność klasyfikatora",clfScore * 100, "%")
print("Niepoprawnie zaklasyfikowanych: ",
      math.ceil(len(train) * (1 - clfScore)))
