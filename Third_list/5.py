from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree


iris = load_iris()
X = iris.data
y = iris.target
train, test, train_targets, test_targets = train_test_split(
    X, y,       test_size=0.50)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
y = clf.predict(X)

score = (iris.target == y).sum()
print("Poprawnie zaklasyfikowanych. : ", score)
print("Sprawnosc: ", float(score) / len(y))
