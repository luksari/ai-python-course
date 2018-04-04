from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydot
import io
import graphviz


iris = load_iris()
X = iris.data
y = iris.target
train, test, train_targets, test_targets = train_test_split(
    X, y,       test_size=0.50)
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(X, y)
y = clf.predict(X)

score = (iris.target == y).sum()
print("Poprawnie zaklasyfikowanych. : ", score)
print("Sprawnosc: ", float(score) / len(y))

dot_data = io.StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_png("06.2.png")
