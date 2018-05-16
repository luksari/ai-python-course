from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
 
x_train, x_test, y_train, y_test = train_test_split(load_digits().data,
    load_digits().target, train_size=1000, test_size=500)
 
clf = MLPClassifier()
clf.fit(x_train, y_train)
print("Default parameters",clf.score(x_test, y_test))
 
clf = MLPClassifier(solver='lbfgs', alpha=0.5)
clf.fit(x_train, y_train).predict(x_test)
print("LBFGS Solver, Alpha = 0.5 parameters",clf.score(x_test, y_test))
 
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, random_state=1)
clf.fit(x_train, y_train).predict(x_test)
print("LBFGS Solver, Alpha = 0.00001, Rand_state = 1 parameters",clf.score(x_test, y_test))