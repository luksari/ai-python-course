from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target
 
train, test, train_targets, test_targets = train_test_split(X, y,
 test_size=0.50) 

print("Wielkość zbioru treningowego:",len(train))
print("Wielkość zbioru testowego:",len(test))
