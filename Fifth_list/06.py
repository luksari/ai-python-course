import neurolab
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn import svm
 
data, target = sklearn.datasets.load_diabetes(True)
 
x_train, x_test, y_train, y_test = train_test_split(data,
    target, test_size=0.3, random_state=42)
 
tar = y_train.reshape(len(x_train), 1)
 
errorP = neurolab.net.newp([[-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2],
    [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], 1).train(x_train, tar)
 
errorMLP = neurolab.net.newff([[-2, 2],[-2, 2], [-2, 2],[-2, 2], [-2, 2],
    [-2, 2], [-2, 2], [-2, 2], [-2, 2], [-2, 2]], [3, 1]).train(x_train,
     tar)
 
print(svm.SVC().fit(x_train, tar.ravel()).predict(x_test))