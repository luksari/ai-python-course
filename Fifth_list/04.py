import scipy.io
from sklearn.neural_network import MLPClassifier
 
data = scipy.io.loadmat('mnist_012.mat')
 
x_train = data['train_images']
y_train = data['train_labels']
x_test = data['test_images']
y_test = data['test_labels']
 
ny, nx, nsamples = x_train.shape
x_train = x_train.reshape((nsamples, nx*ny))
 
ny, nx, nsamples = x_test.shape
x_test = x_test.reshape((nsamples, nx*ny))
 
c1 = MLPClassifier(hidden_layer_sizes=(50, 5)).fit(x_train,
 y_train.ravel(nsamples))
c2 = MLPClassifier(hidden_layer_sizes=(100, 80)).fit(x_train,
 y_train.ravel(nsamples))
c3 = MLPClassifier(hidden_layer_sizes=(5, 2)).fit(x_train,
 y_train.ravel(nsamples))
 
print("Score for Hidden layer size (50,5):",c1.score(x_test, y_test))
print("Score for Hidden layer size (100,80):",c2.score(x_test, y_test))
print("Score for Hidden layer size (50,25):",c3.score(x_test, y_test))