import scipy.io
import neurolab
 
data = scipy.io.loadmat('banana.mat')
 
train_data = data['train_data']
train_labels = data['train_labels']
 
perceptron = neurolab.net.newp([[-2, 2],[-2, 2]], 1)
 
errorP = perceptron.train(train_data, train_labels)
out = perceptron.sim(train_data)