import neurolab
import scipy.io
 
d1 = scipy.io.loadmat('perceptron1.mat')
d2 = scipy.io.loadmat('perceptron1.mat')
 
net = neurolab.net.newp([[-1, 5],[0, 2]], 1)
error = net.train(d1['data'], d1['labels'], epochs=8, show=1, lr=0.1)
error2 = net.train(d2['data'], d2['labels'], epochs=8, show=1, lr=0.1)