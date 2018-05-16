import neurolab
import numpy
 
x = numpy.linspace(1, 2.5, 20)
size = len(x)
y = numpy.log(x) * 0.5
 
n1 = neurolab.net.newff([[0, 2.5]],[5, 1])
n2 = neurolab.net.newff([[-5, 2.5]],[5, 1])
n3 = neurolab.net.newff([[2, 2.5]],[5, 1])
 
n1.trainf = neurolab.train.train_gd
n2.trainf = neurolab.train.train_gd
n3.trainf = neurolab.train.train_gd
 
inp = x.reshape(size, 1)
tar = y.reshape(size, 1)
error1 = n1.train(inp, tar, epochs=500, show=100, goal=0.02)
error2 = n2.train(inp, tar, epochs=500, show=100, goal=0.02)
error3 = n3.train(inp, tar, epochs=500, show=100, goal=0.02)