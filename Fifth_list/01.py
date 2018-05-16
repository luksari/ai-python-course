import neurolab
import numpy
import pylab
 
x = numpy.linspace(0, 6, 20)
size = len(x)
y = numpy.sin(x)
 
inp = x.reshape(size,1)
net = neurolab.net.newff([[0, 6]],[5, 1])
net.trainf = neurolab.train.train_gd
error = net.train(inp, y.reshape(size, 1), epochs=500, show=100, goal=0.02)
 
x2 = numpy.linspace(0,6,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pylab.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pylab.legend(['real value', 'learning score'])
pylab.show()
 
# ---
 
x = numpy.linspace(1, 2.5, 20)
size = len(x)
y = numpy.log(x) * 0.5
 
inp = x.reshape(size,1)
net = neurolab.net.newff([[1, 2.5]],[5, 1])
net.trainf = neurolab.train.train_gd
error = net.train(inp, y.reshape(size, 1), epochs=500, show=100, goal=0.02)
 
x2 = numpy.linspace(1,2.5,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pylab.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pylab.legend(['real value', 'learning score'])
pylab.show()
 
# ---
 
x = numpy.linspace(1, 6, 20)
size = len(x)
y = numpy.cos(x) * x + numpy.log(x) * 0.3
 
inp = x.reshape(size,1)
net = neurolab.net.newff([[1, 6]],[5, 1])
net.trainf = neurolab.train.train_gd
error = net.train(inp, y.reshape(size, 1), epochs=500, show=100, goal=0.02)
 
x2 = numpy.linspace(1,6,150)
y2 = net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = net.sim(inp).reshape(size)
 
pylab.plot(x2, y2, '-',x , y, '.', x, y3, 'p')
pylab.legend(['real value', 'learning score'])
pylab.show()