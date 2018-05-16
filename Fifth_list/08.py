import numpy
import neurolab
 
cahracters = ['A', 'T', 'V']
target = numpy.asfarray([
   [0,0,1,0,0,
    0,1,0,1,0,
    0,1,1,1,0,
    0,1,0,1,0,
    0,1,0,1,0],
 
    [1,1,1,1,1,
    0,0,1,0,0,
    0,0,1,0,0,
    0,0,1,0,0,
    0,0,1,0,0],
 
    [1,0,0,0,1,
    0,1,0,1,0,
    0,1,0,1,0,
    0,1,0,1,0,
    0,0,1,0,0]])
target[target == 0] = -1
 
net = neurolab.net.newhop(target)
for i in range(len(target)):
    print(cahracters[i], (net.sim(target)[i] == target[i]).all())
 
test = numpy.asfarray([0,0,1,0,0,
    0,1,0,1,0,
    0,1,1,1,0,
    0,1,0,1,0,
    0,1,0,1,0])
test[test==0] = -1
 
print((net.sim([test])[0] == target[0]).all(), 'steps',
 len(net.layers[0].outs))