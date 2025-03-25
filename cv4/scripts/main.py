from cv4.scripts.neuron_network import neuron_network
import numpy as np

a = neuron_network(3,3,3,1);
X = np.array([1,0,-1])
y = 1
apoch = 2
a.train(X,y,apoch)
