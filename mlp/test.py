import numpy as np
from value import Value
from neuron import Neuron
from layer import Layer
from mlp import MLP
import pprint
a = Value(3.0); a.label = 'a'
b = Value(2.0); b.label = 'b'

'''
x = [1.2, 3.2, 9.0]
n1 = Neuron(len(x))
out = n1(x)
print(out)
'''


'''
x = [1.2, 3.2, 9.0]
l1 = Layer(len(x), 4)
out = l1(x)
pprint.pprint(out)
'''

'''
x = [1.3, 4.3, 5.9, 4.9]

linfo = [4, 4, 1]
mlp1 = MLP(len(x), linfo)
y = mlp1(x) # <-- forward pass the input data
print(y)
y.backpropogate()
print(y)
parms = mlp1.paramters()
pprint.pprint(parms)
'''


# These are the inputs 
x1 = [
        [2.0, 3.0, -1.0], # case 1
        [3.0, -1.0, 0.5], # case 2
        [0.5, 1.0, 1.0],  # case 3
        [1.0, 1.0, -1.0]  # case 4
     ]

# this is what we want as the output
y1 = [
        1.0,  # case 1 should output this 
        -1.0, # case 2 should output this 
        -1.0, # case 3 should output this 
        1.0   # case 4 should output this 
     ]


l = [4, 4, 1]
mlp1 = MLP(3, l) 
h = 0.1

for index in range(50000):
    yout = [mlp1(case) for case in x1]
    loss = 0.0
    for i, j in zip(y1, yout):
        loss += (i - j)**2


    params = mlp1.getParameters()
    for p in params:
        p.grad = 0.0

    loss.backpropogate()
    params = mlp1.getParameters()
    for p in params:
        p.data += - p.grad * h    

    print(index, loss)

    


print(len(params))




pprint.pprint(y1)
pprint.pprint(yout)
