import numpy as np
import math
class Value:
    """
        This class is used to store the numerical value and the gradient of the value
    """
    def __init__(self, data, parent=(), _op='', label=''):
        self.data = data
        # this will contain the children nodes
        self.parent = set(parent)
        # Record the operation
        self.op = _op
        self.label = label
        self.grad = 0.0
        # this method perfom the backpropgation 
        self._backward = lambda: None # <-- By default there is no function implemented
        
    def __repr__(self):
        return f"Value({self.label}, data={self.data}, grad={self.grad})"
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward_add():
            # In the addition operation the gradiant from the output get distributed to self and other
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward_add # <-- Define the backprop local method
        return out
    
    def __radd__(self, other):
        """ this method get call if __add__ method erroed out. 
        This method allow us to add number to valu object (ex: 1 + value(2.0) = value(3.0))
        """
        return self + other

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = self + (-1 * other)    
        return out
    
    def __rsub__(self, other):
        return -1 * (self - other)
    
    def __mul__(self, other):    
        # if te 'other' object is not a Value object, then convert it first 
        if not isinstance(other, Value):
            other = Value(other)
        
        out = Value(self.data * other.data, (self, other), '*')
        def _backward_mul():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        
        out._backward = _backward_mul # <-- Define the backprop in local method 
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        x = self.data 
        x = x**other
        out = Value(x, (self, ), f'**{other}')
        
        def _backward_pow():
            self.grad += other * (self.data ** (other-1)) * out.grad
        
        out._backward = _backward_pow
        
        return out
    
    def __truediv__(self, other):
        """ Implement the divistion """
        return self * other**-1 # <- this call the __pow__ method 
    
    def exp(self):
        out =  Value(math.exp(self.data), (self, ), 'exp')
        # need to add the _backprop method in exp
        def _backward_exp():
            self.grad = out.grad * math.exp(self.data)
        
        out._backward = _backward_exp
        return out
    
    def tanh(self):
        x = self.data
        # this is tanh(x)
        if isinstance(x, Value):
            x = x.data

        t = (math.exp(2 * x) - 1)/(math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')
    
        def _backward_tanh():
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward_tanh # <-- Define the backprop in local method
        return out
    
    def backpropogate(self):   
        topological_graph = []
        visited_nodes = set()
        
        # Generate the topological sorted list of all the nodes
        def build_topological_graph(v):
            if v not in visited_nodes:
                visited_nodes.add(v)
                for p in v.parent:
                    build_topological_graph(p)
                topological_graph.append(v)

        self.grad = 1.0
        build_topological_graph(self)
        
        # Reverse the topological order and execute the _backwared() method of each node in the reversed list
        for node in reversed(topological_graph):
            node._backward()