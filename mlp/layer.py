from typing import Any
from neuron import Neuron

class Layer:
    """ This is a layer of neurons """

    def __init__(self, nin: int, nn: int) -> None:
        """
            nin: number of inputs to the neuron 
            nn : number of neurons in the layer
        """
        self.neurons = [Neuron(nin) for _ in range(nn)]

    def __call__(self, x: list) -> list:
        """
            x: input data as a list
        """
        out = []
        for n in self.neurons:
            out += [n(x)]

        return out
    

