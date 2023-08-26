from layer import Layer
from value import Value

class MLP:

    def __init__(self, nin: int, layerinfo: list) -> None:
        """
            nin: number of inputs of the data
            layerninfo: list of values represent number of layers
                        and number of neurons witin a layer
        """
        lnin = [nin] + layerinfo
        self.layers = [Layer(lnin[i], layerinfo[i]) for i in range(len(layerinfo))]

    def __call__(self, x: list) -> list:

        for l in self.layers:
            x = l(x)
        
        if len(x) == 1:
            return x[0]
        else:
            return x

    def getParameters(self) -> list:

        params = []
        for l in self.layers:
            for n in l.neurons:
                params += n.w + n.x + [n.b]

        return params