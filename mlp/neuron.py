from value import Value
import random

class Neuron:
    def __init__(self, xlen: int ) -> None:
        self.w = [Value(random.uniform(-1.0, 1.0), label="w{}".format(i)) for i in range(xlen)]
        self.b = Value(random.uniform(-1.0, 1.0), label='b')
        self.x = []

    def __call__(self, x: list) -> Value:
        """ This process the forward pass of the Neuron """

        for i in range(len(x)):
            if not isinstance(x[i], Value):
                x[i] = Value(x[i], label='x')

        self.x = x
        out = sum([x * w for x, w in zip(x, self.w)]) + self.b
        out = out.tanh()
        out.label = 'o'

        return out
