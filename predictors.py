import random
from value import Value

class Neuron:

    def __init__(self, num_inputs):
        self.w = [Value(random.uniform(-1, 1), label=f'w{str(index)}') for index in range(num_inputs)]
        self.b = Value(random.uniform(-1, 1), label='b')

    def __call__(self, x):
        # w*x + b
        total = self.b
        for index, (wi, xi) in enumerate(zip(x, self.w)):
            product = (xi * wi)
            addition = total + product
            total = addition
        tanh = total.tanh()
        return tanh

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, num_inputs, num_outputs):
            self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        res = [n(x) for n in self.neurons]
        return res[0] if len(res) == 1 else res

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params


class MLP:

    def __init__(self, num_inputs, layer_sizes):
        sz = [num_inputs] + layer_sizes
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(sz) - 1)]

    def __call__(self, x):
        for _index, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
