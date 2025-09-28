import random
from typing import List

from .engine import Value


class Neuron():
    """Single neuron in a neural network layer."""
    def __init__(self, num_inputs: int):
        # w = weights, b = bias
        self.w = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.b = Value(0.0)

    def __call__(self, x: List[Value]) -> Value:
        # x = inputs, w = weights, b = bias, act = activation
        # Weighted sum of inputs plus bias (x * w + b)
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = act.tanh()  # Activation function
        return out

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer():
    """Layer of neurons in a neural network."""
    def __init__(self, num_inputs: int, num_neurons: int):
        # num_neurons = number of neurons in the layer
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]

    def __call__(self, x: List[Value]) -> List[Value]:
        # x = inputs, out = outputs
        out = [n(x) for n in self.neurons]
        return out

    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP():
    """Multi-layer perceptron (MLP) neural network."""
    def __init__(self, num_inputs: int, layer_sizes: List[int]):
        # layers = list of layers in the MLP (each element is number of neurons in that layer)
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(num_inputs=sizes[i], num_neurons=sizes[i+1]) for i in range(len(layer_sizes))]

    def __call__(self, x: List[Value]) -> List[Value]:
        # x = inputs, out = outputs
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def gradiend_descent_step(self, learning_rate: float = 0.01):
        for p in self.parameters():
            p.data -= learning_rate * p.gradient
            p.gradient = 0.0  # Reset gradient after update
