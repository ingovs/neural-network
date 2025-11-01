import random
from typing import List, Union

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


class Conv2D():
    """2D convolutional layer."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize kernels (weights) and biases
        self.kernels = [
            [[[Value(random.uniform(-1, 1)) for _ in range(kernel_size)] for _ in range(kernel_size)] for _ in range(in_channels)]
            for _ in range(out_channels)
        ]
        self.biases = [Value(0.0) for _ in range(out_channels)]

    def __call__(self, x: List[List[List[Value]]]) -> List[List[List[Value]]]:
        # x is a 3D list of Value objects with shape (in_channels, height, width)
        in_channels, in_height, in_width = len(x), len(x[0]), len(x[0][0])
        assert in_channels == self.in_channels, "Input channels must match layer's in_channels"

        # Apply padding (zero on borders)
        if self.padding > 0:
            padded_x = [
                [[Value(0.0) for _ in range(in_width + 2 * self.padding)] for _ in range(in_height + 2 * self.padding)]
                for _ in range(in_channels)
            ]

            # # to visualize the input before and after the padding
            # matrix_numbers = [
            #     [
            #         [padded_x[i][j][k].data for k in range(len(padded_x[i][j]))]
            #         for j in range(len(padded_x[i]))
            #     ]
            #     for i in range(len(padded_x))
            # ]

            for c in range(in_channels):
                for h in range(in_height):
                    for w in range(in_width):
                        padded_x[c][h + self.padding][w + self.padding] = x[c][h][w]
            x = padded_x
            in_height += 2 * self.padding
            in_width += 2 * self.padding

        # Calculate output dimensions
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        # Perform convolution
        output = [[[Value(0.0) for _ in range(out_width)] for _ in range(out_height)] for _ in range(self.out_channels)]
        for oc in range(self.out_channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    h_start = oh * self.stride
                    w_start = ow * self.stride

                    conv_sum = Value(0.0)
                    for ic in range(self.in_channels):
                        for kh in range(self.kernel_size):
                            for kw in range(self.kernel_size):
                                conv_sum += self.kernels[oc][ic][kh][kw] * x[ic][h_start + kh][w_start + kw]

                    output[oc][oh][ow] = conv_sum + self.biases[oc]
        return output

    def parameters(self) -> List[Value]:
        params = []
        for kernel_out in self.kernels:
            for kernel_in in kernel_out:
                for row in kernel_in:
                    params.extend(row)
        params.extend(self.biases)
        return params


class Flatten():
    """Flattens a 3D input to a 1D output."""
    def __call__(self, x: List[List[List[Value]]]) -> List[Value]:
        # x is a 3D list of Value objects with shape (channels, height, width)
        return [item for sublist1 in x for sublist2 in sublist1 for item in sublist2]

    def parameters(self) -> List[Value]:
        return []


class MaxPooling2D():
    """2D max pooling layer."""
    def __init__(self, kernel_size: int, stride: int):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x: List[List[List[Value]]]) -> List[List[List[Value]]]:
        # x is a 3D list of Value objects with shape (channels, height, width)
        channels, in_height, in_width = len(x), len(x[0]), len(x[0][0])

        # Calculate output dimensions
        out_height = (in_height - self.kernel_size) // self.stride + 1
        out_width = (in_width - self.kernel_size) // self.stride + 1

        # Perform max pooling
        output = [[[Value(0.0) for _ in range(out_width)] for _ in range(out_height)] for _ in range(channels)]
        for c in range(channels):
            for oh in range(out_height):
                for ow in range(out_width):
                    h_start = oh * self.stride
                    w_start = ow * self.stride

                    # Find the maximum value in the pooling window (this is the kernel[kernel_size x kernel_size] loop)
                    max_val = x[c][h_start][w_start]
                    for kh in range(self.kernel_size):
                        for kw in range(self.kernel_size):
                            current_val = x[c][h_start + kh][w_start + kw]
                            if current_val.data > max_val.data:
                                max_val = current_val
                    output[c][oh][ow] = max_val
        return output

    def parameters(self) -> List[Value]:
        return []


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

    def gradient_descent_step(self, learning_rate: float = 0.01):
        for p in self.parameters():
            p.data -= learning_rate * p.gradient
            p.gradient = 0.0  # Reset gradient after update


class CNN():
    """Convolutional Neural Network (CNN)."""
    def __init__(self, layers: List[Union[Conv2D, MaxPooling2D, Flatten, Layer]]):
        self.layers = layers

    def __call__(self, x: List[List[List[Value]]]) -> List[Value]:
        for layer in self.layers:
            # a layer output x is the input to the next layer
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def gradient_descent_step(self, learning_rate: float = 0.01):
        for p in self.parameters():
            p.data -= learning_rate * p.gradient
            p.gradient = 0.0
