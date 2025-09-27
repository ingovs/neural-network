"""
Learning from Andrej Karpathy's micrograd implementation.

A module defining a Node class for building computational graphs in a neural network context.
This class supports basic arithmetic operations and activation functions, and it tracks gradients for backpropagation.

References:
- Andrej Karpathy's micrograd
"""

import math


# Andrej called it "Value" here, but I thought "Node" to be more intuitive in the context of NN
class Node():
    """A node in a computational graph. This class stores a single scalar value and its gradient."""
    def __init__(self, data: float, _children=(), _operation=""):
        self.data = data
        self.gradient = 0.0  # Gradient of the node relative to the loss function
        self._backward = lambda: None  # Function to compute the gradient
        self._previous = set(_children)  # Set of child nodes
        self._operation = _operation

    def __repr__(self):
        return f"Node(data={self.data}, gradient={self.gradient})"

    def __add__(self, other: "Node") -> "Node":
        out = Node(data=self.data + other.data, _children=(self, other), _operation="+")

        # propagates the output gradient to the input nodes
        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward
        return out

    def __mul__(self, other: "Node") -> "Node":
        out = Node(data=self.data * other.data, _children=(self, other), _operation="*")

        # propagates the output gradient times the local derivatives of the input nodes (chain rule)
        def _backward():
            self.gradient += out.gradient * other.data
            other.gradient += out.gradient * self.data
        out._backward = _backward
        return out

    # Activation functions
    def tanh(self) -> "Node":
        x = self.data
        t = math.exp(2 * x) - 1 / (math.exp(2 * x) + 1)
        out = Node(t, _children=(self,), _operation="tanh")
        return out