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
        # enables adding a Node to a scalar
        other = other if isinstance(other, Node) else Node(other)

        out = Node(data=self.data + other.data, _children=(self, other), _operation="+")

        # propagates the output gradient to the input nodes
        # NOTE: the += is important for cases where a node is used multiple times in the graph, so gradients accumulate
        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward
        return out

    def __mul__(self, other: "Node") -> "Node":
        # enables multiplying a Node by a scalar
        other = other if isinstance(other, Node) else Node(other)

        out = Node(data=self.data * other.data, _children=(self, other), _operation="*")

        # propagates the output gradient times the local derivatives of the input nodes (chain rule)
        def _backward():
            self.gradient += out.gradient * other.data
            other.gradient += out.gradient * self.data
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Node":
        assert isinstance(power, (int, float)), "Only supporting int/float powers for now"
        out = Node(data=self.data ** power, _children=(self,), _operation=f"**{power}")

        # propagates the output gradient times the local derivative of the input node (chain rule)
        def _backward():
            self.gradient += out.gradient * power * (self.data ** (power - 1))
        out._backward = _backward
        return out

    # Activation functions
    def tanh(self) -> "Node":
        x = self.data
        t = math.exp(2 * x) - 1 / (math.exp(2 * x) + 1)
        out = Node(t, _children=(self,), _operation="tanh")

        # 1 - tanh^2(x) is the local derivative of tanh
        def _backward():
            self.gradient += (1 - t ** 2) * out.gradient
        out._backward = _backward
        return out

    # Backpropagation
    def backward(self):
        # Topological order of nodes in the graph
        topo = []
        visited = set()

        def build_topo(node: "Node"):
            if node not in visited:
                visited.add(node)
                for child in node._previous:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.gradient = 1.0  # Seed the gradient of the output node
        for node in reversed(topo):
            node._backward()

    # Additional operations for convenience (right operations)
    def __neg__(self) -> "Node":
        return self * -1

    def __radd__(self, other: float) -> "Node":
        return self + other

    def __sub__(self, other: float) -> "Node":
        return self - other

    def __rsub__(self, other: float) -> "Node":
        return other - self

    def __rmul__(self, other: float) -> "Node":
        return self * other

    def __truediv__(self, other: float) -> "Node":
        return self * (other ** -1)

    def __rtruediv__(self, other: float) -> "Node":
        return other * (self ** -1)
