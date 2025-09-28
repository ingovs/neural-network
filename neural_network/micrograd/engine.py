"""
Learning from Andrej Karpathy's Pytorch-like micrograd implementation.

A module defining a Node class for building computational graphs in a neural network context.
This class supports basic arithmetic operations and activation functions, and it tracks gradients for backpropagation.

References:
- Andrej Karpathy's micrograd
"""

import math


class Value():
    """A value in a computational graph. This class stores a single scalar value and its gradient."""
    def __init__(self, data: float, _children=(), _operation=""):
        self.data = data
        self.gradient = 0.0  # Gradient of the value relative to the loss function
        self._backward = lambda: None  # Function to compute the gradient
        self._previous = set(_children)  # Set of child values
        self._operation = _operation

    def __repr__(self):
        return f"Value(data={self.data}, gradient={self.gradient})"

    def __add__(self, other: "Value") -> "Value":
        # enables adding a Value to a scalar
        other = other if isinstance(other, Value) else Value(other)

        out = Value(data=self.data + other.data, _children=(self, other), _operation="+")

        # propagates the output gradient to the input values
        # NOTE: the += is important for cases where a value is used multiple times in the graph, so gradients accumulate
        def _backward():
            self.gradient += out.gradient
            other.gradient += out.gradient
        out._backward = _backward
        return out

    def __mul__(self, other: "Value") -> "Value":
        # enables multiplying a Value by a scalar
        other = other if isinstance(other, Value) else Value(other)

        out = Value(data=self.data * other.data, _children=(self, other), _operation="*")

        # propagates the output gradient times the local derivatives of the input values (chain rule)
        def _backward():
            self.gradient += out.gradient * other.data
            other.gradient += out.gradient * self.data
        out._backward = _backward
        return out

    def __pow__(self, power: float) -> "Value":
        assert isinstance(power, (int, float)), "Only supporting int/float powers for now"
        out = Value(data=self.data ** power, _children=(self,), _operation=f"**{power}")

        # propagates the output gradient times the local derivative of the input value (chain rule)
        def _backward():
            self.gradient += out.gradient * power * (self.data ** (power - 1))
        out._backward = _backward
        return out

    # Activation functions
    def tanh(self) -> "Value":
        x = self.data
        t = math.exp(2 * x) - 1 / (math.exp(2 * x) + 1)
        out = Value(t, _children=(self,), _operation="tanh")

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

        def build_topo(node: "Value"):
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
    # NOTE: these operations will handle backpropagation correctly, because they work
    # by delegating to the primary operations (__add__, __mul__, __pow__) that already
    # have proper gradient computation.
    def __neg__(self) -> "Value":
        return self * -1

    def __radd__(self, other: float) -> "Value":
        return self + other

    def __sub__(self, other: float) -> "Value":
        return self + (-other)

    def __rsub__(self, other: float) -> "Value":
        return other + (-self)

    def __rmul__(self, other: float) -> "Value":
        return self * other

    def __truediv__(self, other: float) -> "Value":
        return self * (other ** -1)

    def __rtruediv__(self, other: float) -> "Value":
        return other * (self ** -1)
