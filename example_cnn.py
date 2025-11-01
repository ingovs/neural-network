"""
Example of creating and using a CNN with various layers.
This demonstrates how to construct a simple CNN for image classification.
"""

from neural_network.micrograd.engine import Value
from neural_network.micrograd.nn import CNN, Conv2D, MaxPooling2D, Flatten, Layer


def create_simple_cnn():
    """
    Creates a simple CNN architecture:
    - Conv2D layer: 1 input channel -> 8 output channels, 3x3 kernel
    - MaxPooling2D: 2x2 pooling
    - Conv2D layer: 8 -> 16 output channels, 3x3 kernel
    - MaxPooling2D: 2x2 pooling
    - Flatten: converts 3D to 1D
    - Fully connected layer: flattened size -> 10 outputs (e.g., for 10 classes)
    """
    layers = [
        Conv2D(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
        MaxPooling2D(kernel_size=2, stride=2),
        Conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
        MaxPooling2D(kernel_size=2, stride=2),
        Flatten(),
        Layer(num_inputs=16 * 7 * 7, num_neurons=10)  # Assuming 28x28 input -> 7x7 after pooling
    ]

    return CNN(layers)


def create_deep_cnn():
    """
    Creates a deeper CNN architecture with more convolutional layers.
    """
    layers = [
        # First convolutional block
        Conv2D(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
        MaxPooling2D(kernel_size=2, stride=2),

        # Second convolutional block
        Conv2D(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        MaxPooling2D(kernel_size=2, stride=2),

        # Third convolutional block
        Conv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        MaxPooling2D(kernel_size=2, stride=2),

        # Flatten and fully connected layers
        Flatten(),
        Layer(num_inputs=128 * 4 * 4, num_neurons=256),
        Layer(num_inputs=256, num_neurons=10)
    ]

    return CNN(layers)


def create_minimal_cnn():
    """
    Creates a minimal CNN with just one conv layer and a fully connected layer.
    """
    layers = [
        Conv2D(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
        Flatten(),
        Layer(num_inputs=4 * 24 * 24, num_neurons=10)  # For 28x28 input -> 24x24 after conv
    ]

    return CNN(layers)


def example_forward_pass():
    """
    Demonstrates a forward pass through a simple CNN.
    """
    # Create a simple CNN
    cnn = create_simple_cnn()

    # Create a dummy 28x28 grayscale image (1 channel)
    # Shape: (channels=1, height=28, width=28)
    image = [
        [[Value(0.5) for _ in range(28)] for _ in range(28)]
    ]

    # Forward pass
    output = cnn(image)

    print("Input shape: 1 channel, 28x28 pixels")
    print(f"Output shape: {len(output)} neurons")
    print(f"Output values: {[o.data for o in output]}")

    # Get number of parameters
    params = cnn.parameters()
    print(f"\nTotal parameters in the network: {len(params)}")

    return cnn, output


def example_training_step():
    """
    Demonstrates a training step with gradient descent.
    """
    # Create CNN
    cnn = create_minimal_cnn()

    # Create dummy input and target
    image = [[[Value(0.5) for _ in range(28)] for _ in range(28)]]
    target = [Value(0.0) if i != 5 else Value(1.0) for i in range(10)]  # Target class: 5

    # Forward pass
    output = cnn(image)

    # Compute loss (simple MSE)
    loss = sum((out - tgt)**2 for out, tgt in zip(output, target))

    print(f"Loss before training: {loss.data}")

    # Backward pass
    loss.backward()

    # Gradient descent step
    cnn.gradient_descent_step(learning_rate=0.01)

    # Forward pass again to see improvement
    output_after = cnn(image)
    loss_after = sum((out - tgt)**2 for out, tgt in zip(output_after, target))

    print(f"Loss after one training step: {loss_after.data}")


if __name__ == "__main__":
    print("=" * 60)
    print("Simple CNN Example")
    print("=" * 60)
    example_forward_pass()

    print("\n" + "=" * 60)
    print("Training Step Example")
    print("=" * 60)
    example_training_step()

    print("\n" + "=" * 60)
    print("CNN Architectures")
    print("=" * 60)

    simple = create_simple_cnn()
    print(f"\nSimple CNN layers: {len(simple.layers)}")
    print(f"Simple CNN parameters: {len(simple.parameters())}")

    deep = create_deep_cnn()
    print(f"\nDeep CNN layers: {len(deep.layers)}")
    print(f"Deep CNN parameters: {len(deep.parameters())}")

    minimal = create_minimal_cnn()
    print(f"\nMinimal CNN layers: {len(minimal.layers)}")
    print(f"Minimal CNN parameters: {len(minimal.parameters())}")
