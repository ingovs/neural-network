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
        Layer(
            num_inputs=16 * 7 * 7, num_neurons=10, activation=False
        ),  # No activation for output
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
        Layer(num_inputs=256, num_neurons=10),
    ]

    return CNN(layers)


def create_minimal_cnn():
    """
    Creates a minimal CNN with just one conv layer and a fully connected layer.
    """
    layers = [
        Conv2D(in_channels=1, out_channels=4, kernel_size=5, stride=1, padding=0),
        Flatten(),
        Layer(
            num_inputs=4 * 24 * 24, num_neurons=10, activation=False
        ),  # No activation in output layer
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
    image = [[[Value(0.5) for _ in range(28)] for _ in range(28)]]

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
    Demonstrates a training step with gradient descent and backward propagation.
    """
    # Create CNN
    cnn = create_minimal_cnn()

    # Create dummy input and target
    image = [[[Value(0.5) for _ in range(28)] for _ in range(28)]]
    target = [
        Value(0.0) if i != 5 else Value(1.0) for i in range(10)
    ]  # Target class: 5

    # 1. Forward pass
    print("Step 1: Forward pass through CNN")
    output = cnn(image)
    print(f"  Output: {[f'{o.data:.4f}' for o in output]}")

    # 2. Compute loss (simple MSE)
    print("\nStep 2: Compute loss")
    loss = sum((out - tgt) ** 2 for out, tgt in zip(output, target))
    print(f"  Loss: {loss.data:.6f}")

    # Check gradients before backward pass
    print("\nStep 3: Check gradients BEFORE backward propagation")
    grad_info = cnn.check_gradients()
    print(
        f"  Parameters with non-zero gradients: {grad_info['params_with_gradients']}/{grad_info['total_params']}"
    )
    print(f"  Max gradient magnitude: {grad_info['max_gradient']:.6f}")

    # 3. Backward pass
    print("\nStep 4: BACKWARD PROPAGATION (calling loss.backward())")
    loss.backward()  # NOTE: loss is a Value object containing the backward method
    print("  ✓ Gradients computed via automatic differentiation!")

    # Check gradients after backward pass
    print("\nStep 5: Check gradients AFTER backward propagation")
    grad_info = cnn.check_gradients()
    print(
        f"  Parameters with non-zero gradients: {grad_info['params_with_gradients']}/{grad_info['total_params']}"
    )
    print(f"  Max gradient magnitude: {grad_info['max_gradient']:.6f}")

    # Show some sample gradients
    params = cnn.parameters()
    print("\n  Sample parameter gradients:")
    for i in [0, 10, 100, 1000]:
        if i < len(params):
            print(
                f"    param[{i}]: value={params[i].data:.4f}, gradient={params[i].gradient:.6f}"
            )

    # 4. Gradient descent step
    print("\nStep 6: Update parameters (gradient descent)")
    print(f"  Before: param[0] = {params[0].data:.6f}")
    cnn.gradient_descent_step(learning_rate=0.01)
    print(f"  After:  param[0] = {params[0].data:.6f}")

    # Check gradients were reset
    grad_info = cnn.check_gradients()
    print(
        f"  Gradients reset: {grad_info['params_with_gradients']} params with non-zero gradients"
    )

    # 5. Forward pass again to see improvement
    print("\nStep 7: Forward pass after training step")
    output_after = cnn(image)
    loss_after = sum((out - tgt) ** 2 for out, tgt in zip(output_after, target))
    print(f"  New loss: {loss_after.data:.6f}")
    print(f"  Loss change: {loss_after.data - loss.data:.6f}")


def example_training_loop():
    """
    Demonstrates a complete training loop with multiple iterations.
    Shows how backward propagation is called in each iteration.
    """
    print("Training a minimal CNN for 10 iterations...")

    # Create CNN
    cnn = create_minimal_cnn()

    # Create training data
    learning_rate = 0.0001  # small learning rate for stability

    # Create training data ONCE outside the loop
    # We reuse the same input because we're training on one sample
    image = [[[Value(0.5) for _ in range(28)] for _ in range(28)]]
    target = [
        Value(0.0) if i != 5 else Value(1.0) for i in range(10)
    ]  # Target class: 5

    for iteration in range(30):
        # Forward pass
        output = cnn(image)

        # Compute loss
        loss = sum((out - tgt) ** 2 for out, tgt in zip(output, target))

        # Backward pass - compute gradients
        cnn.zero_grad()  # Reset gradients from previous iteration
        loss.backward()  # BACKWARD PROPAGATION - computes gradients

        # Check gradients BEFORE updating (they get reset after update)
        if iteration % 1 == 0:
            grad_info = cnn.check_gradients()
            print(
                f"Iteration {iteration + 1}: loss = {loss.data:.6f}, "
                f"max_gradient = {grad_info['max_gradient']:.6f}"
            )

        # Update parameters (this also resets gradients)
        cnn.gradient_descent_step(learning_rate=learning_rate)

    print("\nTraining complete!")
    print(f"Final output: {[f'{o.data:.4f}' for o in output]}")
    print(f"Target:       {[f'{t.data:.4f}' for t in target]}")


if __name__ == "__main__":
    # print("=" * 60)
    # print("Simple CNN Example")
    # print("=" * 60)
    # example_forward_pass()

    # print("\n" + "=" * 60)
    # print("Training Step Example (Detailed)")
    # print("=" * 60)
    # example_training_step()

    print("\n" + "=" * 60)
    print("Training Loop Example")
    print("=" * 60)
    example_training_loop()

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
