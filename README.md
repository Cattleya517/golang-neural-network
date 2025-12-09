# MNIST Neural Network from Scratch

A pure Go implementation of a neural network for handwritten digit recognition. No PyTorch, TensorFlow, or Gorgonia required - just Go and basic linear algebra.

This project implements backpropagation, gradient descent, and all core neural network operations from scratch using only the `gonum` matrix library. It's designed for learning and experimentation with the MNIST dataset.

<p align="center">
  <img src="demo/demo.gif" alt="DEMO" width="600" />
</p>

## Features

- Fully connected neural network with customizable architecture
- He initialization for weights
- ReLU activation for hidden layers
- Softmax output with cross-entropy loss
- Interactive CLI for model configuration
- GUI drawing board for testing with your own handwritten digits
- Model persistence (save/load as JSON)

## Requirements

- Go 1.21 or later
- MNIST dataset (included in `mnist_data/`)

## Installation

```bash
git clone https://github.com/Cattleya517/golang-neural-network.git
cd golang-neural-network
go mod tidy
```

## Usage

### Run the application

```bash
make run
```

Or without Make:

```bash
go run main.go
```

### Main Menu Options

1. **Train a new model** - Define your own network architecture and train from scratch
2. **Load existing model** - Load a pre-trained model and test with the drawing board
3. **Exit**

### Training a Model

When training, you will be prompted to configure:

- Number of hidden layers
- Number of nodes per hidden layer
- Learning rate
- Number of epochs

Recommended configuration for good accuracy (~96%), which is also the config in `models/basic.json` model:

```
Hidden Layers: 2
Layer 1 nodes: 128
Layer 2 nodes: 64
Learning Rate: 0.01
Epochs: 15
```

After training, the model will be saved to the `models/` directory.

### Testing with Drawing Board

After training or loading a model, a GUI window will open where you can:

1. Draw a digit (0-9) using your mouse
2. Click "Predict" to see the model's prediction
3. Click "Clear" to reset the canvas

## Project Structure

```
.
├── cmd/
│   └── root.go          # CLI interface and menu logic
├── nn/
│   ├── nn.go            # Neural network structure and forward pass
│   ├── train.go         # Training loop and backpropagation
│   ├── mnist.go         # MNIST data loading utilities
│   └── persist.go       # Model save/load functionality
├── drawing/
│   └── canvas.go        # GUI drawing board and image preprocessing
├── mnist_data/          # MNIST dataset files
├── models/              # Saved model files
├── main.go              # Entry point
├── Makefile
└── README.md
```

## How It Works

### Forward Pass

Input (784) -> Hidden Layers (ReLU) -> Output (10) -> Softmax

### Backpropagation

The implementation computes gradients for all layers before applying weight updates, following the standard backpropagation algorithm:

1. Compute output error (prediction - target)
2. Propagate error backwards through layers
3. Compute weight gradients using chain rule
4. Update weights using gradient descent

### Image Preprocessing

Hand-drawn digits are preprocessed to match MNIST format:

1. Find bounding box of the digit
2. Center and scale to 20x20 pixels
3. Place in 28x28 canvas with padding
4. Normalize pixel values to 0-1 range

## Pre-trained Model

A pre-trained model (`models/basic.json`) is included with the following configuration:

- Architecture: 784 -> 128 -> 64 -> 10
- Training: 10 epochs on full MNIST training set
- Validation accuracy: ~96%

## Limitations

- CPU only (no GPU acceleration)
- No batch processing (processes one sample at a time)
- Basic SGD optimizer (no momentum or Adam)

These limitations are intentional - the goal is clarity and learning, not production performance.

## License

MIT

## Acknowledgments

- MNIST dataset by Yann LeCun et al.
- GUI powered by [Fyne](https://fyne.io/)
- Matrix operations by [Gonum](https://www.gonum.org/)
