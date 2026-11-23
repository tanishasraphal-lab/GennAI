# Single Neuron Implementation

## Overview
This assignment demonstrates the fundamental working of an artificial neuron, the basic building block of an Artificial Neural Network (ANN).

The implementation uses Python and NumPy to calculate the weighted sum of input features, add a bias, and apply different activation functions to generate the neuron's output.

## Objectives
- Understand the mathematical working of an artificial neuron.
- Implement weights, inputs, and bias.
- Calculate the weighted sum using the dot product.
- Implement and compare different activation functions.
- Understand how activation functions influence neuron outputs.

## Concepts Covered

### 1. Inputs
The neuron receives three input features:

`X = [0.8, 0.4, 1.0]`

### 2. Weights
Each input is assigned a corresponding weight representing its contribution to the neuron:

`W = [0.5, -0.3, 0.9]`

### 3. Bias
A bias value is added to the weighted sum to shift the neuron's output.

`b = 0.1`

### 4. Weighted Sum
The weighted sum is calculated using:

\[
Z = W \cdot X + b
\]

For the given inputs:

\[
Z = 1.28
\]

### 5. Activation Functions
Three activation functions are implemented:

- **ReLU (Rectified Linear Unit):** Returns the input if positive, otherwise returns zero.
- **Sigmoid:** Maps values to a range between 0 and 1.
- **Tanh:** Maps values to a range between -1 and 1.

## Technologies Used
- Python 3
- NumPy

## Implementation
The Python program:
1. Defines input features, weights, and bias.
2. Calculates the weighted sum using `np.dot()`.
3. Applies the ReLU activation function.
4. Implements Sigmoid and Tanh activation functions.
5. Compares all three activation functions using different test values.

## Expected Output

```text
Weighted sum (Z): 1.2800
Neuron output = 1.2800

     z |  ReLU(z) | Sigmoid(z) |  Tanh(z)
---------------------------------------------
    -3 |   0.0000 |     0.0474 |  -0.9951
    -1 |   0.0000 |     0.2689 |  -0.7616
     0 |   0.0000 |     0.5000 |   0.0000
     1 |   1.0000 |     0.7311 |   0.7616
     3 |   3.0000 |     0.9526 |   0.9951
```

## How to Run

1. Clone or download the repository.
2. Install NumPy:

```bash
pip install numpy
```

3. Run the Python file:

```bash
python SingleNeuron.py
```

## Learning Outcome
This assignment provides an understanding of how an artificial neuron processes input data through weights, bias, and activation functions. It establishes the foundation for understanding multi-layer neural networks and deep learning architectures.

---

*Part of my hands-on learning journey in Generative AI.*