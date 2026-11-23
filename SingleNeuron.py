import numpy as np

# Inputs (3 features)
X = np.array([0.8,0.4,1.0])

# Weights - how much each input matter?
W = np.array([0.5,-0.3,0.9])

# Bias - shifts the output up or down
b = 0.1

# Weighted sum (dot product) + bias
Z = np.dot(W,X) + b
print(f'Weighted sum (Z): {Z:.4f}')

# Activation function - ReLu: pass through if positive, else 0
def relu(z):
    return max(0,z)

output = relu(Z)
print(f'Neuron output = {output:.4f}')

def sigmoid(z):
    return 1/(1+np.exp(-z))

def tanh(z):
    return np.tanh(z)

test_values = [-3,-1,0,1,3]

print(f"{'z':>6} | {'ReLU(z)':>8} | {'Sigmoid(z)':>8} | {'Tanh(z)':>8}")
print("-"*40)
for z in test_values:
    print(f"{z:>6} | {relu(z):>8.4f} | {sigmoid(z):>8.4f} | {tanh(z):>8.4f}")