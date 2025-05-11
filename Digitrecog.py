import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

Y_train = data.iloc[:, 0]
Y_test = test_data.iloc[:, 0]

# Convert to NumPy arrays
Y = Y_train.to_numpy()
X = data.iloc[:, 1:].to_numpy().T  # Transpose for (784, 42000)
Y_t = Y_test.to_numpy()
X_t = test_data.iloc[:, 1:].to_numpy().T  # Transpose for (784, num_test_samples)

# Initialize parameters
num_samples = X.shape[1]  # 42000
n_input = 784
n_hidden = 10
n_output = 10

# Initialize weights and biases
W_1 = np.random.rand(n_hidden, n_input) * 0.01
B_1 = np.zeros((n_hidden, 1))  # Bias for hidden layer
W_2 = np.random.rand(n_output, n_hidden) * 0.01
B_2 = np.zeros((n_output, 1))  # Bias for output layer
def one_hot_encode(Y, num_classes):
    return np.eye(num_classes)[Y].T  # Convert to one-hot encoding

num_classes = 10
Y = one_hot_encode(Y, num_classes)


def calc_Z(W, A, B):
    return W @ A + B

def ReLu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return Z_exp / np.sum(Z_exp, axis=0, keepdims=True)

def forward_propagation(A_0, W_1, B_1, W_2, B_2):
    Z_1 = calc_Z(W_1, A_0, B_1)
    A_1 = ReLu(Z_1)
    Z_2 = calc_Z(W_2, A_1, B_2)
    A_2 = softmax(Z_2)
    return Z_1, A_1, Z_2, A_2

def backward_propagation(A_0, W_1, B_1, W_2, B_2, Z_1, A_1, Z_2, A_2, Y):
    dZ_2 = A_2 - Y
    dW_2 = 1 / A_0.shape[1] * dZ_2 @ A_1.T
    dB_2 = np.mean(dZ_2, axis=1, keepdims=True)

    dZ_1 = (W_2.T @ dZ_2) * (Z_1 > 0)  # dZ_1 is backpropagated
    dW_1 = 1 / A_0.shape[1] * (dZ_1 @ A_0.T)
    dB_1 = np.mean(dZ_1, axis=1, keepdims=True)

    return dW_1, dB_1, dW_2, dB_2

def adjust(W_1, B_1, W_2, B_2, dW_1, dB_1, dW_2, dB_2, learning_rate=0.1):
    W_1 -= learning_rate * dW_1
    B_1 -= learning_rate * dB_1
    W_2 -= learning_rate * dW_2
    B_2 -= learning_rate * dB_2

    return W_1, B_1, W_2, B_2

def prediction(A_2):
    return np.argmax(A_2, axis=0)

def accuracy(A_2, Y):
    A_out = prediction(A_2)
    correct = (A_out == Y)
    return np.mean(correct)

def train(A_0, W_1, B_1, W_2, B_2, Y, learning_rate):
    Z_1, A_1, Z_2, A_2 = forward_propagation(A_0, W_1, B_1, W_2, B_2)
    dW_1, dB_1, dW_2, dB_2 = backward_propagation(A_0, W_1, B_1, W_2, B_2, Z_1, A_1, Z_2, A_2, Y)
    W_1, B_1, W_2, B_2 = adjust(W_1, B_1, W_2, B_2, dW_1, dB_1, dW_2, dB_2, learning_rate)
    print(f'The training Accuracy of the Network is : {accuracy(A_2, Y)}')
    return W_1,B_1,W_2,B_2

# Training Loop
learning_rate = 0.1
num_epochs = 20

for epoch in range(num_epochs):
    W_1, B_1, W_2, B_2 = train(X, W_1, B_1, W_2, B_2, Y, learning_rate)
