import math
import numpy as np

# class neuron_network:
#     def __init__(self, input_size, hidden_size, output_size,learning_rate):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.learning_rate = learning_rate
#
#         self.W1 = np.random.randn(self.input_size, self.hidden_size).astype(np.float32);
#         self.b1 = np.random.randn(self.hidden_size).astype(np.float32);
#         self.W2 = np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32);
#         self.b2 = np.random.randn(self.hidden_size).astype(np.float32);
#
#     def activation(self,x):
#         return math.tanh(x);
#
#     def activation_derivative(x):
#         return 1-math.tan(x)**2;
#
#     def forward(self, X):
#
#     def compute_loss(self, y_pred, y_true):
#         return (y_pred - y_true) ** 2
#
#     def backward(self, X, y_true, y_pred):
#
#     def train(self,X, y, epochs, verbose=False):
#
#     def predict(self,X):
#
#     def save_model(self,filepath):
#
#     def load_model(self,filepath):
#         return False

class neuron_network:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicializace vah a prahů s náhodnými hodnotami
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def activation(self, x):
        return np.tanh(x)

    def activation_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        # Výstupní vrstva bez aktivační funkce (lineární regresní výstup)
        return self.Z2

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        # Gradient výstupní vrstvy
        dZ2 = (y_pred - y_true) / m
        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Gradient skryté vrstvy
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        # Aktualizace vah a prahů
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X, y, epochs, verbose=False):
        history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y_pred, y)
            history.append(loss)
            self.backward(X, y, y_pred)
            if verbose and (epoch % (epochs // 10) == 0):
                print(f"Epoch {epoch}/{epochs}, Loss: {loss}")
        return history

    def predict(self, X):
        return self.forward(X)

    def save_model(self, filepath):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_model(self, filepath):
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
