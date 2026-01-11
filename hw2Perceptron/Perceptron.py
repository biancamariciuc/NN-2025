import numpy as np
import math

class Perceptron:
    def __init__(self, n_features, lr=0.01, n_epochs=100):
        self.n_epochs = n_epochs
        self.lr = lr
        self.weights = np.zeros(n_features)
        self.bias = 0.0

    def activation_function(self, z):
        z = z - np.max(z, axis=1, keepdims=True)  # numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        z = X @ W + b
        return activation_function(z)

    def train(self, X, y):
        batch_size = 32
        for epoch in range(self.n_epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X, y = X[indices], y[indices]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                y_pred = self.forward(X_batch)
                dz = y_pred - y_batch
                dw = np.dot(X_batch.T, dz) / batch_size
                db = np.sum(dz) / batch_size
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    # def train(self, X, y):
    #     for epoch in range(self.n_epochs):
    #         y_pred = self.forward(X)
    #
    #         dz = y_pred - y
    #         dw = np.dot(X.T, dz) / self.n_samples
    #         db = np.sum(dz) / self.n_samples
    #
    #         self.weights -= self.lr * dw
    #         self.bias -= self.lr * db

    def predict(self, X):
        probs = self.forward(X)
        return (probs >= 0.5).astype(int)
