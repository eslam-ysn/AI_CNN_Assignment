import numpy as np

class NeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        A simple 1-hidden-layer neural network.

        input_size:  D_in  (e.g. num_kernels * 17 * 17)
        hidden_size: H     (e.g. 64)
        output_size: D_out (number of classes)
        """
        # Initialize weights and biases
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01  # (H x D_in)
        self.b1 = np.zeros((hidden_size, 1))                        # (H x 1)
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01 # (D_out x H)
        self.b2 = np.zeros((output_size, 1))                        # (D_out x 1)

    def relu(self, Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)

    def relu_derivative(self, Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)

    def softmax(self, Z: np.ndarray) -> np.ndarray:
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward(self, x: np.ndarray):
        """
        Forward pass.
        x: input vector of shape (D_in, 1)
        returns: A2 (output probabilities), cache
        """
        Z1 = self.W1.dot(x) + self.b1    # (H, 1)
        A1 = self.relu(Z1)               # (H, 1)
        Z2 = self.W2.dot(A1) + self.b2   # (D_out, 1)
        A2 = self.softmax(Z2)            # (D_out, 1)
        cache = (x, Z1, A1, Z2, A2)
        return A2, cache

    def compute_loss(self, A2: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-entropy loss for a single example.
        A2: predicted probabilities (D_out,1)
        y: one-hot true label (D_out,1)
        """
        return -np.sum(y * np.log(A2 + 1e-9))

    def backward(self, cache, y: np.ndarray) -> dict:
        """
        Backward pass: compute gradients.
        cache: (x, Z1, A1, Z2, A2)
        y: one-hot true label (D_out,1)
        Returns: dict with dW1, db1, dW2, db2
        """
        x, Z1, A1, Z2, A2 = cache
        m = 1  # single-sample batch

        # Output layer gradients
        dZ2 = A2 - y                        # (D_out,1)
        dW2 = dZ2.dot(A1.T) / m             # (D_out, H)
        db2 = dZ2 / m                       # (D_out,1)

        # Hidden layer gradients
        dA1 = self.W2.T.dot(dZ2)           # (H,1)
        dZ1 = dA1 * self.relu_derivative(Z1)
        dW1 = dZ1.dot(x.T) / m             # (H, D_in)
        db1 = dZ1 / m                      # (H,1)

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def update_params(self, grads: dict, lr: float = 0.01):
        """
        Gradient descent parameter update.
        """
        self.W1 -= lr * grads["dW1"]
        self.b1 -= lr * grads["db1"]
        self.W2 -= lr * grads["dW2"]
        self.b2 -= lr * grads["db2"]

# Smoke-test
if __name__ == "__main__":
    D_in, H, D_out = 7 * 17 * 17, 64, 9  # adjust D_out to your number of classes
    nn = NeuralNetwork(D_in, H, D_out)

    # dummy example
    x = np.random.randn(D_in, 1)
    y_true = np.zeros((D_out, 1))
    y_true[np.random.randint(0, D_out), 0] = 1

    A2, cache = nn.forward(x)
    loss = nn.compute_loss(A2, y_true)
    print("Initial loss:", loss)

    grads = nn.backward(cache, y_true)
    nn.update_params(grads, lr=0.01)

    A2b, _ = nn.forward(x)
    print("Loss after one update:", nn.compute_loss(A2b, y_true))
