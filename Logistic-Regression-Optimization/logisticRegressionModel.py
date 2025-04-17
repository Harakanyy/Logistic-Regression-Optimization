import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class LogisticRegression:
    def __init__(self, lr=0.01, epochs=500, mode='batch', batch_size=32):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.mode = mode  # 'batch', 'sgd', 'mini-batch'
        self.batch_size = batch_size
        print(f"Initialized LogisticRegression with lr={self.lr}, epochs={self.epochs}, mode={self.mode}")

    def fit(self, X, y):
        self.losses = []  # store losses as a class attribute
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            if self.mode == 'batch':
                X_batch, y_batch = X, y
                self._update_weights(X_batch, y_batch, n_samples)

            elif self.mode == 'sgd':
                for i in range(n_samples):
                    X_batch = X[i].reshape(1, -1)
                    y_batch = np.array([y[i]])
                    self._update_weights(X_batch, y_batch, 1)

            elif self.mode == 'mini-batch':
                indices = np.random.permutation(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                for i in range(0, n_samples, self.batch_size):
                    X_batch = X_shuffled[i:i + self.batch_size]
                    y_batch = y_shuffled[i:i + self.batch_size]
                    self._update_weights(X_batch, y_batch, X_batch.shape[0])

            # Calculate loss for monitoring
            linear_model = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_model)
            loss = - (1/n_samples) * np.sum(y * np.log(predictions + 1e-9) + (1 - y) * np.log(1 - predictions + 1e-9))
            self.losses.append(loss)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}")


    def _update_weights(self, X_batch, y_batch, batch_size):
        linear_model = np.dot(X_batch, self.weights) + self.bias
        predictions = sigmoid(linear_model)

        dw = (1 / batch_size) * np.dot(X_batch.T, (predictions - y_batch))
        db = (1 / batch_size) * np.sum(predictions - y_batch)

        self.weights -= self.lr * dw
        self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred_proba = sigmoid(linear_model)
        y_pred_class = [1 if prob > 0.5 else 0 for prob in y_pred_proba]
        return y_pred_class, y_pred_proba