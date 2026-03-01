# Stochastic Gradient Descent (SGD) from scratch
# Dataset: sklearn Diabetes dataset

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load and split data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


class SGD:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.intercept = None
        self.coef = None
        self.loss_history = []

    def fit(self, X_train, y_train):
        # Initialize weights
        self.intercept = 0
        self.coef = np.ones(X_train.shape[1])

        for _ in range(self.epochs):
            for _ in range(X_train.shape[0]):
                # Pick a random sample (stochastic)
                idx = np.random.randint(0, X_train.shape[0])
                y_hat = self.intercept + np.dot(X_train[idx], self.coef)

                # Gradient of MSE loss and weight update
                intercept_der = -2 * (y_train[idx] - y_hat)
                self.intercept = self.intercept - self.learning_rate * (intercept_der)

                coef_der = -2 * np.dot((y_train[idx] - y_hat), X_train[idx])
                self.coef = self.coef - self.learning_rate * (coef_der)

            # Track full MSE loss after each epoch
            y_hat_full = self.intercept + np.dot(X_train, self.coef)
            self.loss_history.append(np.mean((y_train - y_hat_full) ** 2))

    def predict(self, X_test):
        return self.intercept + np.dot(X_test, self.coef)


if __name__ == "__main__":
    model = SGD(learning_rate=0.01, epochs=100)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("R² Score:", r2_score(y_test, y_pred))