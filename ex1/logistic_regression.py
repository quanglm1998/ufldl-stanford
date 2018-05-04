import numpy as np
import matplotlib.pyplot as plt

import reader
import plot_data


class LogisticRegression(object):
    """Classify image 0 -> 9"""

    def __init__(self, X, y, alpha=0.3, num_iter=1000, num_labels=10):
        self.X = X
        self.X = 1. * self.X / 256  # Normalize X
        self.X = np.insert(self.X, 0, 1, axis=1)    # Insert bias column
        (self.m, self.n) = self.X.shape

        self.num_labels = num_labels
        self.num_iter = num_iter
        self.alpha = alpha
        self.Y = 1. * np.equal(
            np.tile(y, (1, self.num_labels)),
            np.tile(np.arange(self.num_labels), (self.m, 1)),
        )
        self.Theta = np.random.rand(self.num_labels, self.n) * 0.01
        self._gradient_descent()

    def _hypothesis(self, X):
        """Return matrix size m * num_labels"""

        return self._sigmoid(np.dot(X, self.Theta.T))

    @classmethod
    def _sigmoid(self, X):
        return 1. / (np.exp(-X) + 1.)

    def _get_cost(self, X, Y):
        h = self._hypothesis(X)
        return -1. * np.sum(Y * np.log(h) + (1. - Y) * np.log(1. - h)) / self.m

    def _gradient_descent(self):
        cost = []
        for i in range(self.num_iter):
            if i % 100 == 0:
                print("Iteration #%d" % i)
            h = self._hypothesis(self.X)
            self.Theta = self.Theta - (self.alpha / self.m) * np.dot((h - self.Y).T, self.X)
            cost.append(self._get_cost(self.X, self.Y))
        print("Done!\n")
        plt.plot(cost)
        plt.show()

    def predict(self, X):
        X = 1. * X / 256    # Normalize X
        X = np.insert(X, 0, 1, axis=1)  # Add bias
        
        h = self._hypothesis(X)
        res = np.argmax(h, axis=1)
        return np.reshape(res, (res.shape[0], 1))


def filter_array(X, y):
    """Filter number 0 and 1"""

    X = []
    y = []
    for i in range(X_test.shape[0]):
        if y_test[i][0] == 0 or y_test[i][0] == 1:
            X.append(X_test[i])
            y.append(y_test[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

if __name__ == "__main__":
    # Read data
    (X_train, y_train) = reader.load_mnist('data', kind='train')
    (X_test, y_test) = reader.load_mnist('data', kind='t10k')
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test
    y_test = y_test

    # X_train, y_train = filter_array(X_train, y_train)
    # X_test, y_test = filter_array(X_test, y_test)

    ex1b = LogisticRegression(X_train, y_train)
    y_guess = ex1b.predict(X_test)

    # Get accuracy
    sum = 0
    for i in range(y_test.shape[0]):
        sum += y_guess[i][0] == y_test[i][0]
    
    print("Test cases: %d" % y_test.shape[0])
    print("Correct   : %d" % sum)
    print("Rate      : %0.2f" % (1. * sum / y_test.shape[0]))
