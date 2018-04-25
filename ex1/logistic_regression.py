import numpy as np
import matplotlib.pyplot as plt

import reader
import plot_data


class LogisticRegression(object):
    """
    Classify image 0 -> 9  
    """

    def __init__(self, X, y, alpha=0.01, num_iter=500, num_labels=10):
        self.X = X

        # Insert bias column
        self.X = np.insert(self.X, 0, 1, axis=1)
        (self.m, self.n) = self.X.shape

        self.num_labels = num_labels
        self.num_iter = num_iter
        self.alpha = alpha
        self.Y = np.equal(
            np.tile(y, (1, self.num_labels)),
            np.tile(np.arange(self.num_labels), (self.m, 1)),
        )
        self.Theta = np.random.rand(self.num_labels, self.n) * 2. - 1.
        self._gradient_descent()

    def _hypothesis(self, X):
        """
        Return matrix size m * num_labels
        """
        return self._sigmoid(np.dot(X, self.Theta.T))

    def _sigmoid(self, X):
        return 1. / (np.exp([-1] * X) + 1.)

    def _gradient_descent(self):
        for i in range(self.num_iter):
            if i % 100 == 0:
                print("Iteration #%d" % i)
            h = self._hypothesis(self.X)
            self.Theta = self.Theta - (self.alpha / self.m) * np.dot((h - self.Y).T, self.X)
        print("Done!\n")

    def predict(self, X):
        # Add bias
        X = np.insert(X, 0, 1, axis=1)
        
        h = self._hypothesis(X)
        res = np.argmax(h, axis=1)
        return np.reshape(res, (res.shape[0], 1))


def filter_array(X, y):
    """
    Filter number 0 and 1
    """
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
    (X_train, y_train) = reader.load_mnist('data', kind='train')
    (X_test, y_test) = reader.load_mnist('data', kind='t10k')
    
    # X_train = X_train[:5000]
    # y_train = y_train[:5000]
    # X_test = X_test[:1000]
    # y_test = y_test[:1000]

    X_train, y_train = filter_array(X_train, y_train)
    X_test, y_test = filter_array(X_test, y_test)

    ex1b = LogisticRegression(X_train, y_train, num_labels=2)
    y_guess = ex1b.predict(X_test)
    sum = 0
    for i in range(y_test.shape[0]):
        sum += y_guess[i][0] == y_test[i][0]
        if y_guess[i][0] != y_test[i][0]:
            print(y_guess[i][0], y_test[i][0])
            plot_data.show_image(np.reshape(X_test[i], (28, 28)))
    
    print("Test cases: %d" % y_test.shape[0])
    print("Correct   : %d" % sum)
    print("Rate      : %0.2f" % (1. * sum / y_test.shape[0]))

    while 1:
        plot_data.show_image(
            np.reshape(
                X_test[np.random.randint(X_test.shape[0])],
                (28, 28),
            )
        )
