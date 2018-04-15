import numpy
import scipy
import matplotlib
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, X, y, alpha=0.3, numIter=5000):
        self.X = X
        (self.mu, self.sigma) = self.normalize()
        # Add x0 = 1 
        self.X = numpy.hstack(((numpy.ones((self.X.shape[0], 1)), self.X)))
        self.y = y
        self.alpha = alpha
        self.theta = numpy.ones((self.X.shape[1], 1))
        self.numIter = numIter
        (self.m, self.n) = self.X.shape

    def normalize(self):
        mu = numpy.mean(self.X, axis=0)
        sigma = numpy.std(self.X, axis=0)
        self.X -= numpy.tile(mu, (self.X.shape[0], 1))
        self.X /= numpy.tile(sigma, (self.X.shape[0], 1))
        return (mu, sigma)

    def gradientDescent(self):
        for i in range(self.numIter):
            print("Iteration:", i, end='\r')
            self.theta -= (self.alpha / self.m) * (self.X.T.dot(self.X.dot(self.theta) - self.y))  

    def getNormalize(self, X):
        X -= numpy.tile(self.mu, (X.shape[0], 1))
        X /= numpy.tile(self.sigma, (X.shape[0], 1))

    def getCost(self):
        return (1 / 2 / self.m) * sum((self.X.dot(self.theta) - self.y) ** 2)

    def predictCost(self, X):
        self.getNormalize(X)
        # Add x0 = 1
        X = numpy.hstack(((numpy.ones((X.shape[0], 1)), X)))
        return X.dot(self.theta)


data = numpy.loadtxt("data/housing.data")
# Random data set
id = numpy.random.permutation(data.shape[0])

# The first 400 rows are training set, others are testing set
data = data[id, :]
X = data[:400, :-1]
y = data[:400, -1:]
test_X = data[400:, :-1]
test_y = data[400:, -1:]

ex1 = LinearRegression(X, y)
ex1.gradientDescent()
predict = ex1.predictCost(test_X)

# Sort test_y and predict
id = numpy.argsort(test_y.T)
test_y = test_y[id[0]]
predict = predict[id[0]]

# Plot data
plt.plot(test_y, 'rx', label="Actual price")
plt.plot(predict, 'bx', label="Predicted price")
plt.xlabel("House #")
plt.ylabel("House price ($1000)")
plt.legend(loc=2, prop={'size': 10})
plt.show()