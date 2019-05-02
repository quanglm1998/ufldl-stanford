import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt    

class LinearRegression:
    def __init__(self, X, y, lambda1, lambda2, alpha=0.1, numIter=10000):
        self.X = X.copy()
        (self.mu, self.sigma) = self.normalize()
        self.X = np.hstack(((np.ones((self.X.shape[0], 1)), self.X)))
        self.y = y
        self.alpha = alpha
        self.theta = np.ones((self.X.shape[1], 1))
        self.numIter = numIter
        (self.m, self.n) = self.X.shape
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def normalize(self):
        mu = np.mean(self.X, axis=0)
        sigma = np.std(self.X, axis=0)
        self.X -= np.tile(mu, (self.X.shape[0], 1))
        self.X /= np.tile(sigma, (self.X.shape[0], 1))
        return (mu, sigma)

    def gradientDescent(self):
        for i in range(self.numIter):
            # print("Iteration:", i, end='\r')
            self.theta -= (self.alpha / self.m) * (self.X.T.dot(self.X.dot(self.theta) - self.y)) + self.lambda1 * np.sign(self.theta) + self.lambda2 * self.theta * 2

    def getNormalize(self, X):
        res = X.copy()
        res -= np.tile(self.mu, (res.shape[0], 1))
        res /= np.tile(self.sigma, (res.shape[0], 1))
        return res

    def getCost(self, X, y):
        Xnorm = self.getNormalize(X)
        # Add x0 = 1
        Xnorm = np.hstack(((np.ones((Xnorm.shape[0], 1)), Xnorm)))
        #print("getCost X", X.shape)
        #print("theta", self.theta.shape)
        return (1 / 2 / self.m) * sum((Xnorm.dot(self.theta) - y) ** 2)

    def getLoss(self):
        return self.getCost(self.X, self.y) + self.lambda1 * np.sum(np.abs(self.theta)) + self.lambda2 * np.sum(self.theta * self.theta)

    def predictCost(self, X):
        Xnorm = self.getNormalize(X)
        # Add x0 = 1
        Xnorm = np.hstack(((np.ones((Xnorm.shape[0], 1)), Xnorm)))
        return Xnorm.dot(self.theta)


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)



data = np.loadtxt("data/housing.data")
# Add x0 = 1 
#data = np.hstack(((np.ones((data.shape[0], 1)), data)))
# Random data set
id = np.random.permutation(data.shape[0])
#id = np.arange(data.shape[0])

print(data.shape)

TRAIN_SIZE = 50

# The first TRAIN_SIZE rows are training set, others are testing set
data = data[id, :]
X = data[:TRAIN_SIZE, :-1]
y = data[:TRAIN_SIZE, -1:]
test_X = data[TRAIN_SIZE:, :-1]
test_y = data[TRAIN_SIZE:, -1:]

ex1 = LinearRegression(X, y, 0, 0)
ex1.gradientDescent()
print("cost simple", ex1.getCost(test_X, test_y))

ex2 = LinearRegression(X, y, 0.05, 0)
ex2.gradientDescent()
print("cost lasso", ex2.getCost(test_X, test_y))

ex3 = LinearRegression(X, y, 0, 0.05)
ex3.gradientDescent()
print("cost ridge", ex3.getCost(test_X, test_y))

ex4 = LinearRegression(X, y, 0.04, 0.005)
ex4.gradientDescent()
print("cost elastic", ex4.getCost(test_X, test_y))

print(ex1.theta)
print()
print(ex2.theta)

predict = ex1.predictCost(test_X)

# Sort test_y and predict
id = np.argsort(test_y.T)
test_y = test_y[id[0]]
predict = predict[id[0]]

# Plot data
plt.plot(test_y, 'rx', label="Actual price")
plt.plot(predict, 'bx', label="Predicted price")
plt.xlabel("House #")
plt.ylabel("House price ($1000)")
plt.legend(loc=2, prop={'size': 10})
plt.show()
