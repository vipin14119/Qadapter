import numpy as np
import scipy as sp
from scipy.optimize import minimize as fmin
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import plotData as Pl

from featureNormalize import normalize
from sklearn.model_selection import train_test_split

# Load Training data and prepare classes from it

class Adaptive:
    """
        Adaptive algorithm class for learning question levels from data
    """

    def __init__(self, path):
        """
        @method: constructor of Adaptive class
        @param: path -> path of dataset
        """

        self.path = path
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.levels = None
        self.nlevels = None

        self.mu = None
        self.sigma = None

        self.computed_curve = None


    def prepare_data(self):
        """
        @method: divide data into classes and input variables
        """

        print('=== Preparing data ===')
        data = np.loadtxt(self.path, dtype=np.int16, delimiter=',')
        self.X = data[:, 0:2]
        self.y = data[:,2]
        self.y.shape = np.size(self.y), 1
        self.levels = np.arange(1, 17)

        seed = 7
        test_size = 0.33
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=seed)


    def feature_normalize(self):
        """
        @method: normalize input feature and append a unity column
        """

        print('=== Normalizing input features ===')
        mu = np.mean(self.X_train)
        sigma = np.std(self.X_train)
        X_norm = np.hstack((np.ones((np.size(self.y_train), 1)), (self.X_train - mu)/sigma))
        return X_norm, mu, sigma

    def plot_data(self):
        """
        @method: draw a scatter plot of data corresponding to levels specified
        """

        colors = cm.rainbow(np.linspace(0, 1, len(self.levels)))
        fig = plt.figure()

        for l,c in zip(self.levels, colors):
            indices = np.where(self.y == l)[0]
            ax1 = fig.add_subplot(111)
            ax1.scatter(self.X[indices,0], self.X[indices,1], s=10, color=c, marker='s', label='Level'+str(l))
        plt.legend(loc='Upper left');
        plt.show()

    def gradient_descent(self, theta, alpha, iters):
        J_hist = np.zeros((iters, 1))
        for i in range(iters):
            J, grad = self.costFunction(theta)
            theta -= alpha * grad
            J_hist[i] = J
        return theta, J_hist

    @staticmethod
    def sigmoid(z):
        """
            @method: compute sigmoid of given vector
            @param: z -> column vector
        """
        
        return 1/(1 + np.exp(-z))


    def costFunction(self, theta, y_new):
        m = float(np.size(y_new))
        h = self.sigmoid(np.dot(self.X_train, theta))
        # print(h)
        cost = np.dot(-y_new.T, np.log(h)) - np.dot((1 - y_new).T, np.log(1.00001 - h))
        J = (1/m) * cost
        grad = (1/m) * (np.dot(self.X_train.T, h) - np.dot(self.X_train.T, y_new))
        return J

    def predict(self, X, thetas):

        predictions = np.zeros((np.shape(X)[0], 1))
        for i in range(np.shape(X)[0]):
            probs = np.zeros((1, np.size(self.levels)))
            for l in self.levels:
                prob = self.sigmoid(np.dot(thetas[l-1], X[i].T))
                probs[0, l-1] = prob

            temp = np.where(probs == np.max(probs))[1][0]
            # print(mu, sigma)
            X[i] = (X[i]*self.sigma)+self.mu
            print('Predicted Level for {} sec and {} accuracy is {} with {} Confidence'.format(X[i,1], X[i,2], temp+1, probs[0,temp]*100))
            predictions[i,0] = temp + 1
        # print(predictions)
        return predictions

    def run(self):
        print('=== Running Adaptive method ===')

        adaptive.prepare_data()
        self.X_train, self.mu, self.sigma = self.feature_normalize()

        thetas = np.zeros((np.size(self.levels), np.shape(self.X_train)[1]))

        colors = cm.rainbow(np.linspace(0, 1, len(self.levels)))
        for l, c in zip(self.levels, colors):
            y_new = 1*(self.y_train == l)
            theta = self.get_optimized_theta(y_new)
            thetas[l-1] = theta


        X = np.hstack((np.ones((np.size(self.y_test), 1)), (self.X_test - self.mu)/self.sigma))
        print X
        # print(thetas)
        print('\n============= Training Data Accuracy ==========')
        print(np.mean(1.0*self.predict(X, thetas) == self.y_test)*100)
        print('==================================================\n')

    def get_optimized_theta(self, y):
        print(np.shape(self.X_train))
        theta = np.zeros((np.shape(self.X_train)[1], 1))
        theta = fmin(self.costFunction, theta, args=(y), method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
        return theta.x

    def compute_error(self):
        """
        @method: compute square mean error in hypothesis and predicted values
        """

        return np.sum((self.computed_curve(self.X) - self.y)**2)

    def compute_curve(self):
        # fp1, res, rank, sv, rcond = np.ployfit(X, y, 1, full=True)
        print(np.polyfit(self.X, self.y, 1, full=True))


if __name__ == '__main__':

    path = '../datasets/levelsData16.txt'
    adaptive = Adaptive(path)
    adaptive.run()
    # adaptive.prepare_data()
    # adaptive.plot_data()
