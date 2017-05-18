import numpy as np
import matplotlib.pyplot as plt
import plotData as Pl
from featureNormalize import normalize
# Load Training data and prepare classes from it

def main():
    data = np.loadtxt('../datasets/levelsData1.txt', dtype=np.int16, delimiter=',')
    X = data[:, 0:2]
    y = data[:,2]
    y.shape = np.size(y), 1
    levels = np.arange(1,9)

    # Plot data of question levels for visualisation
    # Pl.plot(X, y, levels)

    X_norm, mu, sigma = normalize(X)


    # Appending identity column in data set

    X_norm = np.hstack((np.ones((np.size(y), 1)), X_norm))
    thetas = np.zeros((np.size(levels), np.shape(X_norm)[1]))

    fig = plt.figure()
    xc = [1, 1, 1, 1, 2, 2, 2, 2]
    yc = [1, 2, 3, 4, 1, 2, 3, 4]
    for l, xci, yci in zip(levels, xc, yc):
        # ax = fig.add_subplot(xci, yci, 1)
        y_new = 1*(y == l)
        theta = np.zeros((np.shape(X_norm)[1], 1))
        alpha = 5.7
        iters = 10000
        theta, J_hist = gradientDescent(X_norm, y_new, theta, alpha, iters)
        # ax.plot(np.arange(iters), J_hist[:, 0], label='Level '+str(l))
        thetas[l-1] = theta.T
    # print(thetas)
    print(J_hist)
    # plt.show()
    print('Training data accuracy = ',np.mean(1.0*predict(X_norm, thetas, levels) == y)*100)

    data = np.loadtxt('../datasets/testSet.txt', dtype=np.int16, delimiter=',')
    X = data[:, 0:2]
    y = data[:,2]
    y.shape = np.size(y), 1

    # Plot data of question levels for visualisation
    # Pl.plot(X, y, levels)

    X_norm, mu, sigma = normalize(X)


    # Appending identity column in data set

    X_norm = np.hstack((np.ones((np.size(y), 1)), X_norm))
    print('Test data accuracy = ',np.mean(1.0*predict(X_norm, thetas, levels) == y)*100)
    # print(J_hist)



def sigmoid(z):
    return 1/(1 + np.exp(-z))


def costFunction(X, y, theta):
    m = float(np.size(y))
    h = sigmoid(np.dot(X, theta))

    cost = np.dot(-y.T, np.log(h)) - np.dot((1 - y).T, np.log(1 - h))
    J = (1/m) * cost
    grad = (1/m) * (np.dot(X.T, h) - np.dot(X.T, y))
    return J, grad


def gradientDescent(X, y, theta, alpha, iters):
    J_hist = np.zeros((iters, 1))
    for i in range(iters):
        J, grad = costFunction(X, y, theta)
        theta -= alpha * grad
        J_hist[i] = J
    return theta, J_hist


def predict(X, thetas, levels):
    predictions = np.zeros((np.shape(X)[0], 1))
    for i in range(np.shape(X)[0]):
        probs = np.zeros((1, np.size(levels)))
        for l in levels:
            prob = sigmoid(np.dot(thetas[l-1], X[i].T))
            probs[0, l-1] = prob
        temp = np.where(probs == np.max(probs))[1][0]
        predictions[i,0] = temp + 1
    # print(predictions)
    return predictions

if __name__ == '__main__':
    main()
