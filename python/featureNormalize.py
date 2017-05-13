from numpy import *

def normalize(X):
    print('---- Normalizing you data ----')
    mu = mean(X)
    sigma = std(X)
    return (X - mu)/sigma, mu, sigma
