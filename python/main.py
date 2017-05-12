from numpy import *
import plotData as Pl

# Load Training data and prepare classes from it

data = loadtxt('../datasets/levelsData1.txt', dtype=int16, delimiter=',')
X = data[:, 0:2]
y = data[:,2]
levels = arange(1,9)

# Plot data of question levels for visualisation

Pl.plot(X, y, levels)
