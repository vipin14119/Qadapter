from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def plot(X, y, levels):
    colors = cm.rainbow(linspace(0, 1, len(levels)))
    fig = plt.figure()

    for l,c in zip(levels, colors):
        indices = where(y == l)[0]
        ax1 = fig.add_subplot(111)
        ax1.scatter(X[indices,0], X[indices,1], s=10, color=c, marker='s', label='Level'+str(l))
    plt.legend(loc='Upper left');
    plt.show()
