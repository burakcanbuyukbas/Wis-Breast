import numpy as np
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plt
import itertools


def tanh(x):
    return (np.tanh(x))


def tanh_prime(x):
    return (1 - np.tanh(x) ** 2)


# loss function and its derivative
def mse(y_true, y_pred):
    return (np.mean(np.power(y_true - y_pred, 2)))


def mse_prime(y_true, y_pred):
    return (2 * (y_pred - y_true) / y_true.size)

def Sigmoid(x):
    return 1/(1+np.exp(-x))

def dSigmoid(x):
    s = 1/(1+np.exp(-x))
    dx = s * (1-s)
    return dx

def ReLU(x):
    return np.maximum(0, x)

def dReLU(x):
    y = (x > 0) * 1
    return y

def plotConfusionMatrix(a,b):
    cf = confusion_matrix(a, b)
    plt.imshow(cf, cmap=plt.get_cmap('Blues'), interpolation='nearest')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    tick_marks = np.arange(len(set(a)))
    class_labels = ['0', '1']
    plt.xticks(tick_marks, class_labels)
    plt.yticks(tick_marks, class_labels)
    thresh = cf.max() / 2.
    for i, j in itertools.product(range(cf.shape[0]), range(cf.shape[1])):
        plt.text(j, i, format(cf[i, j], 'd'), horizontalalignment='center',
                 color='white' if cf[i, j] > thresh else 'black')
    plt.show()

