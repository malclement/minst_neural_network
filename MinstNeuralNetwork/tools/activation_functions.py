import numpy as np


def selu(x, alpha=1, scale=1):  # SELU and ELU
    return np.where(x <= 0, scale * alpha * (np.exp(x - x.max()) - 1), scale * x)


def selu_dash(x, alpha=1, scale=1):  # SELU and ELU derivative
    return np.where(x <= 0, scale * alpha * np.exp(x - x.max()), scale)


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, axis=0)


def sigmoid(x, derivative=False):
    if derivative:
        return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)
    return 1 / (1 + np.exp(-x))
    
def softsign(x, derivative=False):
    if derivative:
        return 1 / ( 1 + np.abs(x))**2
    return x / (1 + np.abs(x))
