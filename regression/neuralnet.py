"""
Three hidden layer neural network
"""


import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def linear(X, weights):
    """
    z = Xw
    X should be (rows, columns) and w should match (columns, hidden units)
    """
    z = np.dot(X, weights)
    return z

def MSE(y, yhat, n=4):
    return np.mean((y - yhat)**2).round(n)


def ReLu(hidden_input):
    return np.maximum(hidden_input, 0)

def dReLu(hidden_input):
    """
    Derivative of ReLu
    """
    ind = hidden_input <= 0
    out = np.ones_like(hidden_input)
    out[ind] = 0
    return out



class NeuralNetwork:
    def __init__(self, input_size, learning_rate=0.0001, h1_units = 9, h2_units = 9, h3_units = 1, seed=None):

        self.learning_rate = 0.0001
        self.h1_units = h1_units
        self.h2_units = h2_units
        self.h3_units = h3_units
        self.input_size = input_size

        np.random.seed(seed)
        self.h1_weights = np.random.normal(size = (input_size, h1_units))
        self.h2_weights = np.random.normal(size = (h1_units, h2_units))
        self.h3_weights = np.random.normal(size = (h2_units, h3_units))

    def forward(self, X):

        # input to first layer
        z = linear(X, self.h1_weights)
        h1 = ReLu(z)

        # hidden1 to hidden2
        z = linear(h1, self.h2_weights)
        h2 = ReLu(z)

        # hidden2 to hidden3
        z = linear(h2, self.h3_weights)
        h3 = ReLu(z)

        return h1, h2, h3

    def backprop(self, h1, h2, h3, X, y, N):

        error = y - h3 # h3 is output

        # gradient wrt h3 (output) weights
        grad3 = error * h2 / N

        # gradient wrt h2 weights
        grad2 = error * self.h3_weights * h1 * dReLu(h1) / N
        # (9x9) = float * W_(9x1) * H_(9x1) * H'_(9x1)

        # gradient wrt h1 weights
        grad1 = error * self.h3_weights * self.h2_weights * dReLu(h1) * X * dReLu(X) / N
        # () = float * W_(9x1) * W_(9x9) * H_(9x1)

        return grad1, grad2, grad3

    
    def gradient_descent(self, update1, update2, update3):

        self.h1_weights += self.learning_rate * update1

        self.h2_weights += self.learning_rate * update2

        self.h3_weights += self.learning_rate * update3

    
    def train(self, epochs, X, y):
        N = X.shape[0]
        self.history = list()
        for _ in range(epochs):
            update1 = np.zeros_like(self.h1_weights)
            update2 = np.zeros_like(self.h2_weights)
            update3 = np.zeros_like(self.h3_weights)
            for x, target in zip(X,y):
                h1, h2, h3 = self.forward(x)
                grad1, grad2, grad3 = self.backprop(h1, h2, h3, x, target, N)
                update1 += grad1
                update2 += grad2
                update3 += grad3
            self.gradient_descent(update1, update2, update3)
            yhat, _ = self.forward(X=X)
            mse_stat = MSE(y = y, yhat = yhat)
            self.history.append(mse_stat)
            print("Epoch {}: | MSE: {}".format(y, mse_stat))

        return self.history





def example_data(rows = 20, columns = 3):
    #np.random.seed(123)
    X = np.random.normal(size=(rows, columns))
    X[:, 0] = np.linspace(start=-10, stop=10, num=rows)**4
    X[:, 1] = np.linspace(start=-10, stop=10, num=rows)
    X[:, 2] = np.linspace(start=-10, stop=10, num=rows)**3
    y = 1.5 * X[:, 0] + X[:, 1] + 2 * X[:, 2]
    y = y[:, None]
    ## scale the values ##
    X = X / 1000
    y = (y - 68) / 500
    return (X, y.round(3))


def forward_run_example():
    X, y = example_data()
    model = NeuralNetwork(X, y)
    print("Forward Run")
    for x, _ in zip(X,y): 
        hidden_output2, hidden_output1 = model.forward(X=x) # returns (final_output, hidden_output1)
        #print(x,target)
    return model, hidden_output2, hidden_output1, X, y

forward_run_example()