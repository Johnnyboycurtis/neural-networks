import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def linear(X_input, weights):
    """
    out = Xw
    X should be (rows, columns) and w should match (columns, hidden units)
    """
    #print(f"X_{X_input.shape} \\times weights_{weights.shape}")
    a = np.dot(X_input, weights)
    #print(f"X_{X_input.shape} \\times weights_{weights.shape} = a_{a.shape}")
    return a

def sigmoid(hidden_input):
    a = 1 / (1 + np.exp(hidden_input))
    return a

def MSE(y, yhat):
    return np.mean((y - yhat)**2)


class NeuralNetwork:

    def __init__(self, X, y, hidden_units = 9, learning_rate = 0.1):
        """
        Two hidden layer neural network for regression
        """
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.weights1 = np.random.normal(loc=0, scale=0.5, size=(X.shape[1], hidden_units)) # (input units, hidden units)
        self.weights2 = np.random.normal(loc=0, scale=0.5, size=(hidden_units, 1)) # for now there will only be one output unit


    def forward(self, X = None):
        if isinstance(X, np.ndarray):
            hidden_output1 = linear(X, self.weights1)
            hidden_output1 = sigmoid(hidden_output1) # apply the nonlinear transformation

            hidden_output2 = linear(hidden_output1, self.weights2)

            return hidden_output2, hidden_output1
        else:
            raise ValueError("need a vector in the forward function")


    def backpropogation(self, hidden_output1, hidden_output2, x, target):
        output_error = (target - hidden_output2) #  -(target - y); 20 x 1

        # update hidden layer weights
        hidden_error = output_error * self.weights2 # neccessary for hidden layer updates; 9 x 1
        update1 = hidden_error * hidden_output2 * (1 - hidden_output2) # logistic
        update1 = update1 * x
        #print("update1: ", update1.shape)

        # update output layer weights; linear not logistic
        update2 = output_error * hidden_output1 # use hidden layer outputs to update output layer weights
        #print("update2: ", update2.shape)
        return update2, update1

    def gradient_descent(self, update1, update2):
        #print(self.weights1.shape, update1.T.shape)
        #print(self.weights2.shape, update2.shape)
        #print(self.weights2 + update2[:, None])
        self.weights1 += self.learning_rate * update1.T # hidden layer weights
        self.weights2 += self.learning_rate * update2[:, None] # output layer weights
        return None


    def train(self, n_epochs = 15):
        for _ in tqdm(range(n_epochs)):
            for x, y in zip(self.X, self.y):
                hidden_output2, hidden_output1 = self.forward(X=x)
                update2, update1 = self.backpropogation(hidden_output1, hidden_output2, x=x, target=y)
                self.gradient_descent(update1, update2)
            yhat, _ = self.forward(X=self.X)
            mse_stat = MSE(y = self.y, yhat = yhat)
            #print("MSE: ", mse_stat)

        return yhat




def example_data(rows = 20, columns = 3):
    #np.random.seed(123)
    X = np.random.normal(size=(rows, columns))
    X[:, 0] = np.linspace(start=-10, stop=10, num=rows)**3
    X[:, 1] = np.linspace(start=-10, stop=10, num=rows)
    X[:, 2] = np.linspace(start=-10, stop=10, num=rows)**2
    X = X / 10
    y = 1.5 * X[:, 0] + X[:, 1] + 2 * X[:, 2]
    y = y[:, None]
    ## scale the values ##
    X = X / 10
    y = (y - 68) / 500
    return (X, y)


def forward_run_example():
    X, y = example_data()
    model = NeuralNetwork(X, y)
    print("Forward Run")
    for x, _ in zip(X,y): 
        hidden_output2, hidden_output1 = model.forward(X=x) # returns (final_output, hidden_output1)
        #print(x,target)
    return model, hidden_output2, hidden_output1, X, y

#forward_run_example()



def backprop_run_example():
    model, hidden_output2, hidden_output1, X, y = forward_run_example()
    print("Backward Run")
    for (x, target) in zip(X, y):
        update2, update1 = model.backpropogation(hidden_output1, hidden_output2, x, target)
    return model, update2, update1


#backprop_run_example()

def gradient_descent_example():
    model, update2, update1 = backprop_run_example()
    model.gradient_descent(update1, update2)


#gradient_descent_example()


def run_example():
    X, y = example_data()
    model = NeuralNetwork(X, y, hidden_units=50, learning_rate=0.05)
    yhat = model.train(n_epochs=100)
    yhat = yhat*500 + 68
    y = y*500 + 68
    mse_stat = MSE(y = y, yhat = yhat)
    print("Final MSE (not scaled): ", mse_stat)
    for i in range(3):
        plt.title("X[:, {}] and y".format(i))
        plt.scatter(X[:, i], y=y, marker='o')
        plt.scatter(X[:, i], y=yhat, marker='x')
        plt.show()
    return model

run_example()