import sys
import numpy as np



def linear(weights, X_input):
    #print(f"X_{X_input.shape} \\times weights_{weights.shape}")
    a = np.dot(X_input, weights)
    #print(f"X_{X_input.shape} \\times weights_{weights.shape} = a_{a.shape}")
    return a

def sigmoid(hidden_input):
    a = 1 / (1 + np.exp(hidden_input))
    return a

def _relu(u):
    if u <= 0:
        return 0
    else:
        return u

ReLu = np.vectorize(_relu, otypes=[float]) # need to vectorize to apply to numpy arrays

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
        self.weights1 = np.random.normal(loc=0, scale=1, size=(X.shape[1], hidden_units))
        self.weights2 = np.random.normal(loc=0, scale=1, size=(hidden_units, 1)) # for now there will only be one output unit


    def forward(self, X = None):
        if not isinstance(X, np.ndarray):
            X = self.X
        hidden_output1 = linear(self.weights1, X)
        hidden_output1 = sigmoid(hidden_output1) # apply the nonlinear transformation

        hidden_output2 = linear(self.weights2, hidden_output1)

        return hidden_output2, hidden_output1


    def backpropogation(self, hidden_output1, hidden_output2):
        #print("Hidden Output Shapes", hidden_output1.shape, hidden_output2.shape)
        output_error = (self.y - hidden_output2) #  -(target - y); 20 x 1

        # update hidden layer weights
        hidden_error = linear(self.weights2.T, output_error) # neccessary for hidden layer updates; 20 x 9
        update1 = hidden_error * hidden_output2 * (1 - hidden_output2) # logistic
        update1 = np.dot(update1.T, self.X)
        #print("update1: ", update1.shape)

        # update output layer weights; linear not logistic
        update2 = output_error * hidden_output1 # use hidden layer outputs to update output layer weights
        update2 = update2.sum(axis=0).reshape((9,1))
        #print("Hidden Output1", hidden_output1.shape, (output_error * hidden_output1).shape)
        #print("update2: ", update2.shape)
        
        return update2, update1

    def gradient_descent(self, update1, update2):
        update1 = update1.sum(axis=1)
        self.weights1 += self.learning_rate * update1 # hidden layer weights
        self.weights2 += self.learning_rate * update2 # output layer weights
        return None


    def train(self, n_epochs = 15):
        for _ in range(n_epochs):
            hidden_output2, hidden_output1 = self.forward()
            mse = MSE(self.y, hidden_output2)
            print("MSE: ", round(mse, 4), flush=True)
            sys.stdout.flush()
            update2, update1 = self.backpropogation(hidden_output1, hidden_output2)
            self.gradient_descent(update1, update2)




def example_data(rows = 20, columns = 3):
    np.random.seed(123)
    X = np.random.normal(size=(rows, columns))
    y = 1.5 * X[:, 0] + X[:, 1] + 2 * X[:, 2]
    y = y.reshape((rows, 1))
    return (X, y)


def forward_run_example():
    X, y = example_data()
    model = NeuralNetwork(X, y)
    print("Forward Run")
    hidden_output2, hidden_output1 = model.forward() # returns (final_output, hidden_output1)
    return model, hidden_output2, hidden_output1





def backprop_run_example():
    model, hidden_output2, hidden_output1 = forward_run_example()
    
    # back prop step
    print("Backward Run")
    update2, update1 = model.backpropogation(hidden_output1, hidden_output2)
    return model, update2, update1




def gradient_descent_example():
    model, update2, update1 = backprop_run_example()
    model.gradient_descent(update1, update2)


#gradient_descent_example()


def run_example():
    X, y = example_data()
    model = NeuralNetwork(X, y)
    model.train(n_epochs=50)
    return model

run_example()