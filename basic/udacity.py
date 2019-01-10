import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        np.random.seed(123)
        self.weights_input_to_hidden = np.random.normal(0.0, 1.0,
                                       (self.input_nodes, self.hidden_nodes)).round(3)

        self.weights_hidden_to_output = np.random.normal(0.0, 1.0,
                                       (self.hidden_nodes, self.output_nodes)).round(3)

        print("Weights 1")
        print(self.weights_input_to_hidden.round(3))
        print("Weights 2")
        print(self.weights_hidden_to_output.round(3))

        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        # self.activation_function = lambda x : 0  # Replace 0 with your sigmoid calculation.

        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your
        # implementation there instead.
        #
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here

        self.activation_function = sigmoid


    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):

            final_outputs, hidden_outputs = self.forward_pass_train(X, verbose=True)  # Implement the forward pass function below

            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y,
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X, verbose=False):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs,
                      self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        if verbose:
            pass
            #print("hidden output: ", hidden_outputs.round(3))
            #print("output: ", final_outputs.round(3))

        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y,
                        delta_weights_i_h, delta_weights_h_o):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers

        '''
        #### Implement the backward pass here ####
        ### Backward pass ###

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.

        output_error_term = error

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(self.weights_hidden_to_output,
                              output_error_term)
        # TODO: Backpropagated error terms - Replace these values with your calculations.

        hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

        # Weight step (input to hidden)
        update1 = hidden_error_term * X[:, None]
        delta_weights_i_h += update1
        # Weight step (hidden to output)
        delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        return delta_weights_i_h, delta_weights_h_o

    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = final_inputs # signals from final output layer

        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 50
learning_rate = 0.1
hidden_nodes = 9
output_nodes = 1


def MSE(y, yhat):
    return np.mean((y - yhat)**2)

def example_data(rows = 20, columns = 3):
    #np.random.seed(123)
    X = np.random.normal(size=(rows, columns))
    X[:, 0] = np.linspace(start=-10, stop=10, num=rows)**4
    X[:, 1] = np.linspace(start=-10, stop=10, num=rows)
    X[:, 2] = np.linspace(start=-10, stop=10, num=rows)**3
    y = 1.5 * X[:, 0] + X[:, 1] + 2 * X[:, 2]
    y = y[:, None]
    ## scale the values ##
    X = X / 100
    y = (y - 68) / 500
    return (X, y.round(3))

def run_example():
    iterations = 1000
    learning_rate = 0.09
    hidden_nodes = 20
    output_nodes = 1
    X, y = example_data()
    model = NeuralNetwork(input_nodes=3, hidden_nodes=hidden_nodes, output_nodes=1, learning_rate=learning_rate)
    for _ in range(iterations):
        model.train(features=X, targets=y)
        yhat, _ = model.forward_pass_train(X = X)

    yhat = yhat*500 + 68
    y = y*500 + 68
    for i in range(3):
        plt.title("X[:, {}] and y".format(i))
        plt.scatter(X[:, i], y=y, marker='o')
        plt.scatter(X[:, i], y=yhat, marker='x')
        plt.show()

    print("Weights 1")
    print(model.weights_input_to_hidden.round(3))
    print("Weights 2")
    print(model.weights_hidden_to_output.round(3))

run_example()
