import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def MSE(y, yhat):
    return np.mean((y - yhat)**2)

def example_data(rows = 20, columns = 3):
    #np.random.seed(123)
    X = np.random.normal(size=(rows, columns))
    X[:, 0] = np.linspace(start=-10, stop=10, num=rows)**4
    X[:, 1] = np.linspace(start=-10, stop=10, num=rows)
    X[:, 2] = np.linspace(start=-10, stop=10, num=rows)**3
    X = X / 10
    y = 1.5 * X[:, 0] + X[:, 1] + 2 * X[:, 2]
    y = y[:, None]
    ## scale the values ##
    X = X / 100
    y = (y - 68) / 500
    return (X, y)


X, y = example_data()

model = MLPRegressor(hidden_layer_sizes=50, activation='logistic', 
            solver='sgd', alpha=0, learning_rate='constant', learning_rate_init=0.05, 
            max_iter=100, validation_fraction=0)



model.fit(X = X, y = y)

yhat = model.predict(X) 
yhat = yhat* 500 + 68
y = y * 500 + 68

mse_stat = MSE(y = y, yhat = yhat)
print("Final MSE (not scaled): ", mse_stat)
for i in range(3):
    plt.title("X[:, {}] and y".format(i))
    plt.scatter(X[:, i], y=y, marker='o')
    plt.scatter(X[:, i], y=yhat, marker='x')
    plt.show()

