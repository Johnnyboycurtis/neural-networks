import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

def MSE(y, yhat):
    return np.mean((y - yhat)**2)

def example_data(rows = 100):
    x = np.linspace(start=0, stop=30, num = 100).reshape((rows, 1))
    #y = 2*np.sin(x) + x
    y = np.sqrt(x)
    # scale the data
    x = (x - 15)/10
    y = y /10
    return x, y.reshape(len(y))


X, y = example_data()

model = MLPRegressor(hidden_layer_sizes=(50,50), activation='relu', shuffle=False,
            solver='sgd', alpha=0, learning_rate='constant', learning_rate_init=0.0001, 
            max_iter=10000, validation_fraction=0)



model.fit(X = X, y = y)

yhat = model.predict(X) 
plt.plot(y)
plt.plot(yhat)
plt.show()