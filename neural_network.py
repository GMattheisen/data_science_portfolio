
# coding: utf-8

# In[1]:


from sklearn.datasets import make_moons
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


def plot(X, y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

def sigmoid(x): # sigmoid function that accepts NumPy arrays
    return 1.0 / ( 1.0 + np.e ** -x)

def feed_forward(X, weights):
    d1 = np.dot(X, weights[0]) # calculate dot product of X and eights of first layer
    output1 = sigmoid(d1) # apply sigmoid function on result
    input2 = np.hstack([output1, np.ones((output1.shape[0], 1))]) # apply extra 1 for bias
    d2 = np.dot(input2, weights[1]) # calculate dot product of X and weights of second layer
    output2 = sigmoid(d2) # apply sigmoid function on the result
    return output1, output2 # return intermediate results of the sigmoids

def accuracy(ytrue, ypred):
    ypred = ypred.round().flatten().astype(np.int64)
    return sum(ypred == ytrue) / y.shape[0]

def loss(ytrue, ypred): # calculate log loss
    return -(ytrue * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred))

def get_weights():
    weights = [
        np.random.normal(size=(3, 2)),  # two neurons in the first layer
        np.random.normal(size=(3, 1)),  # neuron in the second layer
    ]
    return weights


def backpropagate(weights, out1, out2, X, ytrue, learning_rate=1.0):
    ypred = out2.flatten()
    error = (ypred - ytrue) * loss(ytrue, ypred)   # (N,)

    # calculate weight modification for output layer
    grad_y = ypred * (1 - ypred) * error    # (N,)
    out1_bias = np.hstack([out1, np.ones((out1.shape[0], 1))])
    delta_wout = np.dot(-grad_y, out1_bias) * learning_rate
    weights[1] += delta_wout.reshape((3,1))

    # calculate weight modification for hidden layer
    dp = np.dot(grad_y.reshape(50, 1), weights[1][:2].T)
    grad_h = out1 * (1 - out1) * dp
    delta_whiddden = np.dot(-grad_h.T, X) * learning_rate
    weights[0] += delta_whiddden.T


X, y = make_moons(n_samples=50, noise=0.2, random_state=42)

# add an extra column for the bias
X = np.hstack([X, np.ones((X.shape[0], 1))])

if __name__ == '__main__':
    acc = 0.0
    weights = get_weights()
    while acc < 0.90:
        out1, out2 = feed_forward(X, weights)
        backpropagate(weights, out1, out2, X, y, learning_rate=0.1)
        acc = accuracy(y, out2)
        vloss = loss(y, out2.flatten())
        print(f"accuracy: {acc:5.2}   loss: {sum(vloss):8.5}")
        # break for testing

