
# coding: utf-8

# # Neural Network from Scratch

# In[ ]:


import numpy as np
from neural_network import sigmoid, feed_forward, loss
from neural_network import X, get_weights

weights = get_weights()

# test for sigmoid function
a = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
expected = np.array([0.0, 0.27, 0.5, 0.73, 1.0])
assert np.all(sigmoid(a).round(2) == expected)

# test the weights
assert weights[0].shape == (3, 2)
assert weights[1].shape == (3, 1)

# test the feed-forward step
out1, out2 = feed_forward(X, weights)
assert out1.shape == (50, 2)
assert out2.shape == (50, 1)

# test the log-loss function
ytrue = np.array([0.0 , 0.0 , 1.0 , 1.0])
ypred = np.array([0.01, 0.99, 0.01, 0.99])
expected = np.array([0.01, 4.61, 4.61, 0.01])
assert np.all(loss(ytrue, ypred).round(2) == expected)

# test the feed-forward step with values that give a known result
Xref = np.array([[1.0, 2.0, 1.0]])
wref = [np.array([[1.0, -1.0],
                  [2.0, -2.0],
                  [0.0,  0.0]
                    ]),
           np.array([[1.0], [-1.0], [0.5]])
          ]
out1, out2 = feed_forward(Xref, wref)
assert np.all(out1.round(2) == np.array([[0.99, 0.01]]))
assert np.all(out2.round(2) == np.array([[0.82]]))


# # Backpropagation Algorithm

# In[ ]:



from sklearn.datasets import make_moons
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np


def plot(X, y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.show()

def sigmoid(x):
    """sigmoid function that accepts NumPy arrays"""
    return 1.0 / ( 1.0 + np.e ** -x)

def feed_forward(X, weights):
    """
    1. calculate the dot product of X                 (N, 3)
       and the weights of the first layer  (2, 3) --> (N, 2)
    2. apply the sigmoid function on the result       (N, 2) return this
    3. append an extra 1 for the bias to the result   (N, 3)
    4. calculate the dot product of X
       and the weights of the second layer (1, 3) --> (N, 1)
    5. apply the sigmoid function on the result       (N, 1) return this
    6. return intermediate results of the sigmoids
    """
    d1 = np.dot(X, weights[0])
    output1 = sigmoid(d1)
    input2 = np.hstack([output1, np.ones((output1.shape[0], 1))])
    d2 = np.dot(input2, weights[1])
    output2 = sigmoid(d2)
    return output1, output2

def accuracy(ytrue, ypred):
    ypred = ypred.round().flatten().astype(np.int64)
    return sum(ypred == ytrue) / y.shape[0]

def loss(ytrue, ypred):
    """Calculate the log loss"""
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
        # break # for testing

