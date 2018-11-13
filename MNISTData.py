
# coding: utf-8

# In[1]:


import keras


# In[34]:


import tensorflow
from keras.models import Sequential
from keras.layers import Dense


# In[35]:


from keras.datasets import mnist


# In[36]:


(xtrain, ytrain), (xtest, ytest) = mnist.load_data()


# In[5]:


from matplotlib import pyplot as plt

for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(xtrain[i], cmap=plt.cm.Greys)
    plt.axis('off')


# In[6]:


xtrain.dtype


# In[37]:


xtrain.shape # 3D, 60000 is number data points, x dimension of image, y dimension


# In[38]:


xtrain = xtrain.reshape((60000,28*28))


# In[9]:


xtrain.shape #784 is number of features


# In[10]:


xsmall = xtrain[:1000]
ysmall = ytrain[:1000]


# In[39]:


from sklearn.decomposition import PCA


# In[12]:


m_PCA = PCA(n_components=10)


# In[13]:


m_PCA.fit(xtrain)


# In[14]:


m_PCA.components_.shape


# In[15]:


m_PCA.explained_variance_ratio_ # one number for every principle component


# In[16]:


xt = m_PCA.transform(xtrain)


# In[17]:


xback = m_PCA.inverse_transform(xt)
xback.shape


# In[18]:


xback = xback.reshape((60000,28,28))


# In[19]:


def draw(input):
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(input[i], cmap=plt.cm.Greys)
        plt.axis('off')


# In[20]:


draw(xback)


# In[21]:


m_PCA.components_.shape


# In[23]:


comp = m_PCA.components_.reshape((10,28,28))
draw(comp) # each box represents each of the principal components


# In[24]:


#take xt and try to run a prediction on these 10 components


# In[25]:


xt.shape


# In[26]:


from sklearn.linear_model import LogisticRegression


# In[27]:


m = LogisticRegression()


# In[28]:


m.fit(xt,ytrain)


# # Confusion Matrix

# In[29]:


from sklearn.metrics import confusion_matrix


# In[30]:


ypred = m.predict(xt)


# In[31]:


conf = confusion_matrix(ypred,ytrain)


# In[32]:


import seaborn as sns


# In[33]:


sns.heatmap(conf)


# # Decision Tree

# In[34]:


from sklearn.tree import DecisionTreeClassifier


# In[35]:


DecTree = DecisionTreeClassifier(max_depth = 10)


# In[36]:


DecTree.fit(xtrain,ytrain)


# In[37]:


print ("Training score: " + str(DecTree.score(xtrain,ytrain)))


# In[38]:


xtest = xtest.reshape((10000,28*28))


# In[39]:


print ("Test score: " + str(DecTree.score(xtest,ytest)))


# In[40]:


ypred = DecTree.predict(xtest)


# # Keras

# In[41]:


ytrain.shape


# In[42]:


from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


# In[43]:


# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", ytrain.shape)
Y_train = np_utils.to_categorical(ytrain, n_classes)
Y_test = np_utils.to_categorical(ytest, n_classes)
print("Shape after one-hot encoding: ", Y_train.shape)


# In[44]:


model = Sequential([
    Dense(500, input_shape=(784,)),
    Activation('sigmoid'),
    Dense(500),
    Activation('sigmoid'),
    Dense(10),
    Activation('softmax'),
])


# In[45]:


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[46]:


model.fit(xtrain,Y_train,epochs=10, batch_size=500)


# In[47]:


scores = model.evaluate(xtrain, Y_train)
scores


# In[48]:


400 - 0.9818


# In[49]:


model.summary()


# In[50]:


predicted_classes = model.predict(xtest)


# In[51]:


predicted_classes[0].max()


# In[52]:


predicted_classes.shape


# In[53]:


C = np.where(predicted_classes > 0.5, 1, 0)


# In[54]:


C.shape


# In[55]:


correct_indices = np.nonzero(C == Y_test)[0]
incorrect_indices = np.nonzero(C != Y_test)[0]
print()
print(len(correct_indices)," classified correctly")
print(len(incorrect_indices)," classified incorrectly")


# In[56]:


#test accuracy
model.evaluate(xtest, Y_test)


# In[57]:


what = model.predict_classes(xtest)


# In[58]:


what


# In[59]:


# adapt figure size to accomodate 18 subplots
plt.rcParams['figure.figsize'] = (7,14)

figure_evaluation = plt.figure()

# plot 9 correct predictions
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(6,3,i+1)
    plt.imshow(xtest[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted: {}, Truth: {}".format(what[correct],
                                        ytest[correct]))
    plt.xticks([])
    plt.yticks([])
    plt.show()

# plot 9 incorrect predictions
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(6,3,i+10)
    plt.imshow(xtest[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title(
      "Predicted {}, Truth: {}".format(what[incorrect], 
                                       ytest[incorrect]))
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
figure_evaluation


# # Drawing Keras Model

# In[60]:


from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


# In[61]:


SVG(model_to_dot(model,  show_shapes=True,
    show_layer_names=True, rankdir='HB').create(
       prog='dot', format='svg'))


# # Tensor Board

# In[62]:


from keras.callbacks import TensorBoard


# In[69]:


tboard = keras.callbacks.TensorBoard(log_dir='./output',histogram_freq=5, write_graph=True, write_images=True)


# In[ ]:


model.fit(xtrain,Y_train,epochs=5, batch_size=500, callbacks = [tboard],validation_split=0.2)


# # Convulted Neural Network

# In[124]:


from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D
from keras import backend as K
import keras
from keras.optimizers import SGD
from keras.utils import to_categorical


# In[136]:


(X, y), (xtest, ytest) = mnist.load_data()


# In[137]:


ytrain = to_categorical(y,num_classes =10)


# In[138]:


ytrain.shape


# In[139]:


xtrain.shape


# In[134]:


ytest = to_categorical(ytest,num_classes =10)


# In[131]:


X = X.reshape(60000,28,28,1)


# In[132]:


xtest = xtest.reshape(10000,28,28,1)


# In[62]:


model = Sequential([
    Dense(4, input_shape=(28,28)),
    Activation('relu'),
    Flatten(),
    Dense(10),
    Activation('softmax'),
])


# In[63]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])


# In[125]:


model = Sequential([
    Conv2D(filters=8,kernel_size=(3,3),strides=(1,1), input_shape=(28,28,1),padding ='same'),
    MaxPooling2D(pool_size=(2,2),padding='valid'),
    Activation('relu'),
    Conv2D(filters=16,kernel_size=(3,3),strides=(1,1), padding ='same'),
    MaxPooling2D(pool_size=(2,2),padding='valid'),
    Activation('relu'),
    Flatten(),
    Dense(10),
    Activation('softmax'),
])


# In[126]:


model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])


# In[127]:


model.fit(X,ytrain, batch_size=200, epochs=5)


# In[135]:


print('Train score: ' + str(model.evaluate(X,ytrain)))
print('Test score: ' + str(model.evaluate(xtest,ytest)))

