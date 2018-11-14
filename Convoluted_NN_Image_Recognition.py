
# coding: utf-8

# In[6]:


from matplotlib import pyplot as plt
import numpy as np
from keras.applications import ResNet50
from keras.applications.resnet50 import decode_predictions
from keras.preprocessing import image


# In[4]:


im = plt.imread('rainy_day.jpg')
plt.imshow(im[::2,::2])
plt.axis('off')


# In[7]:


m = ResNet50()


# In[255]:


a = []
def splice(image):
    for i in range(1,6):
        for j in range(1,7):
            a.append(im[224*(i-1):224*i,224*(j-1):224*j])


# In[256]:


splice(im)


# In[261]:


a = np.array(a)
predictions = m.predict(a)


# In[265]:


decoded = decode_predictions(predictions, top =3)


# In[287]:


for lev1 in decoded:
    for _,lb,prob in lev1:
        if prob > 0.25:
            print("Label: ",lb,prob)

