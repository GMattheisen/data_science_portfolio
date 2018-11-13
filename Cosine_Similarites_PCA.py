
# coding: utf-8

# In[2]:


import pandas as pd

PATH = "movie_recommendations (1).xlsx"

df = pd.read_excel(PATH)

# produce clean unique index
df.sort_values(by=['Name', 'Genre', 'Reviewer'], inplace=True)
clean = df.drop_duplicates().dropna()
df['Genre'] = df['Genre'].str.lower()


long = df.groupby(['Genre', 'Reviewer'])['Rating'].mean()

wide = long.unstack(0)
wide.fillna(0.0, inplace=True)


# In[3]:


import math


# In[4]:


wide


# In[127]:


def cosim(v, w):
    return dot(v, w) / math.sqrt(dot(v, v) * dot(w, w))


# In[19]:


kristian = wide.loc['Kristian']
mano = wide.loc['Manohar']
paul = wide.loc['Paul']
tom = wide.loc['Tom']
julian = wide.loc['Julian']


# In[20]:


print(cosim(kristian, mano))


# In[21]:


print(cosim(tom, mano))


# In[22]:


print(cosim(kristian, paul))


# In[24]:


print(cosim(paul, mano))


# In[45]:


users = df['Reviewer'].unique()


# In[46]:


users


# In[51]:


for user in users:
    print (user, cosim(wide.loc[user],wide.loc[user]))


# In[106]:


import itertools
subset_list = []
cosim_result = []
for user in range(2, 3):
    for subset in itertools.combinations(users, user):
        print(subset, cosim(wide.loc[subset[0]],wide.loc[subset[1]]))
        subset_list.append(subset)
        cosim_result.append(cosim(wide.loc[subset[0]],wide.loc[subset[1]]))


# In[107]:


cosim_result = pd.DataFrame(cosim_result)


# In[108]:


import numpy as np


# In[109]:


cosim_result['user1'] = np.nan
cosim_result['user2'] = np.nan


# In[110]:


for i in range(0,55):
    cosim_result['user1'][i] = subset_list[i][0]
    cosim_result['user2'][i] = subset_list[i][1]


# In[114]:


cosim_result.rename(columns = {0:'cosim'},inplace=True)


# In[118]:


cosim_result.sort_values(by='cosim',ascending=False,inplace=True)


# In[119]:


cosim_result.head(5)


# In[124]:


names =wide.index


# In[130]:


#alternative solution
result = []
for n1 in names:
    for n2 in names:
        c = cosim(wide.loc[n1],wide.loc[n2])
        result.append((n1,n2,c))


# In[131]:


result


# In[132]:


df= pd.DataFrame(result)


# In[135]:


df2 = df.set_index([0,1])


# In[136]:


#two part heirarchical index
df2


# In[141]:


df2 = df2.unstack()


# In[142]:


df2


# In[159]:


#alternative solution
result = []
for n1 in names:
    for n2 in names:
        c = cosim(wide.loc[n1],wide.loc[n2])
        result.append(c)


# In[160]:


a = np.array(result)


# In[161]:


a.shape


# In[162]:


a = a.reshape((11,11))


# In[163]:


df4 = pd.DataFrame(a, index = names,columns = names)


# In[168]:


df4


# In[164]:


#alternative solution
result = []
for n1 in names:
    row = []
    for n2 in names:
        c = cosim(wide.loc[n1],wide.loc[n2])
        row.append(c)
    result.append(row)
    


# In[165]:


result


# In[166]:


df5 = pd.DataFrame(result, index=names,columns=names)


# In[167]:


df5


# In[169]:


import seaborn as sns


# In[171]:


sns.heatmap(df5,annot=True)


# In[173]:


def cosim2(x, y):
    num = 0.0
    for genre in x.index:
        num += x[genre] * y[genre]
    xsum = math.sqrt(sum(x ** 2))
    ysum = math.sqrt(sum(y ** 2))
    denom = xsum * ysum
    return num/denom
    


# In[174]:


#alternative solution
result = []
for n1 in names:
    row = []
    for n2 in names:
        c = cosim2(wide.loc[n1],wide.loc[n2])
        row.append(c)
    result.append(row)
    


# In[175]:


df6 = pd.DataFrame(result, index=names,columns=names)


# In[ ]:


import matplotlib.pyplot as plt


# In[182]:


sns.heatmap(df5,annot=True)
plt.show()


# In[183]:


sns.heatmap(df6,annot=True)
plt.show()


# In[184]:


from sklearn.metrics.pairwise import cosine_similarity


# In[185]:


cosine_similarity(wide)


# # PCA

# In[189]:


import pandas as pd


# In[215]:


df = pd.read_csv('items_csv.csv',index_col=0)


# In[216]:


df.var()


# In[217]:


df


# In[218]:


from sklearn.decomposition import PCA


# In[219]:


from sklearn.preprocessing import MinMaxScaler


# In[223]:


df - df.mean()


# In[228]:


X = df - df.mean()


# In[225]:


pca = PCA(n_components = 3)


# In[229]:


pca.fit(X)


# In[230]:


Xt = pca.transform(X)


# In[231]:


Xt


# In[232]:


pca.components_


# In[233]:


pca.explained_variance_ratio_


# In[238]:


df2 = pd.DataFrame(Xt,columns = ['size','longness','discness'],index = df.index).round(2)


# In[239]:


df2


# In[241]:


df2.sort_values(by='longness')


# In[242]:


pca = PCA(n_components=2)


# In[244]:


Xt = pca.fit_transform(X)


# In[245]:


Xt


# In[246]:


pca.explained_variance_ratio_


# In[247]:


# explained variance no longer adds up to 1.0


# In[248]:


# fewer dimensions, sacrified some of the explained variance, but end up with smaller # of #s


# In[249]:


# important for finding most important features


# In[250]:


pca.components_


# In[251]:


# first component - height * 0.97 then add to it width * 0.24 + depth * -0.03
#linear combo of first three original features

