
# coding: utf-8

# In[1]:


import os
import pandas as pd
import matplotlib.pyplot as plt

#create a list of files in your path
path = '/Users/glynismattheisen/Desktop/names/'
file_list = os.listdir(path)
file_list


# In[2]:


# remove unwanted files
file_list.remove('.DS_Store')
file_list.remove('.ipynb_checkpoints')
file_list.remove('Day 2- Assignment.ipynb')
file_list.remove('Week 1 Solutions Part 2.ipynb')
file_list.remove('Week 1 Solutions Part 3.ipynb')
file_list.remove('Week 1 Solutions Part 3-PPN.ipynb')


# In[3]:


#confirm all files deleted
file_list


# In[4]:


#create empty list
data_list = []

#cycle through importing all files from file_list and giving appropriate column names
for fi in file_list:
    full_path = path + fi
    
    #load file
    data_df = pd.read_csv(full_path, names = ['name','gender', 'count'])
    #index_col = 0 says take the first column and use it as an index column
    #set index to 'names' column
    #data_df.set_index('name',inplace=True)
    #add a file_name column to data_df
    data_df["file_name"] = fi
    
    #add file into list
    data_list.append(data_df)
    
#show the contents of data_list
data_list


# In[5]:


#combine in a data frame
df = pd.concat(data_list)
df


# In[6]:


df.info()


# In[7]:


#create a column with the years extrace from file names
def get_year (s):
    #slice out the portion of file name contianing the year and convert from string to integer
    return int(s[3:7])

#apply function
df['year'] = df['file_name'].apply(get_year)
df


# In[8]:


#group by name and year
grouped_df=df.groupby(['name','year']).sum()
grouped_df


# Goal 3: Create plots with pandas
#     
# * medium: plot a time series with one name over all years

# In[9]:


#graph name use over time for Aaban
grouped_df.loc['Aaban'].plot.line(x=None, y='count')
plt.title('Name Aaban Over Time')
plt.ylabel('count')
#graph name use over time for Glynis
grouped_df.loc['Glynis'].plot.line(x=None, y='count')
plt.title('Name Glynis Over Time')
plt.xlabel('year')
plt.ylabel('count')


# In[10]:


#shape of df
df.shape


# In[11]:


#create two new dataframes splitting df between boys and girls
girls_df = df[df.gender == 'F']
boys_df = df[df.gender == 'M']


# In[12]:


boys_df


# In[13]:


#confirm that all data is still there: boys_df.shape + girls_df.shape = df.shape
print(girls_df.shape)
print(boys_df.shape)


# In[14]:


#create empty list for unique boy names
boys_unique = []

for i in range (1880,2018):
        #mask for each year in sequence
        mask = boys_df['year'] == i
        #create a new list containing just values from mask = True
        unique_1 = boys_df[mask]
        #determine unique vales within the new list
        unique_1 = unique_1['name'].unique()
        #calculate length of unique list to get # of unique names
        data_1 = [len(unique_1), i]
        #append to boys_unique list
        boys_unique.append(data_1)
        #print to confirm proper calculation being made
        print(data_1)
        #increase value to perform function of next year
        i += 1


# In[15]:


#as above except girls this time
girls_unique = []

for i in range (1880,2018):
        mask = girls_df['year'] == i
        unique_1 = girls_df[mask]
        unique_1 = unique_1['name'].unique()
        data_1 = [len(unique_1), i]
        girls_unique.append(data_1)
        print(data_1)
        i += 1


# In[16]:


#convert to dataframe
boys_unique_df = pd.DataFrame(boys_unique, columns=['# unique names','year']) 
boys_unique_df


# In[17]:


#convert to dataframe
girls_unique_df = pd.DataFrame(girls_unique, columns=['# unique names','year']) 
girls_unique_df


# Goal 3: Create plots with pandas
#     
# * hard: plot the number of distinct boy/girl names over time

# In[18]:


#plot unique boy/girl names over time
import matplotlib.pyplot as plt
plt.plot(girls_unique_df['year'], girls_unique_df['# unique names'],label='girls')
plt.plot(boys_unique_df['year'], boys_unique_df['# unique names'],label='boys')
plt.legend()
plt.title('number of distinct names over time')
plt.ylabel('number of distinct names')
plt.xlabel('year')
plt.show()


# In[19]:


#get just the Glynis data
Glynis_df = df[df.name == 'Glynis']


# In[20]:


Glynis_df.sort_values(by='year')


# In[21]:


#dropping file name column
Glynis_df = Glynis_df.drop('file_name',axis=1)


# In[22]:


Glynis_df


# In[23]:


# X would be year and y would be # of births


# In[24]:


#x list of glynis years
X = Glynis_df[['year']].values


# In[25]:


# y list of glynis values
y = Glynis_df[['count']].values


# In[26]:


from sklearn.linear_model import LinearRegression
import numpy as np


# In[27]:


#create linear regression model
m = LinearRegression()
m


# In[28]:


X.shape, y.shape


# In[29]:


X = np.array(X).reshape(-1,1)


# In[30]:


y = np.array(y)


# In[31]:


X.shape, y.shape


# In[32]:


#fit the data with linear regression model
m.fit(X,y)


# In[33]:


#checking that it works and returns values
m.coef_


# In[34]:


ypred = m.predict(X)


# ### Goal 4: Build a supervised learning model
# 
# * easy: build a linear regression model with scikit-learn for one name

# In[35]:


# plot data v prediction from model
plt.plot(X,y,'bo', label='Glynis per year')
plt.plot(X, ypred,'rx', label ='Predicted Glynis per year')
plt.xlabel('Year')
plt.ylabel('# of Glynis')
plt.legend()
plt.title('Glynis per Year')
plt.show()


# In[36]:


# conclusion: Glynis heading swiftly toward extinction


# In[37]:


#get total number of births ever (for data set)
df['count'].sum()


# In[38]:


#create empty list for total number births per year
births_year = []

for i in Glynis_df['year']:
        #mask for each year in sequence
        mask = df['year'] == i
        #create a new list containing just values from mask = True
        just_year = df[mask]
        #calculate sum of babies for that year and store as data_2
        data_2 = [just_year['count'].sum(), i]
        #append to to births_year
        births_year.append(data_2)
        #print to confirm proper calculation being made
        print(data_2)
        #increase value to perform function of next year
        i += 1


# In[39]:


births_year


# In[40]:


births_year_df = pd.DataFrame(births_year, columns = ['total births','year'])
births_year_df 


# In[41]:


pd.DataFrame(births_year_df).sum()


# In[42]:


births_year_df = births_year_df.sort_values(by='year')
Glynis_df = Glynis_df.sort_values(by='year')
births_year_df.reset_index(inplace=True)
Glynis_df.reset_index(inplace=True)
births_year_df


# In[43]:


Glynis_df


# In[44]:


births_year_df.shape


# In[45]:


Glynis_df['count'].shape


# In[46]:


new = pd.DataFrame(births_year_df['total births'])
new


# In[47]:


Glynis_df['total births'] = new
Glynis_df


# In[48]:


#normalize Glynis to overall number of babies
Glynis_df['norm'] = Glynis_df['count'] / new['total births']


# In[49]:


Glynis_df['norm']


# In[50]:


Glynis_df


# In[51]:


y2 = Glynis_df[['norm']].values


# In[52]:


m.fit(X,y2)


# In[53]:


m.coef_


# In[54]:


y2pred = m.predict(X)


# ### Goal 4: Build a supervised learning model
# 
# * medium: normalize the data by the total number of births

# In[55]:


# plot data v prediction from model
plt.plot(X,y2,'bo', label='Glynis per year')
plt.plot(X, y2pred,'rx', label ='Predicted Glynis per year')
plt.xlabel('Year')
plt.ylabel('# of Glynis')
plt.title('Glynis per Year Normalized')
plt.legend()
plt.ylim(-.00001,0.0001)
plt.show()


# ### Goal 4: Build a supervised learning model
# 
# * hard: improve the fit using a polynomial regression

# In[56]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 


# In[57]:


#set degree of polynomial fit
poly = PolynomialFeatures(degree=5)
#perfrom transformation on X data
X_ = poly.fit_transform(X)
X_


# In[58]:


lg = LinearRegression()

# Fit
lg.fit(X_, y2)

# Obtain coefficients
lg.coef_
y5 = lg.predict(X_)
y5


# In[59]:


plt.plot(X,y2,'bo', label='Glynis per year')
plt.plot(X, y2pred,'rx', label ='Linear Predicted Glynis per year')
plt.plot(X, y5,'go', label ='Polynomial Predicted Glynis per year')
plt.xlabel('Year')
plt.ylabel('# of Glynis')
plt.title('Glynis per Year Normalized')
plt.legend()
plt.ylim(-.00001,0.0002)
plt.show()

