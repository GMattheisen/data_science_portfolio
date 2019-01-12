
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import random
from random import randint
from matplotlib import pyplot as plt
from fuzzywuzzy import process
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#importing ratings and movies csv files
PATH_ratings = "ratings.csv"
PATH_movies = "movies.csv"
ratings, movies_ind = pd.read_csv(PATH_ratings), pd.read_csv(PATH_movies)

del ratings['timestamp'] #format ratings dataframe
ratings.set_index(['userId','movieId'], inplace=True)
ratings = ratings.unstack(0)

ratings_count = ratings.count(axis=1) # count the number of ratings for each movie as a measure of popularity
top = pd.DataFrame(ratings_count.sort_values(ascending = False).head(10)) # create a dataframe of the top 10 most popular movies
top.reset_index(inplace=True)
movies_ind.set_index('movieId',inplace=True)
top_movies = movies_ind.loc[top['movieId']]['title'].values # get movie titles from Id
movies_ind.reset_index(inplace=True)
top_movies_index = movies_ind.index[top['movieId']].values

ratings = ratings.fillna(0) # fill unknowns with 0 rating
ratings = ratings["rating"]
ratings = ratings.transpose()


# In[3]:


def get_input(top_movies, top_movies_index):
    print("Of the following movies, rate all that you have seen on a scale of 1-5.")
    print("If you have not seen a movie, rate 0.")
    
    #creates a list of ratings for the prompted movies
    user_input = []
    for i in range(0,10):
        answer = int(input("How would you rate " + str(top_movies[i])))
        if answer > 5:
            answer = 5
        elif answer < 0:
            answer = 0
        user_input.append(answer)
        
    user_ratings = np.zeros(9724) # create an empty array the length of number of movies in system
    # inputs user rating into large array (9,000+ count) at appropriate indexes
    for i in range(0,10):
        user_ratings[top_movies_index[i]] = user_input[i]
    return user_ratings


# In[4]:


user_ratings = get_input(top_movies, top_movies_index)


# In[5]:


def model_ratings_NMF(ratings, movies_ind, n_components):
    R = pd.DataFrame(ratings) # model assumes R ~ PQ'
    model = NMF(n_components=n_components, init='random', random_state=10)
    model.fit(R)

    P = model.components_  # Movie feature
    Q = model.transform(R)  # User features

    query = user_ratings.reshape(1,-1)

    t=model.transform(query)
    
    # prediction movie ratings of input user
    outcome = np.dot(t,P)
    outcome=pd.DataFrame(outcome)
    outcome = outcome.transpose()
    outcome['movieId'] = movies_ind['movieId']
    outcome = outcome.rename(columns={0:'rating'})
    top = outcome.sort_values(by='rating',ascending=False).head(150) # top 100 ratings from predictions list
    
    return top


# In[6]:


top = model_ratings_NMF(ratings, movies_ind, n_components =5)


# In[7]:


# collects titles of the top movie predictions
top_movie_recs = movies_ind.loc[top['movieId']]['title'].values


# In[8]:


#importing genres
PATHG = "movie_genres_years.csv"
movie_genres = pd.read_csv(PATHG)


# In[9]:


genres = movie_genres.columns.values[3:22] # creates list of genres


# In[10]:


# dictionary with keys equal to genre
b,c = {}, {}
for x in genres:
    key = x
    value = ''
    b[key],c[key] = value, value


# In[11]:


# fills keys with list of movies that belong to respective genre
for x in genres:
    li = []
    for id in top['movieId']:
        if id in list(movie_genres.loc[movie_genres[x] == 1]['movieId']):
            li.append(movies_ind[movies_ind['movieId']==id]['title'].values)
    c[x] = li


# In[12]:


#fills keys with random choice in the list of films within a genre
for x in genres:
    if len(c[x])>0:
        b[x] = c[x][randint(0, len(c[x])-1)][0]
    else:
        b[x] = ""


# In[13]:


# add an option for not choosing a genre
genres_for_q = np.append(genres, 'none')


# In[14]:


genre_answer = process.extractOne(input("What genre of film would you like to watch?"),genres_for_q)


# In[15]:


#picks a top movie of the selected genre
for x in genres:
    if genre_answer[0] == x:
        if len(b[x]) == 0:
            print('No ' +x+ ' recommedations')
        else:
            print('We recommend ' + b[x])
            
# if they don't want a specific genre
if genre_answer[0] == 'none':
        Select = top_movie_recs[randint(0, 4)]
        print('We recommend ' + Select)

