
# coding: utf-8

# In[1]:


import pymongo
from pymongo import MongoClient
import re
from atlasclient.client import Atlas
import pandas as pd
import matplotlib.pyplot as plt
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)
from collections import Counter
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from collections import Counter
import nltk
from nltk.corpus import stopwords
import seaborn as sns


# # Connect

# In[2]:


# connect to database
atlasclient = pymongo.MongoClient()
db = atlasclient.twitter

# look at data
print(db.collections.tweets_labeled.find_one())


# # Mongo Commands

# In[ ]:


# print item where user_favorite_counts equal to 4
print(db.collections.tweets_labeled.find_one({'user_favorites_count': 4}))


# In[ ]:


# print item with hastags 'BigData' and 'DataInnovators'
print(db.collections.tweets_labeled.find_one({'hashtags': ['BigData','DataInnovators']}))


# In[ ]:


# count the number of tweets with 'BigData' hashtag
# unfortunately, this will only count tweets where 'BigData' is the ONLY hashtag
db.collections.tweets_labeled.find({'hashtags': ['BigData']}).count()


# # Download to Notebook

# In[ ]:


sample_tweets = db.collections.tweets_labeled.find()


# In[ ]:


sample_tweets


# In[ ]:


# download all tweets and store in list
tweets = []
for x in sample_tweets:
    tweets.append(x)

# verify number of tweets
len(tweets)


# # Extract Hashtags

# In[ ]:


def extract_hashtags(list_of_tweets):
    # extract hashtags and put into a list
    pattern = "(?<=hashtags': \[)(.*)(?=\], 'media_url')"
    hashtags = []
    for tweet in tweets:
        match = re.findall(pattern, str(tweet))
        match = str(match).split(',')
        for hash_item in match:
            hashtags.append(hash_item)

    # delete empty items
    hashtags = list(filter(lambda x: x!= "['']", hashtags))
    hashtags = list(filter(lambda x: x!= '["]', hashtags))

    # remove unwanted brackets
    clean_hash = []
    for hash_item in hashtags:
        pattern = "\w+"
        match = re.findall(pattern, hash_item)
        clean_hash.append(match)
    
    clean_hash = list(filter(lambda x: x!= [], clean_hash))
    
    for hashtag in clean_hash:
        hashtag[0] = str(hashtag[0]).lower()

    df = pd.DataFrame(clean_hash, columns=['hashtag'])  # convert to dataframe
    df['freq'] = df.groupby('hashtag')['hashtag'].transform('count') # make a frequency column
    dict = df.set_index('hashtag').T.to_dict('list') # Convert to dictionary
    unique = df['hashtag'].unique() # make a separate dataframe of unique hashtag values
    unique = pd.DataFrame(unique)
    unique.rename( columns=({0:'hashtag'}),inplace=True)

    # add frequency values for hashtags
    number = []
    for i in unique['hashtag']:
        number.append(dict[i][0])

    unique['count'] = number

    unique.sort_values(by='count', ascending=False,inplace=True)

    return unique


# In[ ]:


extract_hashtags(tweets)


# # The Easy Way

# In[8]:


atlasclient = pymongo.MongoClient()
db = atlasclient.twitter

tweet_data = pd.DataFrame(list(db.collections.tweets_labeled.find())) # create dataframe of tweets


# In[9]:


# extract only the words and edit them for word frequency
tweets_text = " ".join(line.strip() for line in tweet_data['text'])   # extract all words in text
tweets_text = re.sub("\W", ' ', tweets_text)
stop = stopwords.words('english')
newStopWords = ['https', 'how', 'big', 'co', 'via', 'the', 'I', 'will', 'amp', 'a', 'you', 'we']
stop.extend(newStopWords) # more stopwords to stop word corpus

tokens = [word.lower() for word in tweets_text.split( )] # extract all words
tokens = [words for words in tokens if words not in stop] # just words not in stop corpus


# In[6]:


common_words = Counter(tokens).most_common(40) # count words
common_words_df = pd.DataFrame(common_words, columns = ['Word', 'Count'])


# In[7]:


common_words_df.plot.bar(x='Word',y='Count', figsize=(20,5))


# In[8]:


tweet_tokens = []
def preprocessing(text):
    tokens = [word.lower() for word in text.split( )]
    stop = stopwords.words('english')
    tokens = [token for token in tokens if token not in stop] # remove stopwords
    tokens = [word.lower() for word in tokens]
    
    tokens = [re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)","", word) for word in tokens] # get rid of @name and #s
    tokens = [re.sub("http\S+", "", word) for word in tokens]
    preprocessed_text = ' '.join(tokens)
    tweet_tokens.append(preprocessed_text)


# In[9]:


tweets = tweet_data['text'] # process text
for tweet in tweets:
    preprocessing(tweet)


# In[10]:


tweet_counts = []
for tweet in tweet_tokens: # counts number of times each word used in text
    count = {}
    for word in tweet.split(' '):
        if word not in count:
            count[word] = 1
        else:
            count[word] += 1
    tweet_counts.append(count)


# In[11]:


mini_corpus = pd.DataFrame(tweet_counts)
mini_corpus = mini_corpus.fillna(0)


# In[12]:


N_TOPICS = 2
lda = LatentDirichletAllocation(n_components=N_TOPICS, learning_method='online')
lda.fit(mini_corpus)

topics = pd.DataFrame(lda.components_, columns=mini_corpus.columns)
topics = topics.transpose()

def get_top_words(topic, i):
    t = topic.sort_values(ascending=False).head(15)
    t = pd.DataFrame({f'word_{i}': t.index})
    t['num'] = range(15)
    t.set_index('num', inplace=True)
    return t

c = [get_top_words(topics[i], i) for i in range(N_TOPICS)]
result = pd.concat(c, axis=1)
print(result)


# # More Analysis

# In[10]:


tweet_data.head(5)


# In[11]:


count = tweet_data['_id'].count()


# In[12]:


def get_hashtags(df):
    
    # extract just day, month, year from 'created_at'
    pattern = "(?<=\w\w\w )(.*)(?= \d\d:\d\d:\d\d \+)|(?<=\+0000 )(.*)" 
    #pattern = "(?<=\w\w\w )(.*)(?= \+)|(?<=\+0000 )(.*)" for time included
    m = []
    dates = []
    for x in df['created_at']:
        match = re.findall(pattern,str(x))
        dates.append(str(match[0][0]) + " " + str(match[1][1]))
    
    df['date'] = pd.DataFrame(dates) # create dataframe of dates
    
    date_time = []
    for x in tweet_data['date']:
        date_time.append(pd.to_datetime(x, infer_datetime_format=True)) # convert to date time formatting

    tweet_data['date_time']=pd.DataFrame(date_time) # append to tweet data 
    
    stacked = df.set_index('date_time') # delete everything but hashtags
    del stacked['_id']
    del stacked['created_at']
    del stacked['favorites']
    del stacked['followers']
    del stacked['id']
    del stacked['interesting']
    del stacked['media_type']
    del stacked['media_url']
    del stacked['retweets']
    del stacked['text']
    del stacked['user_favorites_count']
    del stacked['username']
    del stacked['date']
    stacked = pd.DataFrame(stacked.stack()) # organize by day
    stacked.reset_index(inplace=True)
    dates = stacked['date_time'].unique() # collect list of all days in the dataset
    
    del stacked['level_1']
    
    dictionary = {} # make dictionary of dates
    for x in dates:
        dictionary[x] = ''
        
    for date in dates:
        array_hash = np.where(stacked['date_time'] == date) # locate indexes of hastags for each date
        array_hash = array_hash[0]
        hash_day = []
        for x in array_hash: # list all hastags for each days
            for b in stacked[0][x]:
                hash_day.append(b.lower()) # lower case for all hashtags
        dictionary[date] = hash_day # add all kashtags as values for key of date
        
    dictionary2 = {} # create empty dictionary of dates
    for x in dates:
        dictionary2[x] = ''
    
    counts = []
    for key in dictionary: # count the number of times each hashtag appears
        dictionary2[key] = Counter(dictionary[key])
    
    dict2 = pd.DataFrame(dictionary2)
    dict2.fillna(0, inplace=True)
    dict2 = dict2.T
    rankings = dict2.sum()

    rankings = pd.DataFrame(rankings) # total number of times each hastag used across all days
    rankings = rankings.sort_values(by=0, ascending=False)
    rankings_top1000 = rankings.head(1000) # locate top 1000 hashtags
    rankings_top50 = rankings.head(50) # locate top 50 hashtags
    rankings_top1000.reset_index(inplace=True)
    rankings_top50.reset_index(inplace=True)
    rankings_top1000.rename(columns={'index':'hashtag'},inplace=True)
    rankings_top50.rename(columns={'index':'hashtag'},inplace=True)

    short_list =  pd.DataFrame(dict2['0day']) # create dataframe for manipulation
    short_list_2 =  pd.DataFrame(dict2['0day']) # create dataframe for manipulation
    
    for word in rankings_top1000['hashtag']:
        short_list[word] = dict2[word]
        
    for word in rankings_top50['hashtag']:
        short_list_2[word] = dict2[word]
    
    del short_list['0day'] # delete useless key
    del short_list_2['0day'] # delete useless key
    
    short_list.sort_index(inplace=True)
    short_list_2.sort_index(inplace=True)
    
    sums = short_list_2.sum(axis=1)
    short_list_norm = short_list.div(short_list.sum(axis=1), axis=0) # divide by total hashtags that day
    short_list_norm_2 = short_list_2.div(short_list_2.sum(axis=1), axis=0) # divide by total hashtags that day
    
    short_list_norm = short_list_norm.drop(short_list_norm.index[12]) # drop dec 5 = incomplete data
    short_list_norm_2 = short_list_norm_2.drop(short_list_norm_2.index[12]) # drop dec 5 = incomplete data

    short_list_norm = short_list_norm.drop(short_list_norm.index[0]) # drop sept 17 = incomplete data
    short_list_norm_2 = short_list_norm_2.drop(short_list_norm_2.index[0]) # drop sept 17 = incomplete data

        
    plotly.offline.iplot([{
    'x': short_list_norm_2.index,
    'y': short_list_norm_2[col], 'name': col}  for col in short_list_norm_2.columns])
     
    return sums, short_list, short_list_norm 


# In[13]:


sums, short_list, short_list_norm = get_hashtags(tweet_data)


# # Graphing Increases and Decreases in Hashtag Usages

# In[14]:


def graph_change_in_high_values(short_list):
    first_day_preN = pd.DataFrame(short_list.iloc[1]) # first day for comparison before normalization
    last_day_preN = pd.DataFrame(short_list.iloc[-2]) # last day for comparison before normalization
    first_day_T = first_day_preN.T # transform for hashes in column
    last_day_T = last_day_preN.T #transform

    first_day_clean = first_day_preN[first_day_preN >= 5] # select only those hashtags used more than five times
    first_day_clean = first_day_clean .dropna()

    last_day_clean = last_day_preN[last_day_preN >= 5] # select only those hashtags used more than five times
    last_day_clean = last_day_clean .dropna()

    two_days = pd.concat([last_day_clean, first_day_clean],axis=1, join='inner') # join on hashtag axis
    two_days = two_days.T  # transform for dates on column

    two_days_norm = two_days.div(two_days.sum(axis=1), axis=0) # divide by total hashtags that day

    change_uppers = pd.DataFrame((1-(two_days_norm.iloc[0]/two_days_norm.iloc[1]))*100) # create data frame of percent growth or decrease in usage

    decrease_uppers = change_uppers.sort_values(by=0).head(25) # top 25 decreases
    increase_uppers = change_uppers.sort_values(by=0, ascending=False).head(25) # top 25 increases

    # plot
    plt.rcParams.update({'font.size': 22})

    plt.figure(1)
    plt.figure(figsize=(20,20))
    plt.subplot(211)
    plt.plot(increase_uppers, 'go',)
    plt.ylabel('% change')
    plt.title('Biggest Hashtag Increases over Tweego\'s Lifetime')
    plt.xticks(rotation=80)

    plt.subplot(212)
    plt.plot(decrease_uppers, 'ro')
    plt.gca().invert_yaxis()
    plt.ylabel('% change')
    plt.title('Biggest Hashtag Decreases over Tweego\'s Lifetime')
    plt.xticks(rotation=80)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=1)

    plt.show()
    
    return first_day_preN, last_day_preN


# In[15]:


first_day_preN, last_day_preN = graph_change_in_high_values(short_list)


# # Low mentions (<=5) on day 1 to > 10 on day 2

# In[21]:


first_day_zeros = first_day_preN[first_day_preN <= 5] # select only those hashtags used more than five times
first_day_zeros.dropna(inplace=True) # words that appear on last day but not first

last_day_big = last_day_preN[last_day_preN > 10]
last_day_big.dropna(inplace=True) # words that appear on last day but not first

mismatch = pd.concat([first_day_zeros, last_day_big],axis=1, join='inner') # join on hashtag axis
mismatch = mismatch.sort_values(by='2018-12-10 00:00:00',ascending=False)


# In[22]:


mismatch


# In[23]:


def color_font(val):
    color = 'white' if val >= 0 else 'black'
    return 'color: %s' % color


# In[24]:


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "17pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "20pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '20pt')])
]


# In[25]:


mismatch.style.background_gradient(cmap='viridis', low=1, high=0.75, axis=0).set_properties(**{'max-width': '80px', 'font-size': '17pt'}).applymap(color_font).set_table_styles(magnify())

