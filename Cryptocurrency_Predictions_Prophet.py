
# coding: utf-8

# # Cryptocurrencies

# # Goal 1: Retrieve 1 cryptocurrency via API

# In[1]:


import requests
import pandas as pd


# In[2]:


base = "https://api.coingecko.com/api/v3/"
url = base + "coins/list"


# In[3]:


result = requests.get(url)


# In[4]:


result.text[:100]


# In[5]:


j = result.json()


# In[6]:


df = pd.DataFrame(j)


# In[7]:


df.head(2)


# In[8]:


base = "http://api.coingecko.com/api/v3/coins/"
coin = "bitcoin"
url = base + coin + "/market_chart?vs_currency=usd&days=300"


# In[9]:


result = requests.get(url)


# In[10]:


result


# In[11]:


j = result.json()


# In[12]:


j


# ### data presented as a dictionary

# In[13]:


j.keys()


# In[14]:


j_df = pd.DataFrame(j['prices'], columns = ['date','price'])


# In[15]:


j_df.head()


# In[16]:


# convert to date time formatting
j_df['date'] = pd.to_datetime(j_df['date'],unit = 'ms').dt.round('1min')


# In[17]:


import matplotlib.pyplot as plt


# In[18]:


j_df.plot('date','price')
plt.show()


# In[19]:


a = j_df.head(10)


# In[20]:


# now you have a y for sklearn


# In[21]:


# create a seperate column for x


# In[22]:


#shifts a down by 1 replace by NaN
a['x-1'] = a['price'].shift(1)


# In[23]:


a['x-2'] = a['price'].shift(2)
a['x-3'] = a['price'].shift(3)


# In[24]:


a.head(20)


# In[25]:


#target value is now associated with period 1, 2, 3 before


# In[26]:


a.dropna(inplace=True)


# # Simple linear regression

# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


X = a[['x-1','x-2','x-3']]
X


# In[29]:


m = LinearRegression()


# In[30]:


y = a['price']
y


# In[31]:


m.fit(X,y)


# In[32]:


m.score(X,y)


# In[33]:


ypred = m.predict(X)


# In[34]:


a['ypred'] = pd.DataFrame(ypred)


# In[35]:


a.plot('date','ypred')
a.plot('date','price')
plt.show()


# # Goal 2: Plot an autocorrelation

# In[36]:


from statsmodels.graphics.tsaplots import plot_acf


# In[37]:


j_df.head(2)


# In[38]:


j_df.set_index(['date'],inplace=True)


# In[39]:


plot_acf(j_df,  alpha=.05, use_vlines=True)
# Confidence intervals are drawn as a cone.
#By default set to a 95% confidence interval, suggesting that correlation values outside cone are very likely a correlation and not a statistical fluke


# In[40]:


b= round(j_df['price'],2)
import matplotlib.pyplot as plot


# In[41]:


plot.acorr(b)
plt.show()


# # Make a list of 10 coins

# In[42]:


url_list = "https://api.coingecko.com/api/v3/coins/list"
coin_list = requests.get(url_list)
coin_list = coin_list.json()


# In[43]:


coin_list = pd.DataFrame(coin_list)


# In[44]:


coins = coin_list['id'][1:11]


# In[45]:


coins


# # Automate my import

# In[46]:


for i in coins:
    print(i)


# In[47]:


data_list = []

for i in coins:
    base = "http://api.coingecko.com/api/v3/coins/"
    url = base + i + "/market_chart?vs_currency=usd&days=300"
    result = requests.get(url)
    data = result.json()
    data_df = pd.DataFrame(data['prices'], columns = ['date','price'])
    data_df["file_name"] = i
    data_list.append(data_df)


# In[48]:


data_list


# In[49]:


data_df = pd.concat(data_list)


# In[50]:


data_df.info()


# In[51]:


data_df['date'] = pd.to_datetime(data_df['date'],unit = 'ms').dt.round('1min')


# In[52]:


data_df.rename(columns={'file_name': 'coin'},inplace=True)


# In[53]:


data_df.set_index(['coin'])


# In[54]:


coins


# # Goal 1: Retrieve 10 cryptocurrency timelines

# In[55]:


import matplotlib.pyplot as plt
for i in coins:
    individual = data_df[data_df.coin == i]
    plt.plot(individual['date'], individual['price'], label = i)
    plt.legend()
    plt.title('coins over time')
    plt.ylabel('price')
    plt.xlabel('year')
    plt.xticks(rotation='vertical')


# In[56]:


from numpy import convolve


# In[57]:


def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


# # Goal 2: Plot a rolling average

# In[58]:


import matplotlib.pyplot as plt
import numpy as np

for i in coins:
    individual = data_df[data_df.coin == i]
    yMA = movingaverage(individual['price'],5)
    plt.plot(individual['date'][len(individual['date'])-len(yMA):], yMA, label = i)
    plt.legend()
    plt.title('coins over time')
    plt.ylabel('price')
    plt.xlabel('year')
    plt.xticks(rotation='vertical')


# # Goal 1: Use Quandl to retrieve other financial data

# In[59]:


base = "https://api.coingecko.com/api/v3/coins/"
url = base + "omni/market_chart?vs_currency=usd&days=1825"
result = requests.get(url)


# In[60]:


j = result.json()


# In[61]:


df = pd.DataFrame(j['prices'], columns = ['date','price'])


# In[62]:


df['date'] = pd.to_datetime(df['date'],unit = 'ms').dt.round('1min')


# In[63]:


df.sort_values(by='date')


# In[66]:


cd ..


# In[67]:


JP_Morgan = pd.read_csv('EOD-JPM.csv')


# In[68]:


import plotly.plotly as py
import plotly.graph_objs as go


# In[69]:


import pandas_datareader as web
import plotly
plotly.tools.set_credentials_file(username='gmattheisen', api_key='dwBWwrEaic7U3SHHZNyl')


# In[70]:


JP_Morgan.head(3)


# In[71]:


trace = go.Candlestick(x=JP_Morgan.Date,
                       open=JP_Morgan.Open,
                       high=JP_Morgan.High,
                       low=JP_Morgan.Low,
                       close=JP_Morgan.Close)


# In[72]:


data = [trace]


# # Create a candlestick plot

# In[73]:


py.iplot(data, filename='simple_candlestick')


# In[74]:


data_df.head(6)


# In[75]:


feathercoin = data_df[data_df.coin == 'feathercoin']


# In[76]:


del feathercoin['coin']
feathercoin.head(5)


# In[77]:


feathercoin.plot('date','price')
plt.title('Feathercoin over time')
plt.show()


# In[78]:


from sklearn.linear_model import LinearRegression


# In[79]:


LinReg = LinearRegression()


# In[80]:


import datetime as dt
feathercoin['date']=feathercoin['date'].map(dt.datetime.toordinal)


# In[81]:


feathercoin.head(5)


# In[82]:


import numpy as np
x = feathercoin['date']
X = np.array(x)
X = X.reshape(-1,1)


# In[83]:


y = feathercoin['price']
y = np.array(y)


# In[91]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)


# In[92]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[98]:


LinReg.fit(X_train,y_train)
print('Train score: ' + str(LinReg.score(X_train,y_train)))
print('Test score: ' + str(LinReg.score(X_test,y_test)))


# In[95]:


ypred = LinReg.predict(X_test)


# In[88]:


x_g = feathercoin['date']
y_g= feathercoin['price']


# In[293]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.legend(['LinReg'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Polynomials

# In[165]:


from sklearn.preprocessing import PolynomialFeatures


# In[166]:


PolyReg = PolynomialFeatures(degree = 4)


# In[167]:


Xpoly = PolyReg.fit_transform(X_train)


# In[168]:


PolyReg.fit(Xpoly, y_train)


# In[169]:


LinReg2 = LinearRegression()
LinReg2.fit(Xpoly, y_train)


# In[170]:


LinReg2.score(Xpoly,y_train)


# In[171]:


ypred_poly = LinReg2.predict(PolyReg.fit_transform(X_test))


# In[292]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.legend(['LinReg','PolyReg'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Lasso (L1)

# In[173]:


from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler


# In[174]:


scaler = MinMaxScaler()


# In[175]:


scaled_x = scaler.fit_transform(Xpoly)


# In[234]:


Lasso_fit = Lasso(alpha = 0.001).fit(scaled_x,y_train)


# In[235]:


Lasso_fit.score(scaled_x,y_train)


# In[236]:


ypred_poly_lasso = Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X_test)))


# In[237]:


#ypred_Lasso = Lasso_fit.predict(scaled_x)


# In[291]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.plot(X, Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'g--')
plt.legend(['LinReg','PolyReg','Lasso'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Ridge (L2)

# In[249]:


Rid_fit = Ridge(alpha = 0.0000001).fit(scaled_x,y_train)


# In[250]:


Rid_fit.score(scaled_x,y_train)


# In[251]:


ypred_Rid = Rid_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X_test)))


# In[285]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.plot(X, Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'g--')
plt.plot(X,Rid_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'k:')
plt.legend(['LinReg','PolyReg','Lasso','Ridge'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Param Alpha

# In[255]:


coeff = []
param = np.linspace(0.0000001,100,20)

for i in np.linspace(0.0000001,100,20):
    Rid_fit = Ridge(alpha = i).fit(scaled_x,y_train)
    coeff.append(Rid_fit.score(scaled_x,y_train))
    i += 1


# In[256]:


plt.plot(param,coeff)
plt.ylabel('coef')
plt.title('Accuracy over Regularization strength -Ridge')
plt.xlabel('param')
plt.show()


# In[257]:


coeff2 = []
param2 = np.linspace(0.0000000001,0.01,10)

for i in np.linspace(0.0000000001,0.01,10):
    Lasso_fit = Lasso(alpha = i).fit(scaled_x,y_train)
    coeff2.append(Lasso_fit.score(scaled_x,y_train))
    i += 1


# In[258]:


plt.plot(param2,coeff2)
plt.ylabel('coef')
plt.title('Accuracy over Regularization strength -Lasso')
plt.xlabel('param')
plt.show()


# # Prophet

# In[260]:


from fbprophet import Prophet


# In[261]:


feathercoin_new = data_df[data_df.coin == 'feathercoin']


# In[262]:


feathercoin_new.head(5)


# In[263]:


del feathercoin_new['coin']


# In[264]:


feathercoin_new=feathercoin_new.rename(columns={'date':'ds', 'price':'y'})


# In[265]:


feathercoin_new.set_index('ds').y.plot()


# In[266]:


model = Prophet(weekly_seasonality=True)
model.fit(feathercoin_new)


# In[267]:


future = model.make_future_dataframe(periods=24,freq='W')
future.tail()


# In[268]:


forecast = model.predict(future)


# In[269]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[270]:


model.plot(forecast)
plt.title('Feathercoin Forecast')
plt.ylabel('price')
plt.xlabel('date');


# # Bitcoin

# In[294]:


bitcoin = j_df.copy()


# In[295]:


bitcoin.head(5)


# In[299]:


bitcoin.reset_index(inplace=True)


# In[300]:


bitcoin['date']=bitcoin['date'].map(dt.datetime.toordinal)


# In[301]:


x = bitcoin['date']
X = np.array(x)
X = X.reshape(-1,1)


# In[302]:


y = bitcoin['price']
y = np.array(y)


# In[303]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)


# In[304]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[305]:


LinReg.fit(X_train,y_train)
print('Train score: ' + str(LinReg.score(X_train,y_train)))
print('Test score: ' + str(LinReg.score(X_test,y_test)))


# In[306]:


ypred = LinReg.predict(X_test)


# In[308]:


x_g = bitcoin['date']
y_g= bitcoin['price']


# In[309]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.legend(['LinReg'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Polynomials

# In[310]:


PolyReg = PolynomialFeatures(degree = 4)


# In[311]:


Xpoly = PolyReg.fit_transform(X_train)


# In[312]:


PolyReg.fit(Xpoly, y_train)


# In[313]:


LinReg2 = LinearRegression()
LinReg2.fit(Xpoly, y_train)


# In[314]:


LinReg2.score(Xpoly,y_train)


# In[315]:


ypred_poly = LinReg2.predict(PolyReg.fit_transform(X_test))


# In[316]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.legend(['LinReg','PolyReg'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Lasso (L1)

# In[318]:


scaler = MinMaxScaler()


# In[319]:


scaled_x = scaler.fit_transform(Xpoly)


# In[320]:


Lasso_fit = Lasso(alpha = 0.001).fit(scaled_x,y_train)


# In[321]:


Lasso_fit.score(scaled_x,y_train)


# In[323]:


ypred_poly_lasso = Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X_test)))


# In[324]:


#ypred_Lasso = Lasso_fit.predict(scaled_x)


# In[325]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.plot(X, Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'g--')
plt.legend(['LinReg','PolyReg','Lasso'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Ridge (L2)

# In[326]:


Rid_fit = Ridge(alpha = 0.0000001).fit(scaled_x,y_train)


# In[327]:


Rid_fit.score(scaled_x,y_train)


# In[328]:


ypred_Rid = Rid_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X_test)))


# In[329]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.plot(X, Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'g--')
plt.plot(X,Rid_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'k:')
plt.legend(['LinReg','PolyReg','Lasso','Ridge'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # More detailed prediction

# In[330]:


bitcoin = j_df.copy()
bitcoin.head(2)


# In[331]:


bitcoin.reset_index(inplace=True)
bitcoin['date']=bitcoin['date'].map(dt.datetime.toordinal)
bitcoin.head(3)


# In[332]:


bitcoin['p-mean'] = bitcoin['price'].rolling(10).mean()


# In[333]:


bitcoin = bitcoin.dropna()


# In[334]:


x = bitcoin['date']
X = np.array(x)
X = X.reshape(-1,1)


# In[337]:


y = bitcoin['p-mean']
y = np.array(y)


# In[338]:


X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)


# In[341]:


LinReg= LinearRegression()
LinReg.fit(X_train,y_train)
print('Train score: ' + str(LinReg.score(X_train,y_train)))
print('Test score: ' + str(LinReg.score(X_test,y_test)))


# In[342]:


ypred = LinReg.predict(X_test)


# In[343]:


x_g = bitcoin['date']
y_g= bitcoin['price']


# In[344]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.legend(['LinReg'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Polynomial

# In[362]:


PolyReg = PolynomialFeatures(degree =10)


# In[363]:


Xpoly = PolyReg.fit_transform(X_train)


# In[364]:


PolyReg.fit(Xpoly, y_train)


# In[365]:


LinReg2 = LinearRegression()
LinReg2.fit(Xpoly, y_train)


# In[366]:


LinReg2.score(Xpoly,y_train)


# In[367]:


ypred_poly = LinReg2.predict(PolyReg.fit_transform(X_test))


# In[368]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.legend(['LinReg','PolyReg'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Lasso

# In[369]:


scaler = MinMaxScaler()


# In[370]:


scaled_x = scaler.fit_transform(Xpoly)


# In[371]:


Lasso_fit = Lasso(alpha = 0.001).fit(scaled_x,y_train)


# In[372]:


Lasso_fit.score(scaled_x,y_train)


# In[373]:


ypred_poly_lasso = Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X_test)))


# In[374]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.plot(X, Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'g--')
plt.legend(['LinReg','PolyReg','Lasso'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Ridge

# In[375]:


Rid_fit = Ridge(alpha = 0.0000001).fit(scaled_x,y_train)


# In[376]:


Rid_fit.score(scaled_x,y_train)


# In[377]:


ypred_Rid = Rid_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X_test)))


# In[378]:


plt.scatter(x_g,y_g,color = 'b',s = 0.15)
plt.plot(X_test,ypred, 'r--')
plt.plot(X,LinReg2.predict(PolyReg.fit_transform(X)),'b--')
plt.plot(X, Lasso_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'g--')
plt.plot(X,Rid_fit.predict(scaler.fit_transform(PolyReg.fit_transform(X))),'k:')
plt.legend(['LinReg','PolyReg','Lasso','Ridge'])
plt.ylabel('price')
plt.xlabel('date')
plt.show()


# # Prophet

# In[417]:


bitcoin = j_df.copy()


# In[418]:


bitcoin.reset_index(inplace=True)
bitcoin.head(3)


# In[419]:


bitcoin['date'] = pd.to_datetime(bitcoin['date'])


# In[420]:


bitcoin.head(3)


# In[425]:


bitcoin_new=bitcoin.rename(columns={'date':'ds', 'price':'y'})


# In[426]:


bitcoin_new.head(3)


# In[427]:


model = Prophet(weekly_seasonality=True)
model.fit(bitcoin_new)


# In[428]:


future = model.make_future_dataframe(periods=24,freq='W')
future.tail()


# In[429]:


forecast = model.predict(future)


# In[430]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[432]:


model.plot(forecast)
plt.title('Bitcoin Forecast')
plt.ylabel('price')
plt.xlabel('date');


# # MORE

# In[450]:


bitcoin = j_df.copy()
bitcoin.head(2)


# In[451]:


bitcoin.reset_index(inplace=True)


# In[454]:


bitcoin['p-mean'] = bitcoin['price'].rolling(10).mean()

bitcoin = bitcoin.dropna()


# In[455]:


bitcoin.head(4)


# In[456]:


del bitcoin['price']


# In[457]:


bitcoin_new=bitcoin.rename(columns={'date':'ds', 'p-mean':'y'})


# In[458]:


bitcoin_new.head(3)


# In[459]:


model = Prophet(weekly_seasonality=True)
model.fit(bitcoin_new)


# In[460]:


future = model.make_future_dataframe(periods=24,freq='W')
future.tail()


# In[461]:


forecast = model.predict(future)


# In[462]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[463]:


model.plot(forecast)
plt.title('Bitcoin Forecast')
plt.ylabel('price')
plt.xlabel('date');


# In[ ]:


291	2018-10-19 12:38:00	6456.351634	5463.246658	7493.136280
292	2018-10-21 12:38:00	6473.041813	5446.126732	7478.723946

