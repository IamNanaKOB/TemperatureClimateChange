#!/usr/bin/env python
# coding: utf-8

# ### INTRODUCTION

# From the Berkeley Earth Data page, this dataset in made up or temperature recordings from the Earth’s surface.
# 
# The data ranges from November 1st, 1743 to December 1st, 2013. The dataset files used are: 
# 
# GlobalLandTemperaturesByMajorCity
# GlobalLandTemperaturesByState.
# 
# And also Continents data to be integrated to aid in attainment of objectives, which is from Machin github.
# 
# ##### GlobalLandTemperaturesByMajorCity
# This data is made of columns of date, Country, Average temperature, temperature uncertainty,City, Latitude and Longitude.
# 
# #### GlobalLandTemperaturesByState
# This data is made of columns of date, Country, Average temperature, temperature uncertainty,State.
# 
# ###### Continents data
# It contains data of columns name, alpha-2, alpha-3, region, region code, sub-region and ISO code.
# 
# 
# The goals of the undertaking;
# <li>Maximum temperature among the continents.</li>
# <li>Average temperature among the continents.</li>
# <li>The maximum and average temperature recorded by a continent.</li>
# <li>Temperature differences among countries.</li>
# <li>Also, a focus on Italy and the city Rome</li>
# <li>And, other expressiveness including timeseries.</li>
#  

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import scipy
from itertools import product
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load of dataset

df = pd.read_csv('GlobalLandTemperaturesByMajorCity.csv')
forecast_df = df.copy()


# In[3]:


#A look on the cities
city_data = df.drop_duplicates(['City'])
city_data.head()


# In[4]:


#inference on the data

df.shape


# In[5]:


df.columns


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.rename(columns={'dt':'Date','AverageTemperature':'Avg_temp','AverageTemperatureUncertainty':'AvgTempUncertainty','Latitude':'Latitude','Longitude':'Longitude'}, inplace=True)
df.head()


# In[9]:


#Checking for any null values 

df.isnull().values.any()


# In[10]:


df = df.dropna(how='any', axis=0)
df.shape


# In[11]:


#Statistical inference on the data

df.describe()


# In[12]:


df['Date2'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date2'].dt.year

#Group by year
by_year = df.groupby(by =['Year','City','Country','Latitude','Longitude']).mean().reset_index()

world_map = pd.read_csv('continents2.csv') # load the continent data


# In[13]:


world_map.head()


# In[14]:


world_map['Country'] = world_map['name']
world_map = world_map[['Country', 'region','alpha-2', 'alpha-3']]


# In[15]:


import chart_studio.plotly as py
import plotly.express as px
import plotly.graph_objects as go
import colorlover as cl
from plotly.subplots import make_subplots


# In[16]:


data_new = pd.merge(left = by_year, right = world_map, on = 'Country', how = 'left')

#Year specifically chosen 1840 to represnt industrial revolution
data_new = data_new[data_new['Year']>= 1840]

#region, year and city assign
region = data_new.dropna(axis = 0).groupby(by = ['region', 'Year']).mean().reset_index()
df_new = data_new.dropna(axis = 0).groupby(by = ['region', 'Country','Year']).mean().reset_index()
City = data_new.dropna(axis = 0).groupby(by = ['region', 'Country','City','Year', 'Latitude','Longitude']).mean().reset_index()


# In[17]:


#Continents determination

continent = make_subplots(rows=1, cols=2, insets=[{'cell':(1,1), 'l': 0.7, 'b': 0.3}])
continent.update_layout(title="Max and Average Temperatures by Continents", title_font_size = 20, 
                        template = "ggplot2", hovermode = "closest")
continent.update_xaxes(showline=True, linewidth=1, linecolor='gray')
continent.update_yaxes(showline=True, linewidth=1, linecolor='gray')

continent.add_trace(go.Scatter(x= region[region['region']=='Europe']['Year'], y = region[region['region']=='Europe']
                               ['Avg_temp'], name='Europe', marker_color='rgb(120,0,0)'), row=1,col=1)

continent.add_trace(go.Scatter(x= region[region['region']=='Americas']['Year'], y = region[region['region']=='Americas']
                               ['Avg_temp'], name='Americas', marker_color='rgb(200, 100,40)'), row=1,col=1)

continent.add_trace(go.Scatter(x= region[region['region']=='Asia']['Year'], y = region[region['region']=='Asia']
                               ['Avg_temp'], name='Asia', marker_color='rgb(145,205,56)'), row=1,col=1)

continent.add_trace(go.Scatter(x= region[region['region']=='Africa']['Year'], y = region[region['region']=='Africa']
                               ['Avg_temp'], name='Africa', marker_color='rgb(125,145,56)'), row=1,col=1)

continent.add_trace(go.Scatter(x= region[region['region']=='Oceania']['Year'], y = region[region['region']=='Oceania']
                               ['Avg_temp'], name='Oceania', marker_color='rgb(245,105,150)'), row=1,col=1)

leftside = np.round(region.groupby(by = 'region')['Avg_temp'].mean().tolist(), 1)
rightside = np.round(region.groupby(by = 'region')['Avg_temp'].max().tolist(), 1)

continent.add_trace(go.Bar(x = region['region'].unique(), y = region.groupby(by ='region')['Avg_temp'].mean().tolist(),
                   name = 'Average temperature', marker_color ='rgb(3, 121, 110)', text = leftside, textposition = 'auto'),
                row = 1, col =2)

continent.add_trace(go.Bar(x = region['region'].unique(), y = region.groupby(by ='region')['Avg_temp'].max().tolist(),
                   name = 'Maximum temperature', marker_color ='rgb(255, 134, 10)', text = rightside, textposition = 'auto'),
                 row = 1, col =2)


# In[18]:


#Next is the temperature differences among countries , calculated and visualised

meanvalue = df_new.groupby(['Country','region'])['Avg_temp'].mean().reset_index()
maxvalue = df_new.groupby(['Country','region'])['Avg_temp'].max().reset_index()

total = pd.merge(left = meanvalue, right = maxvalue, on = ['Country','region'])
total['diff'] = total['Avg_temp_y'] - total['Avg_temp_x'] 

rank = go.Figure()
rank.update_layout(title='Difference in the Temperature Among Countries', title_font_size = 20, 
                  template = "ggplot2", autosize = False, height = 3000, width = 900)
rank.update_xaxes(showline=True, linewidth=1)
rank.update_yaxes(showline=True, linewidth=1)

sort_diff = total[['Country','region', 'diff']].sort_values(by = 'diff', ascending = True)
rank.add_trace(go.Bar(x = sort_diff['diff'], y = sort_diff['Country'], orientation = 'h', 
                     marker = dict(color='rgb(113,168,131)', line = dict(color='rgb(156,114,250)',width=0.6))))
rank.show()


# In[19]:


world_map = data_new.dropna(axis =0).groupby(by = ['region','Country','Year','alpha-3']).mean().reset_index()

world_map['Avg_temp'] = world_map['Avg_temp'] + 6 #due to negative in values

wmap = px.scatter_geo(world_map, locations='alpha-3', color='region',
                     color_discrete_sequence = ('rgb(154, 65, 17)','rgb(152, 0, 73)', 'rgb(249, 160, 26)', 'rgb(128,255,25)','rgb(255,0, 0)'),
                     hover_name='Country', size="Avg_temp", size_max = 20, opacity = 0.5,
                     animation_frame="Year", projection="natural earth", title='World Map Temperature')

wmap.show()


# ###### I have chosen to isolate Rome and consider the data of that city to be my dataset. But, abit infer look on Italy. 

# In[20]:


#A look at Italy first

italy_country = df[df['Country']=='Italy']
italy_country = italy_country.reset_index()
italy_country = italy_country.drop(columns=['index'])
italy_country.Date = pd.to_datetime(italy_country.Date)


# In[21]:


YEAR = []
MONTH = []
DAY = []
WEEKDAY = []
for i in range(len(italy_country)):
    WEEKDAY.append(italy_country.Date[i].weekday())
    DAY.append(italy_country.Date[i].day)
    MONTH.append(italy_country.Date[i].month)
    YEAR.append(italy_country.Date[i].year)

italy_country['Year'] = YEAR
italy_country['Month'] = MONTH
italy_country['Day'] = DAY 
italy_country['Weekday'] = WEEKDAY

change_year_index = []
change_year = []
year_list = italy_country['Year'].tolist()
for y in range(0,len(year_list)-1):
    if year_list[y]!=year_list[y+1]:
        change_year.append(year_list[y+1])
        change_year_index.append(y+1)
        
italy_country.loc[change_year_index].head()


# In[22]:


#Average Temperature in Italy from !950 to 2013

a1950=italy_country[italy_country.Year>=1950]
plt.figure(figsize=(15,4))
plt.subplot(121)
sns.lineplot(x = a1950["Year"], y = a1950["Avg_temp"])
plt.title("Average Temperature and Year")
plt.xlabel("Year")
plt.ylabel("Average Temperature")
plt.xticks(rotation = 45)
plt.show()


# In[23]:


Rome_data = df[df['City']=='Rome']


# In[24]:


Rome_data = Rome_data.reset_index()


# In[25]:


Rome_data = Rome_data.drop(columns=['index'])


# In[26]:


Rome_data.Date = pd.to_datetime(Rome_data.Date)


# In[27]:


YEAR = []
MONTH = []
DAY = []
WEEKDAY = []
for i in range(len(Rome_data)):
    WEEKDAY.append(Rome_data.Date[i].weekday())
    DAY.append(Rome_data.Date[i].day)
    MONTH.append(Rome_data.Date[i].month)
    YEAR.append(Rome_data.Date[i].year)

Rome_data['Year'] = YEAR
Rome_data['Month'] = MONTH
Rome_data['Day'] = DAY 
Rome_data['Weekday'] = WEEKDAY

change_year_index = []
change_year = []
year_list = Rome_data['Year'].tolist()
for y in range(0,len(year_list)-1):
    if year_list[y]!=year_list[y+1]:
        change_year.append(year_list[y+1])
        change_year_index.append(y+1)
        
Rome_data.loc[change_year_index].head()


# In[28]:


Rome1950 = Rome_data[Rome_data.Year>=1950]
plt.figure(figsize=(15,4))
plt.subplot(121)
sns.lineplot(x = Rome1950["Year"], y = Rome1950["Avg_temp"])
plt.title("Average Temperature and Year")
plt.xlabel("Year")
plt.ylabel("Average Temperature")
plt.xticks(rotation = 45)
plt.show()


# In[29]:


last_year_data = Rome_data[Rome_data.Year>=2010].reset_index().drop(columns=['index'])
P = np.linspace(0,len(last_year_data)-1,5).astype(int)


# In[30]:


def get_timeseries(start_year,end_year):
    last_year_data = Rome_data[(Rome_data.Year>=start_year) & (Rome_data.Year<=end_year)].reset_index().drop(columns=['index'])
    return last_year_data

def plot_timeseries(start_year,end_year):
    last_year_data = get_timeseries(start_year,end_year)
    P = np.linspace(0,len(last_year_data)-1,5).astype(int)
    plt.plot(last_year_data.Avg_temp,marker='.',color='blue')
    plt.xticks(np.arange(0,len(last_year_data),1)[P],last_year_data.Date.loc[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')
   
def plot_from_data(data,time,c='firebrick',with_ticks=True,label=None):
    time = time.tolist()
    data = np.array(data.tolist())
    P = np.linspace(0,len(data)-1,5).astype(int)
    time = np.array(time)
    if label==None:
        plt.plot(data,marker='.',color=c)
    else:
        plt.plot(data,marker='.',color=c,label=label)
    if with_ticks==True:
        plt.xticks(np.arange(0,len(data),1)[P],time[P],rotation=60)
    plt.xlabel('Date (Y/M/D)')
    plt.ylabel('Average Temperature')


# In[31]:


plt.figure(figsize=(20,20))
plt.suptitle('Plotting 4 decades',fontsize=40,color='black')

plt.subplot(2,2,1)
plt.title('Starting year: 1800, Ending Year: 1810',fontsize=15)
plot_timeseries(1800,1810)
plt.subplot(2,2,2)
plt.title('Starting year: 1900, Ending Year: 1910',fontsize=15)
plot_timeseries(1900,1910)
plt.subplot(2,2,3)
plt.title('Starting year: 1950, Ending Year: 1960',fontsize=15)
plot_timeseries(1900,1910)
plt.subplot(2,2,4)
plt.title('Starting year: 2000, Ending Year: 2010',fontsize=15)
plot_timeseries(1900,1910)
plt.tight_layout()


# In[32]:


#Check stationarity

result = adfuller(Rome_data.Avg_temp)
print('ADF Statistic on the entire dataset: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))


# #### Forecasting

# Facebook Prophet was used in the making of the forecast

# In[33]:


forecast_df.head()


# In[34]:


forecast_df = forecast_df.reset_index()
forecast_df = forecast_df.drop(columns=['index'])
forecast_df.dt = pd.to_datetime(forecast_df.dt)


# In[35]:


YEAR = []
MONTH = []
DAY = []
WEEKDAY = []
for i in range(len(forecast_df)):
    WEEKDAY.append(forecast_df.dt[i].weekday())
    DAY.append(forecast_df.dt[i].day)
    MONTH.append(forecast_df.dt[i].month)
    YEAR.append(forecast_df.dt[i].year)

forecast_df['Year'] = YEAR
forecast_df['Month'] = MONTH
forecast_df['Day'] = DAY 
forecast_df['Weekday'] = WEEKDAY

change_year_index = []
change_year = []
year_list = italy_country['Year'].tolist()
for y in range(0,len(year_list)-1):
    if year_list[y]!=year_list[y+1]:
        change_year.append(year_list[y+1])
        change_year_index.append(y+1)
        
forecast_df.loc[change_year_index].head()


# In[36]:



forecast_df=forecast_df[forecast_df.Year>=1950]


# In[37]:


forecast_df = forecast_df.dropna(how='all')
forecast_df = forecast_df.reset_index(drop=True)


# In[38]:


print(forecast_df.dtypes)


# In[41]:


df_prophet = forecast_df[['AverageTemperature', 'dt']]


# In[43]:


from prophet import Prophet

jh = df_prophet.rename(columns={'dt': 'ds', 'AverageTemperature': 'y'})
jh_model = Prophet(interval_width=0.95)
jh_model.fit(jh)


# In[44]:


jh_forecast = jh_model.make_future_dataframe(periods=36, freq='MS')
jh_forecast = jh_model.predict(jh_forecast)
plt.figure(figsize=(18, 6))
jh_model.plot(jh_forecast, xlabel = 'Date', ylabel = 'Temperature')
plt.title('Temperature of the World Rates')


# In[45]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index


# In[46]:


df['Year'] = df.index.year
latest_df = df.loc['1980':'2013']


# In[47]:


#A glance of statistics inference from the data

df.describe()
latest_df.describe()


# #### EXPLORING THE DATA DISTRIBUTION

# Each of the estimates covered above on the data is in a single number to describe the location or variability of the data. It is also useful to explore how the data is distributed overall.

# Boxplots are based on percentiles and give a quick way to visualize the distribution of data.
# The median is shown by the horizontal line in the box.
# The dashed lines, referred to as whiskers, extend from the top and bottom to indicate the range for the bulk of the data.
# 
# 

# In[48]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot(latest_df["Avg_temp"])


# In[49]:


#Density plot considered as smooth histogram

hist_withf = sns.distplot(latest_df["Avg_temp"], bins=10, kde = True)


# ### Next on Global temperature by state

# In[50]:


#Look at the dataset on temperatyre by state

df1 = pd.read_csv('GlobalLandTemperaturesByState.csv')


# In[51]:


df1.head()


# In[52]:


df1.dtypes


# In[53]:


df1.shape


# In[54]:


df1.isnull().sum()


# In[55]:


df1 = df1.dropna(how='any', axis=0)
df1.shape


# In[56]:


df1.rename(columns={'dt':'Date','AverageTemperature':'Avg_temp','AverageTemperatureUncertainty':'Internal_temp'}, inplace=True)
df1.head()


# #### EXPLORING THE DATA DISTRIBUTION

# Each of the estimates covered above on the data is in a single number to describe the location or variability of the data. It is also useful to explore how the data is distributed overall.

# Boxplots are based on percentiles and give a quick way to visualize the distribution of data. The median is shown by the horizontal line in the box. The dashed lines, referred to as whiskers, extend from the top and bottom to indicate the range for the bulk of the data.

# In[57]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot(df1["Avg_temp"])


# In[58]:


#Density plot considered as smooth histogram

hist_withf = sns.distplot(df1["Avg_temp"], bins=10, kde = True)


# In[59]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)
df1.index


# In[60]:


df1.describe()


# In[61]:


df1['Year'] = df1.index.year
df1.head()


# In[62]:


df1.describe()


# In[63]:


latest_df1 = df1.loc['1980':'2013']
latest_df1.head()


# In[64]:


latest_df1[['Country','Avg_temp']].groupby(['Country']).mean().sort_values('Avg_temp')


# In[65]:


resample_df = latest_df1[['Avg_temp']].resample('A').mean()
resample_df.head()


# In[66]:


resample_df.plot(title='Changes in Temperature from 1980-2013',figsize=(8,5))
plt.ylabel('Temperature',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.legend()


# In[67]:


###stationarity

from statsmodels.tsa.stattools import adfuller

print('The Dickey Fuller Test Results: ')
test_data = adfuller(resample_df.iloc[:,0].values, autolag='AIC')
output_data = pd.Series(test_data[0:4], index=['Test Statistic', 'P-value','Lags Used','Number of Observations Used'])
for key, value in test_data[4].items():
    output_data['Critical Value (%s)' %key] = value
print(output_data)


# Here it can be noted that the test statistic is greater than the critical value. therefore have failed to reject the null hypothesis at this point.Hence, time series is not stationary.

# In[68]:


###plot on decompose

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(resample_df, freq=3)

trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid


# In[69]:


plt.subplot(411)
plt.plot(resample_df)
plt.xlabel('Original')
plt.figure(figsize=(6,5))

plt.subplot(412)
plt.plot(trend)
plt.xlabel('Trend')
plt.figure(figsize=(6,5))

plt.subplot(413)
plt.plot(seasonal)
plt.xlabel('Seasonal')
plt.figure(figsize=(6,5))

plt.subplot(414)
plt.plot(residual)
plt.xlabel('Residual')
plt.figure(figsize=(6,5))

plt.tight_layout()


# In[70]:


# transform our data, by using moving average and exponentiall smoothing.

rol_mean = resample_df.rolling(window=3, center=True).mean()

# Exponential weighted mean
ewm = resample_df.ewm(span=3).mean()

# Rolling standard deviation
rol_std = resample_df.rolling(window=3, center=True).std()

# Creation of subplots next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

#Tempearture graph of rolling mean and exponential weighted mean
ax1.plot(resample_df, label='Original')
ax1.plot(rol_mean, label='Rolling Mean')
ax1.plot(ewm, label='Exponential Weighted Mean')
ax1.set_title('Temperature Changes 1980-2013', fontsize=14)
ax1.set_ylabel('Temperature', fontsize=12)
ax1.set_xlabel('Year', fontsize=12)
ax1.legend()

# Temperature graph with rolling STD
ax2.plot(rol_std,label='Rolling STD' )
ax2.set_title('Temperature Changes 1980-2013', fontsize=14)
ax2.set_ylabel('Temperature', fontsize=12)
ax2.set_xlabel('Year', fontsize=12)
ax2.legend()

plt.tight_layout()
plt.show()


# In[71]:


# Now reapply the Dickey fuller test to check the hypothesis

rol_mean.dropna(inplace=True)
ewm.dropna(inplace=True)

print('Dickey Fuller Test for the rolling mean')
df_test = adfuller(rol_mean.iloc[:,0].values, autolag='AIC')
df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'P-value','Lags Used','Number of Observations Used'])
for key, value in df_test[4].items():
    df_output['Critical Value (%s)' %key] = value
print(df_output)
print('-----------------------')
print('Dickey Fuller Test for the Exponential weighted mean')
df_test = adfuller(ewm.iloc[:,0].values, autolag='AIC')
df_output = pd.Series(df_test[0:4], index=['Test Statistic', 'P-value','Lags Used','Number of Observations Used'])
for key, value in df_test[4].items():
    df_output['Critical Value (%s)' %key] = value
print(df_output)

#It can be still noted that Test statistic is greater than the critical value, therefore failed to reject the null hypothesis


# In[72]:


# Using differencing to remove the rolling mean or exponential from the original time series.

diff_rol_mean = resample_df - rol_mean
diff_rol_mean.dropna(inplace=True)
diff_rol_mean.head()


# In[73]:


diff_ewm = resample_df - ewm
diff_ewm.dropna(inplace=True)
diff_ewm.head()


# In[74]:


#Next is the plotting of the difference graph

df_rol_mean_diff = diff_rol_mean.rolling(window=3, center=True).mean()

# Exponential weighted mean
df_ewm_mean_diff = diff_ewm.ewm(span=3).mean()

# Creation of subplots next to each other
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))

#Tempearture graph of rolling mean
ax1.plot(diff_rol_mean, label='Original')
ax1.plot(df_rol_mean_diff, label='Rolling Mean')
ax1.set_title('Temperature Changes !950-2013', fontsize=14)
ax1.set_ylabel('Temperature', fontsize=12)
ax1.set_xlabel('Year', fontsize=12)
ax1.legend()

# Temperature graph with exponential weighted mean
ax2.plot(diff_ewm,label='Original' )
ax2.plot(df_ewm_mean_diff,label='Exponential Weighted Mean' )
ax2.set_title('Temperature Changes 1950-2013', fontsize=14)
ax2.set_ylabel('Temperature', fontsize=12)
ax2.set_xlabel('Year', fontsize=12)
ax2.legend()

plt.tight_layout()


# In[75]:


# Apply Dickey Fuller Test to check hypothesis
print('The Dickey Fuller Test for the Original and Rolling Mean: ')
test_data = adfuller(diff_rol_mean.iloc[:,0].values, autolag='AIC')
output_data = pd.Series(test_data[0:4], index=['Test Statistic', 'P-value','Lags Used','Number of Observations Used'])
for key, value in test_data[4].items():
    output_data['Critical Value (%s)' %key] = value
print(output_data)

print('The Dickey Fuller Test for the Original and Exponential Weighted Mean: ')
test_data = adfuller(diff_ewm.iloc[:,0].values, autolag='AIC')
output_data = pd.Series(test_data[0:4], index=['Test Statistic', 'P-value','Lags Used','Number of Observations Used'])
for key, value in test_data[4].items():
    output_data['Critical Value (%s)' %key] = value
print(output_data)


# Here it can be noted that the test statistics is less than the critical, therefore can reject the null hypoythsis, and confident the data is stationary.

# In[76]:


#Autocorrelation and Partial

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from matplotlib import pyplot

#Plotting the graphs
pyplot.figure(figsize=(10,5))
pyplot.subplot(211)
plot_acf(resample_df, ax=pyplot.gca())
pyplot.subplot(212)
plot_pacf(resample_df, ax=pyplot.gca())
pyplot.show()

