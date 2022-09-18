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


# In[33]:


temp = get_timeseries(1992,2013)
N = len(temp.Avg_temp)
split = 0.95
training_size = round(split*N)
test_size = round((1-split)*N)
series = temp.Avg_temp[:training_size]
date = temp.Date[:training_size]
test_series = temp.Avg_temp[len(date)-1:len(temp)]
test_date = temp.Date[len(date)-1:len(temp)]
#test_date = test_date.reset_index().Date
#test_series = test_series.reset_index().Avg_temp


# In[34]:


def optimize_ARIMA(order_list, exog):
    """
        Return dataframe with parameters and corresponding AIC
        
        order_list - list with (p, d, q) tuples
        exog - the exogenous variable
    """
    
    results = []
    
    for order in tqdm_notebook(order_list):
        #try: 
        model = SARIMAX(exog, order=order).fit(disp=-1)
    #except:
    #        continue
            
        aic = model.aic
        results.append([order, model.aic])
    #print(results)
    result_df = pd.DataFrame(results)
    result_df.columns = ['(p, d, q)', 'AIC']
    #Sort in ascending order, lower AIC is better
    result_df = result_df.sort_values(by='AIC', ascending=True).reset_index(drop=True)
    
    return result_df


# In[35]:


from itertools import product
from tqdm import tqdm_notebook
from statsmodels.tsa.statespace.sarimax import SARIMAX

ps = range(0, 10, 1)
d = 0
qs = range(0, 10, 1)

# Create a list with all possible combination of parameters
parameters = product(ps, qs)
parameters_list = list(parameters)

order_list = []

for each in parameters_list:
    each = list(each)
    each.insert(1, d)
    each = tuple(each)
    order_list.append(each)
    
result_d_0 = optimize_ARIMA(order_list, exog = series)


# In[36]:


result_d_0.head()


# In[37]:


ps = range(0, 10, 1)
d = 1
qs = range(0, 10, 1)

# Create a list with all possible combination of parameters
parameters = product(ps, qs)
parameters_list = list(parameters)

order_list = []

for each in parameters_list:
    each = list(each)
    each.insert(1, d)
    each = tuple(each)
    order_list.append(each)
    
result_d_1 = optimize_ARIMA(order_list, exog = series)

result_d_1


# In[38]:


result_d_1.head()


# In[39]:


final_result = result_d_0.append(result_d_1)


# In[40]:


best_models = final_result.sort_values(by='AIC', ascending=True).reset_index(drop=True).head()
best_model_params_0 = best_models[best_models.columns[0]][0]
best_model_params_1 = best_models[best_models.columns[0]][1]


# In[41]:


best_model_0 = SARIMAX(series, order=best_model_params_0).fit()
print(best_model_0.summary())
best_model_1 = SARIMAX(series, order=best_model_params_1).fit()
print(best_model_1.summary())


# In[42]:


best_model_1.plot_diagnostics(figsize=(15,12))
plt.show()


# Forecasting

# In[43]:


fore_l= test_size-1
forecast = best_model_0.get_prediction(start=training_size, end=training_size+fore_l)
forec = forecast.predicted_mean
ci = forecast.conf_int(alpha=0.05)

s_forecast = best_model_1.get_prediction(start=training_size, end=training_size+fore_l)
s_forec = s_forecast.predicted_mean
s_ci = forecast.conf_int(alpha=0.05)


# In[44]:


error_test=Rome_data.loc[test_date[1:].index.tolist()].AvgTempUncertainty
index_test = test_date[1:].index.tolist()
test_set = test_series[1:]


# In[45]:


lower_test = test_set-error_test
upper_test = test_set+error_test


# In[46]:


fig, ax = plt.subplots(figsize=(16,8), dpi=300)
x0 = Rome_data.Avg_temp.index[0:training_size]
x1=Rome_data.Avg_temp.index[training_size:training_size+fore_l+1]
#ax.fill_between(forec, ci['Lower Load'], ci['Upper Load'])
plt.plot(x0, Rome_data.Avg_temp[0:training_size],'k', label = 'Average Temperature')

plt.plot(Rome_data.Avg_temp[training_size:training_size+fore_l], '.k', label = 'Actual')

forec = pd.DataFrame(forec, columns=['f'], index = x1)
#forec.f.plot(ax=ax,color = 'Darkorange',label = 'Forecast (d = 2)')
#ax.fill_between(x1, ci['lower AverageTemperature'], ci['upper AverageTemperature'],alpha=0.2, label = 'Confidence interval (95%)',color='grey')

s_forec = pd.DataFrame(s_forec, columns=['f'], index = x1)
s_forec.f.plot(ax=ax,color = 'firebrick',label = 'Forecast  (2,1,6) model')
ax.fill_between(x1, s_ci['lower Avg_temp'], s_ci['upper Avg_temp'],alpha=0.2, label = 'Confidence interval (95%)',color='grey')


plt.legend(loc = 'upper left')
plt.xlim(80,)
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
plt.show()


# In[47]:


#plt.plot(forec)
plt.figure(figsize=(12,12))
plt.subplot(2,1,1)
plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plt.plot(test_set,marker='.',label="Actual",color='navy')
plt.plot(forec,marker='d',label="Forecast",color='firebrick')
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')
plt.fill_between(x1, s_ci['lower Avg_temp'], s_ci['upper Avg_temp'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
plt.legend()
plt.subplot(2,1,2)
plt.fill_between(x1, lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plt.plot(test_set,marker='.',label="Actual",color='navy')
plt.plot(s_forec,marker='d',label="Forecast",color='firebrick')
plt.fill_between(x1, ci['lower Avg_temp'], ci['upper Avg_temp'],alpha=0.3, label = 'Confidence inerval (95%)',color='firebrick')
plt.legend()
plt.xlabel('Index Datapoint')
plt.ylabel('Temperature')


# In[48]:


plt.fill_between(np.arange(0,len(test_set),1), lower_test, upper_test,alpha=0.2, label = 'Test set error range',color='navy')
plot_from_data(test_set,test_date,c='navy',label='Actual')
plot_from_data(forec['f'],test_date,c='firebrick',label='Forecast')
plt.legend(loc=2)


# In[49]:


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.index


# In[50]:


df['Year'] = df.index.year
latest_df = df.loc['1980':'2013']


# In[51]:


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

# In[52]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot(latest_df["Avg_temp"])


# In[53]:


#Density plot considered as smooth histogram

hist_withf = sns.distplot(latest_df["Avg_temp"], bins=10, kde = True)


# ### Next on Global temperature by state

# In[54]:


#Look at the dataset on temperatyre by state

df1 = pd.read_csv('GlobalLandTemperaturesByState.csv')


# In[55]:


df1.head()


# In[56]:


df1.dtypes


# In[57]:


df1.shape


# In[58]:


df1.isnull().sum()


# In[59]:


df1 = df1.dropna(how='any', axis=0)
df1.shape


# In[60]:


df1.rename(columns={'dt':'Date','AverageTemperature':'Avg_temp','AverageTemperatureUncertainty':'Internal_temp'}, inplace=True)
df1.head()


# #### EXPLORING THE DATA DISTRIBUTION

# Each of the estimates covered above on the data is in a single number to describe the location or variability of the data. It is also useful to explore how the data is distributed overall.

# Boxplots are based on percentiles and give a quick way to visualize the distribution of data. The median is shown by the horizontal line in the box. The dashed lines, referred to as whiskers, extend from the top and bottom to indicate the range for the bulk of the data.

# In[61]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.boxplot(df1["Avg_temp"])


# In[62]:


#Density plot considered as smooth histogram

hist_withf = sns.distplot(df1["Avg_temp"], bins=10, kde = True)


# In[63]:


df1['Date'] = pd.to_datetime(df1['Date'])
df1.set_index('Date', inplace=True)
df1.index


# In[64]:


df1.describe()


# In[65]:


df1['Year'] = df1.index.year
df1.head()


# In[66]:


df1.describe()


# In[67]:


latest_df1 = df1.loc['1980':'2013']
latest_df1.head()


# In[68]:


latest_df1[['Country','Avg_temp']].groupby(['Country']).mean().sort_values('Avg_temp')


# In[69]:


resample_df = latest_df1[['Avg_temp']].resample('A').mean()
resample_df.head()


# In[70]:


resample_df.plot(title='Changes in Temperature from 1980-2013',figsize=(8,5))
plt.ylabel('Temperature',fontsize=12)
plt.xlabel('Year',fontsize=12)
plt.legend()


# In[71]:


###stationarity

from statsmodels.tsa.stattools import adfuller

print('The Dickey Fuller Test Results: ')
test_data = adfuller(resample_df.iloc[:,0].values, autolag='AIC')
output_data = pd.Series(test_data[0:4], index=['Test Statistic', 'P-value','Lags Used','Number of Observations Used'])
for key, value in test_data[4].items():
    output_data['Critical Value (%s)' %key] = value
print(output_data)


# Here it can be noted that the test statistic is greater than the critical value. therefore have failed to reject the null hypothesis at this point.Hence, time series is not stationary.

# In[72]:


###plot on decompose

from statsmodels.tsa.seasonal import seasonal_decompose

decomp = seasonal_decompose(resample_df, freq=3)

trend = decomp.trend
seasonal = decomp.seasonal
residual = decomp.resid


# In[73]:


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


# In[74]:


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


# In[75]:


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


# In[76]:


# Using differencing to remove the rolling mean or exponential from the original time series.

diff_rol_mean = resample_df - rol_mean
diff_rol_mean.dropna(inplace=True)
diff_rol_mean.head()


# In[77]:


diff_ewm = resample_df - ewm
diff_ewm.dropna(inplace=True)
diff_ewm.head()


# In[78]:


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


# In[79]:


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

# In[80]:


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

