#!/usr/bin/env python
# coding: utf-8

# ##### Import required libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import datatable as dt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


datatable_df = dt.fread("household_power_consumption.txt")


# ##### Convert Datatable into Pandas Dataframe

# In[3]:


df = datatable_df.to_pandas()


# In[4]:


df.head()


# In[5]:


df.columns


# In[6]:


df.shape


# ## 1) SAMPLING

# ##### • Take 100000 samples out of 20752595

# In[7]:


df=df.sample(100000).reset_index().drop('index',axis=1)


# In[8]:


df.head()


# ##### Store sample taken into csv for faster operation in future and also to avoid sampling every time. If we do sampling every time then our results will be impacted

# In[9]:


'''from google.colab import files
df.to_csv('household_power_consumption_100000_samples.csv')
files.download('household_power_consumption_100000_samples.csv')'''


# ##### Read Data From GitHub

# In[10]:


url = 'https://raw.githubusercontent.com/subhashdixit/Regression_Model_Tasks/main/Household_Power_Consumption_Regression_Problem/household_power_consumption_100000_samples.csv'
df = pd.read_csv(url)


# In[11]:


df.head()


# ##### Data Set Information:

# This archive contains 2075259 measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 (47 months)
# 
# ##### We have taken 100000 samples only to predict power consumption
# Notes:
# 
# (global_active_power*1000/60 - sub_metering_1 - sub_metering_2 - sub_metering_3) represents the active energy consumed every minute (in watt hour) in the household by electrical equipment not measured in sub-meterings 1, 2 and 3
# The dataset contains some missing values in the measurements (nearly 1,25% of the rows). All calendar timestamps are present in the dataset but for some timestamps, the measurement values are missing: a missing value is represented by the absence of value between two consecutive semi-colon attribute separators. For instance, the dataset shows missing values on April 28, 2007
# Attribute Information:
# 
# date:
# Date in format dd/mm/yyyy
# 
# time:
# time in format hh:mm:ss
# 
# global_active_power:
# household global minute-averaged active power (in kilowatt)
# 
# global_reactive_power:
# household global minute-averaged reactive power (in kilowatt)
# 
# voltage:
# minute-averaged voltage (in volt)
# 
# global_intensity:
# household global minute-averaged current intensity (in ampere)
# 
# sub_metering_1:
# energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
# 
# sub_metering_2:
# energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
# 
# sub_metering_3:
# energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

# In[12]:


df.columns


# ##### Drop "Unnamed: 0" column because it is of no use

# In[13]:


df.drop(['Unnamed: 0'], axis = 1, inplace = True)


# ## 2) EDA

# ##### Informaation about the dataset

# In[14]:


df.info()


# ##### We will do our analysis on the basis of Daily Data and ignore time column

# In[15]:


df['Date'] = pd.to_datetime(df['Date'])


# In[16]:


df.columns


# In[17]:


df.drop(['Time'], axis = 1, inplace = True)


# In[18]:


df.head


# In[19]:


df.isnull().sum()


# In[20]:


df.duplicated().sum()


# ##### Drop duplicates data

# In[21]:


df.drop_duplicates(inplace = True)


# In[22]:


df.columns


# In[23]:


df['Sub_metering_1'].unique()


# In[24]:


df.replace('?', np.nan, inplace=True)


# In[25]:


df.isnull().sum()


# In[26]:


df.fillna(df.median().round(1), inplace=True)


# In[27]:


df.isnull().sum()


# ##### Convert data to float datatype because all values are in decimal

# In[28]:


conversion = {'Global_active_power' : 'float64', 'Global_reactive_power'  : 'float64', 'Voltage' : 'float64',
       'Global_intensity' : 'float64', 'Sub_metering_1' : 'float64', 'Sub_metering_2' : 'float64',
       'Sub_metering_3' : 'float64'}
df = df.astype(conversion)


# ##### Take date wise data only

# In[29]:


df= df.groupby('Date').sum()


# In[30]:


df.reset_index(inplace = True)


# In[31]:


df['year']=df['Date'].dt.year
df['month']=df['Date'].dt.month


# In[32]:


df.groupby('year').sum()


# In[33]:


df.groupby('month').sum()


# ##### Drop year and month column. We have created these two just to perform basic analyis

# In[34]:


df.shape


# ##### Remove year- 2006 because it may create problem while analysis

# In[35]:


df = df[df['Date']>'2006-12-31']


# In[36]:


df.shape


# In[37]:


df.isnull().sum()


# In[ ]:





# ##### Calculation of target variable - ”power_consumption”

# In[38]:


a = (df['Global_active_power']*1000/60)
b = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']
df['power_consumption'] = a - b
df.head()


# ##### Sum all the values of sub meters into one features i.e., ”Sub_metering”

# In[39]:


df['Sub_metering']=df['Sub_metering_1']+df['Sub_metering_2']+df['Sub_metering_3']


# In[40]:


df = df.drop(['Sub_metering_1','Sub_metering_2','Sub_metering_3'],axis=1)


# In[41]:


df.head()


# In[42]:


df.isnull().sum()


# ## 3) Graphical Analysis

# ##### 3.1 Outliers

# In[43]:


fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
  if col!='Date':
    sns.boxplot(x = col, data = df, ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# In[44]:


def find_boundaries(df, variable, distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    lower_boundary = df[variable].quantile(0.25) - (IQR*distance)
    upper_boundary = df[variable].quantile(0.75) + (IQR*distance)
    return upper_boundary, lower_boundary


# In[45]:


outliers_columns = ['Global_active_power', 'Global_reactive_power','Voltage','Global_intensity','power_consumption','Sub_metering']
for i in outliers_columns:
    upper_boundary, lower_boundary = find_boundaries(df,i, 1.5)
    outliers = np.where(df[i] > upper_boundary, True, np.where(df[i] < lower_boundary, True, False))
    outliers_df = df.loc[outliers, i]
    df_trimed= df.loc[~outliers, i]
    df[i] = df_trimed


# In[46]:


df.isnull().sum()


# In[47]:


df.fillna(df.median().round(1), inplace=True)


# In[48]:


df.dropna(inplace = True)


# In[49]:


df.isnull().sum()


# In[50]:


fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    if col!='Date':
        sns.boxplot(x = col, data = df, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# ##### 3.2 Bar Plots

# In[51]:


fig, ax = plt.subplots(ncols = 5, nrows = 1, figsize=(20,5))
index = 0
ax = ax.flatten()
for col, value in df.items():
    if col not in ['Date', 'year', 'month']:
        sns.barplot(y = df[col], x = df['year'], data = df, ax=ax[index] )
        index += 1 
    if index == 5:
        break 
plt.tight_layout(pad=1, w_pad=1, h_pad=10.0)


# In[52]:


fig, ax = plt.subplots(ncols = 5, nrows = 1, figsize=(20,5))
index = 0
ax = ax.flatten()
for col, value in df.items():
    if col not in ['Date', 'year', 'month']:
        sns.barplot(y = df[col], x = df['month'], data = df, ax=ax[index] )
        index += 1 
    if index == 5:
        break 
plt.tight_layout(pad=1, w_pad=1, h_pad=10.0)


# ##### Observation 
# 1) Power consumption in November and January are on higher side
# 2) Voltage is almost equal in every month

# ##### 3.3 Histplot

# In[53]:


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    if col not in ['Date', 'year', 'month']:
        sns.histplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# ##### 3.4 DistPlot

# In[54]:


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    if col not in ['Date', 'year', 'month']:
        sns.distplot(value, ax=ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# ## 4) Statistical Analysis

# In[55]:


df.head


# In[56]:


sns.set(rc={'figure.figsize':(15,10)})
sns.heatmap(data=df.corr(), annot=True,  vmin=-1, vmax=1)


# ##### Observation 
# Global_active_power, Global_intensity and sub_metering are higly correlated

# In[57]:


df.describe().T


# ##### Observation 
# 1) Maximum power consumption in a day is 2146 w/h
# 2) Average consumption is 631 w/h
# 3) Minimum cosmption is 21 w/h

# ## 5) Segregating Independent and Dependent Features

# In[58]:


X = df.iloc[ : , [1,2,3,4,6,8]]
y = df.iloc[ : , -2]


# In[59]:


X.shape


# In[60]:


y.shape


# In[61]:


X.head


# In[62]:


y.head


# ##### Regplot

# In[63]:


fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(20,10))
index = 0
ax = ax.flatten()
for col, value in df.items():
    if col not in ['Date', 'year', 'month']:
        sns.regplot(x = df[col],y = df["power_consumption"], data = df , ax = ax[index])
        index += 1
plt.tight_layout(pad=0.5, w_pad=0.7, h_pad=5.0)


# ## Train Test Split

# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=7,test_size=0.33)


# ## Scaling

# In[66]:


from sklearn.preprocessing import StandardScaler


# In[67]:


scaler=StandardScaler()


# In[68]:


X_train = scaler.fit_transform(X_train)


# In[69]:


X_test = scaler.transform(X_test)


# In[70]:


len(X_train)


# ## 8) Save Preprocess Model Data Using Pickle

# In[71]:


preprocess_model = [X_train,y_train,X_test,y_test]


# In[72]:


import pickle


# In[73]:


pickle.dump(preprocess_model, open('preprocess_model.pkl','wb'))


# In[74]:


preprocess_model = pickle.load(open('preprocess_model.pkl','rb'))


# ## 9) Save Data into MongoDb

# In[75]:


y_train.T


# In[76]:


database_df = pd.DataFrame([X_train.T[0],X_train.T[1],X_train.T[2],X_train.T[3], X_train.T[4], X_train.T[5],y_train]).T


# In[77]:


database_df.columns=['Global_active_power', 'Global_reactive_power','Voltage', 'Global_intensity', 'month', 'Sub_metering', 'power_consumption']


# In[78]:


database_df.head()


# In[79]:


l=[]
for i ,row in database_df.iterrows():
    l.append(dict(row))


# In[80]:


get_ipython().system('pip install pymongo')


# In[81]:


import pymongo
from pymongo import MongoClient


# In[82]:


client = pymongo.MongoClient("mongodb://localhost:27017")


# In[83]:


db=client['Household_Power_Preprocessed_Data']
collections = db['Training Independent_and_Dependent_Dataset']
collections.insert_many(l)


# ## 10) Load Preprocessed data using Pickle

# In[84]:


preprocess_model = pickle.load(open('preprocess_model.pkl','rb'))


# In[85]:


X_train = preprocess_model[0]
y_train = preprocess_model[1]
X_test = preprocess_model[2]
y_test = preprocess_model[3]


# In[86]:


X_train =pd.DataFrame(X_train)
X_test =pd.DataFrame(X_test)
X_train.columns=['Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity', 'month', 'Sub_metering']
X_test.columns=['Global_active_power', 'Global_reactive_power', 'Voltage','Global_intensity', 'month','Sub_metering']


# ## 11) VIF Check

# ##### • To check multicollinearity

# In[87]:


X_train2 = X_train.copy()
X_train= pd.DataFrame(X_train)


# In[88]:


X_train


# In[89]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(X_train.columns)
print(vif)


# In[90]:


while (max(vif) > 5):    
    indx = vif.index(max(vif)) #Get the index of variable with highest VIF
    print(indx)
    X_train.drop(X_train.columns[indx],axis = 1, inplace = True)
    vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
print(X_train.columns)
print(vif)


# In[91]:


X_test = pd.DataFrame(X_test)
X_test = X_test[X_train.columns]


# ## 12) Model Creation

# In[92]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.model_selection import GridSearchCV


# In[93]:


parameters = {"splitter":["best","random"],
"max_depth" : [1,3,5,7,9,11,12],
"min_samples_leaf":[1,2,3,4,5,6,7,8,9,10],
"min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5],
"max_features":["auto","log2","sqrt",None],
"max_leaf_nodes":[None,10,20,30,40,50,60,70,80,90]
}
## We will train that models
models = {1: DecisionTreeRegressor(random_state=0),
          2: ExtraTreeRegressor(random_state=0),
          3: GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid=parameters,verbose=1, cv=3),
          4: GridSearchCV(ExtraTreeRegressor(random_state=42), param_grid=parameters,verbose=1, cv=3)
}


# In[94]:


map_keys = list(models.keys())


# In[95]:


# Get model name using id from linear_model_collection
def get_model_building_technique_name(num):
    if num == 1:
        return 'DecisionTreeRegressor()'
    if num == 2:
        return 'ExtraTreeRegressor()'
    if num == 3:
        return "GridSearchCV()_DTR"
    if num == 4:
        return "GridSearchCV()_ETR"
        return ''


# In[96]:


results = [];
for key_index in range(len(map_keys)):
    key = map_keys[key_index]
    model = models[key]
    print(key_index)
    model.fit(X_train, y_train)
    
    '''Test Accuracy'''
    y_pred = model.predict(pd.DataFrame(X_test))
    R_Squared_Test = r2_score(y_test,y_pred)
    Adjusted_R_Squared_Test = (1 - (1-R_Squared_Test)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

    '''Train Accuracy'''
    y_pred_train = model.predict(X_train)
    R_Squared_Train = r2_score(y_train,y_pred_train)
    Adjusted_R_Squared_Train = (1 - (1-R_Squared_Train)*(len(y_train)-1)/(len(y_train)-X_test.shape[1]-1))
    results.append({'Model Name' : get_model_building_technique_name(key),
                    'Trained Model' : model,
                    'R_Squared_Test' : R_Squared_Test,
                    'Adjusted_R_Squared_Test' : Adjusted_R_Squared_Test,
                    'R_Squared_Train' : R_Squared_Train,
                    'Adjusted_R_Squared_Train' : Adjusted_R_Squared_Train
    })


# ##### 12.1 Train and Test Accuracy

# In[100]:


print(results)


# In[ ]:




