
# coding: utf-8

# In[501]:


# The aim of this project is to use some Machine learning algortihms 
# that I learnt to achieve a practical aim whcih is to predict a car's 
# market price using its attributes


# In[502]:


# Load the data and see if everything is in order. 

import pandas as pd
cars = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data")
print(cars.head())


# In[503]:


print(cars.columns)


# In[504]:


# Looks like the column names are actually the first values of each row.
# One way to fix that problem would be to reread the dataframe with column
# names. 


# In[505]:


col_names = ["symboling","normalized-losses", "make","fuel-type","aspiration","num-of-doors",
             "body-style","drive-wheels","engine-location","wheel-base","length","width",
             "height", "curb-weight", 'engine-type', 'num-of-cylinders', 'engine-size', 
             'fuel-system', 'bore', 'stroke', 'compression-rate', 'horsepower', 'peak-rpm',
             'city-mpg', 'highway-mpg', 'price']
cars = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data", names = col_names)
print(cars.head())


# In[506]:


# Now to determine which columns are numeric and can be used as features. 
# Reviewing the dataset, I'll make a list of columns to keep and discard 
# the rest. The target column would be the price. 
col_to_keep = ['normalized-losses', 'engine-size','wheel-base', 'length', 'width', 'height',
               'curb-weight', 'bore', 'stroke', 'compression-rate', 'horsepower', 
               'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
cars_new = cars[col_to_keep]
print(cars_new.head())


# In[507]:


# Looking at the dataframe, I can see '?' elements in it. I'll replace that with
# nan values
import numpy as np
cars_new = cars_new.replace("?",np.nan)
print(cars_new.head())


# In[508]:


# Checking how many missing values are there in each column 
print(cars_new.isnull().sum())
print(cars.index)


# In[509]:


# Here it seems that "normalized losses" column has way too many missing values
# Makes sense to drop that columns. Also the other column with few missing values
# can be filled in with the mean of that specific column. 
# Before that, I will drop the rows that have missing values in price. 
cars_new=cars_new.drop(columns = "normalized-losses",axis = 1)
print(cars_new.columns)


# In[510]:


cars_new = cars_new.dropna(subset=["price"] , axis = 0)
print(cars_new.isnull().sum())


# In[511]:


# replace the 4 misiing values with the mean 
# I wont be able to fill it with the mean if the data type is not consistent. 
# Hence, converting the dataframe to a float 
cars_new = cars_new.astype('float')
print(type(cars_new["bore"][0]))
cars_new["bore"].fillna(cars_new["bore"].mean(),inplace=True)
print(cars_new["bore"].value_counts())
print(cars_new.isnull().sum())


# In[512]:


# Doing the same with the other missing values 
cars_new["stroke"].fillna(cars_new["stroke"].mean(),inplace=True)
cars_new["horsepower"].fillna(cars_new["horsepower"].mean(),inplace=True)
cars_new["peak-rpm"].fillna(cars_new["peak-rpm"].mean(),inplace=True)


# In[513]:


print(cars_new.isnull().sum())


# In[514]:


# Now normalizing the feature columns because when we will use the data to calculate
# distances, it will give more weight to bigger numbers and will not take relativity
# into account. Hence, its better to normalize the data at the start to avoid
# making a bad algorithm. 
# The formula to do so is: 
# df_new = (df - df.mean())/(df.max()-df.min())

norm_cars = abs((cars_new-cars_new.mean())/(cars_new.max()-cars_new.min()))


# In[515]:


print(norm_cars.head())


# In[516]:


# Replacing the price column to what it was before as that is our target column
norm_cars["price"] = cars_new["price"]
print(norm_cars.head())


# In[517]:


# Getting to the KNN part of the project. Here, I want to divide the dataframe
# into 'train' and 'test'. The idea is to determine 'k' number of rows in train 
# that have the smallest distance to test. Then we can use those rows for further
# calculations. 


# In[518]:


# I'll write a function that does everything by taking in training column names,
# target column name and the dataframe. 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle 

rmse = dict()
def knn_train_test(df,target_col_name,train_col_name,k):
    
    #Splitting the dataset 

    df = shuffle(df)
    
    point_split = (int(len(df.index)*0.60))
    
    df_train = df.iloc[0:point_split]
    df_test = df.iloc[point_split:]
    
    # the algorithm 
    knn = KNeighborsRegressor(n_neighbors = k)
    knn.fit(df_train[[train_col_name]],df_train[target_col_name])
    
    # getting the mean square error 
    predictions = knn.predict(df_test[[train_col_name]])
    mse= (mean_squared_error(predictions,df_test[target_col_name]))
    rmse = mse**(1/2)
    return rmse 
    
print(knn_train_test(norm_cars,"price","length",5))
    


# In[519]:


# Iterating over all columns to get mean squared errors for all. 
feature_names = norm_cars.columns.tolist()
feature_names.remove("price")
print(feature_names)


# In[520]:


for col in feature_names: 
    rmse[col] = knn_train_test(norm_cars,"price",col,5)
print(rmse)


# In[521]:


rmses_total = pd.Series(rmse)
print((rmses_total))


# In[522]:


# Now to convert the univariate model into a multivariate. Lets try different
# values of 'k'. What this means is that the algorithm will look for 'k' rows
# to use in its calculation


# In[532]:


k_vals = [1,3,5,7,9]
series = dict()
new_plot = {}
for col in feature_names:
    ser = []
    for k in k_vals:
        calc = knn_train_test(norm_cars,"price",col,k)
        ser.append(calc)
    new_plot[col]= ser
    calc_mean = np.mean(calc)
    series[col] = calc_mean
series = pd.Series(series)
# print(series)
# print(new_plot)


# In[533]:


# To plot this data 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.figure(figsize=(9,9))
for key,vals in new_plot.items():
    plt.plot(k_vals,vals,label = key,linewidth=3)
    
    plt.xlabel("k-values")
    plt.ylabel("RMSE")
    plt.legend(loc="upper right",bbox_to_anchor=(1.35,0.9))
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.show()
    


# In[525]:


# Until now, I was just taking one column at a time. Now lets take more and
# see what the results are. 
two_features = ["engine-size","city-mpg"]
three_features=["engine-size","city-mpg","highway-mpg"]
four_features=["engine-size","city-mpg","highway-mpg","width"]
five_features=["engine-size","city-mpg","highway-mpg","width","curb-weight"]




# In[526]:


def knn_train_test_new(df,target,train,k):
    df = shuffle(df)
    point_split = (int(len(df.index)*0.60))
                   
    df_train=df.iloc[0:point_split]
    df_test = df.iloc[point_split:]
    
    knn = KNeighborsRegressor(n_neighbors = k,algorithm='brute')
    
    knn.fit(df_train[train],df_train[target])
    
    predictions = knn.predict(df_test[train])
    mse = mean_squared_error(predictions, df_test[target])
    rmse = mse**(1/2)
    return rmse

rmse_val = dict()
# feature_list = [two_features,three_features,four_features,five_features]
# for i in feature_list:
#     new_rmse = knn_train_test(norm_cars,"price",i,5)
#     rmse_val[i] = new_rmse 
# new_rmse = pd.Series(rmse_val)
# print(new_rmse)    
rmse_val["two_features"] = knn_train_test_new(norm_cars,"price",two_features,5)
rmse_val["three_features"] = knn_train_test_new(norm_cars,"price",three_features,5)
rmse_val["four_features"] = knn_train_test_new(norm_cars,"price",four_features,5)
rmse_val["five_features"] = knn_train_test_new(norm_cars,"price",five_features,5)
rmse_val_ser = pd.Series(rmse_val)
print(rmse_val_ser)


# In[536]:


k_vals = list(np.arange(1,50,1))
count =2

store = dict()
models = [two_features,three_features,five_features]
for i in models: 
    new_dict = {}
    for j in k_vals:
        value = knn_train_test_new(norm_cars,"price",i,j)
        new_dict[j] = value
    if(count!=4):
        store[str(count)+"_"+"features"] = new_dict
        count+=1
    else:
        store["5_features"] = new_dict
print(store)


# In[537]:


import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for key,val in store.items():
    new_ser=[]
    for j in val:
        new_ser.append(val[j])

    plt.plot(k_vals,new_ser,label = key,linewidth=3)
    plt.legend(loc="upper right")
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
plt.show()


# In[538]:


# Looks like bigger the value of k, higher is the rmse value. The model seems
# to do best for 3_features and a k-value of less than 5. 

