
# coding: utf-8

# # House Prices Study (In Python)
# 
# # Introduction: 
# Here, we look at the dataset provided by Kaggle on house prices and house price predictors and follow the following main steps:
# 
# - Descriptive Statistics
# - Feature Engineering
# - Make Predictions
# 
# # 1.1: Load Modules/Packages

# In[1]:


import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os #for loading path

# initial variable analysis tools
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from sklearn.decomposition import PCA
from patsy import dmatrices
import itertools

# data cleaning tools
from sklearn.preprocessing import StandardScaler, LabelEncoder, Imputer

# model evaluation tools
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error
from statsmodels.tools.eval_measures import mse

# machine learning modules
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, IsolationForest
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


# # 1.2 Load and look at data

# In[2]:


house_train = pd.read_csv("C:/KaggleStuff/House_Prices/house_train.csv")
house_test = pd.read_csv("C:/KaggleStuff/House_Prices/house_test.csv")

# rename some columns to use for dmatrices
house_train = house_train.rename(columns = {'1stFlrSF': 'FirstFlrSF', '2ndFlrSF':'SecFlrSF', '3SsnPorch':'ThreeSsnPorch'}) 
house_test = house_test.rename(columns = {'1stFlrSF': 'FirstFlrSF', '2ndFlrSF':'SecFlrSF', '3SsnPorch':'ThreeSsnPorch'})


# In[3]:


house_train.head() # peeking at data


# Looks like our training set has some NaN values we'll need to fix later

# In[4]:


house_train.describe() # some descriptive statistics


# In[5]:


house_train.info() # getting a sense of object types 


# # 1.3 Data Visualization
# Now we will visualize the data with boxplots, histograms and scatterplots when applicable

# In[6]:


# plots the boxplot and histogram of integer valued variables only
for v in house_train.select_dtypes(include = ['int64']).columns.values[1:10]:
    plt.boxplot(house_train[v])
    plt.title(v)
    plt.show()
    plt.hist(house_train[v])
    plt.xlabel(v)
    plt.ylabel("Frequency")
    plt.show()


# In[7]:


# grabbing the boxplots of the categorical variables
for v in house_train.select_dtypes(include = ['object']).columns.values[1:5]:
    sns.countplot(x=v, data=house_train, palette="Greens_d")
    plt.title("Frequency of " + v)
    plt.show()


# In[8]:


# plotting the scatterplot of the float variables
#for v in house_train.select_dtypes(include = ['float64']).columns.values[1:(house_train.shape[0]-1)]:
plt.show()
scatter_matrix(house_train.select_dtypes(include = ['float64'])) 
plt.show()


# As a result of the above, we have a bunch of categorical variables that we need to change to numeric form if we are to perform regression. We also have some variables with only years that we will change from year to number of years till our current year: 2017. 

# # 1.3 Summary Statistics
# Now we look at stats such as R-Squared, P-value of the non categorical variables. 

# In[9]:


# some feature engineering: Changing years: Year Built changed to how many years since current year, Year Remodeled changed to how many years since current year
current_yr = 2017
house_train['YearBuilt'] = abs(house_train['YearBuilt'] - current_yr)
house_train['YearRemodAdd'] = abs(house_train['YearRemodAdd'] - current_yr)

def singleVarModelStat(dataset, variables, dep_var):
    single_var_stat = pd.DataFrame(columns = ['Variable', 'R_Squared', 'P_value', 'F_stat', 'Correlation'])
    i = 0
    for v in variables:
        formula = dep_var + "~" + v
        model = ols(formula, data = dataset).fit()
        single_var_stat.loc[i] = [v, model.rsquared, model.f_pvalue, model.fvalue, np.corrcoef(dataset[dep_var], dataset[v])[0,1]]
        i += 1
    return single_var_stat

single_var_stat = singleVarModelStat(house_train, house_train.select_dtypes(include = ['int64', 'float64']).columns.values[1:35], 'SalePrice')
single_var_stat.head(10)


# # 2.1 Replacing Missing Values
# There are many ways to replace missing values - mean, mode, or using machine learning tools to predict the missing values:
# For simplicity:
# - using mode for categorical missing values
# - using mean for continous missing 

# In[10]:


# This method checks for missing in dataset, removes them and replaces with the mean of that particular predictor var
#  if its an numeric value other wise it replaces the missing value with the most common value 
#
def fillMissingValues(dataset, typeItem):
        dataset_obj = dataset.select_dtypes(include = [typeItem]).copy()
        dataset_null = dataset_obj.isnull().any()[dataset_obj.isnull().any() == True]
        for v in dataset_null.index.values:
            if typeItem == 'int64' or typeItem == 'float64':
                mean_val = np.mean(dataset[v])
                if typeItem == 'int64':
                    mean_val = int(mean_val)
                dataset_obj = dataset_obj.fillna({v: mean_val}) 
            elif typeItem == 'object':
                cur_var = dataset_obj[v].value_counts()
                max_value = cur_var[cur_var.index.values[0]]
                pop_value = cur_var.index.values[0]
                for i in cur_var.index.values: # check for most frequent value
                    max_value = cur_var[i]
                    if cur_var[i] > max_value:
                        max_value = cur_var[i]
                        pop_value = i 
                dataset_obj = dataset_obj.fillna({v: pop_value}) 
        return dataset_obj

# This function replaces the updated columns of the previous processes in the original dataset
def replaceOldDataset(old_dataset, new_columns):
    for col in new_columns.columns.values:
        if(old_dataset[col].dtype == 'float64' or old_dataset[col].dtype == 'int64'):
            old_dataset[col] = new_columns[col]
        else:
            old_dataset[col] = new_columns[col]
    return old_dataset

#since for my dataset, we have missing values for every type
house_train_miss1 = fillMissingValues(house_train, 'int64')
house_train_miss2 = fillMissingValues(house_train, 'float64')
house_train_miss3 = fillMissingValues(house_train, 'object')

house_train = replaceOldDataset(house_train, house_train_miss1)
house_train = replaceOldDataset(house_train, house_train_miss2)
house_train = replaceOldDataset(house_train, house_train_miss3)

house_train.head()


# # 2.2 Encoding Categorical Values

# In[11]:


# This changes categorical values (objects) to type int (creates an Encoder object)
def encodingCategorical(dataset):
    le = LabelEncoder()
    for var in dataset.columns.values:
        le.fit(dataset[var])
        dataset[var] = le.transform(dataset[var])
    return dataset

house_train_miss3 = fillMissingValues(house_train, 'object')
house_train_miss3 = encodingCategorical(house_train_miss3)
house_train = replaceOldDataset(house_train, house_train_miss3)

house_train.head()


# # 2.3 Multicollinearity 
# Another main issue in regression is the deal with multicollinearity which will increase standard errors. We will locate the variables with high multicollinearity by using variance inflation factors (VIFs)and remove the variables that have low correlation with Sale Price

# In[12]:


# setting base variables and dependent variables (so we don't have to keep writing them out)
variables = house_train.columns.values[1:(len(house_train.columns.values)-1)]
dep_var = house_train.columns.values[len(house_train.columns.values)-1]
total_variables = house_train.columns.values[1:len(house_train.columns.values)]


# In[13]:


# used to find the VIF of each variable when all variables are inserted into the model
def VIF(dataset, variables, dep_var):
    features = '+'.join(dataset[variables].columns)
    y, X = dmatrices(dep_var + '~' + features, dataset, return_type ='dataframe')
    vif = pd.DataFrame()
    vif['variables'] = X.columns[1:] # exclude id col
    vif['VIF'] = [variance_inflation_factor(dataset[variables].values, i) for i in range(0, dataset[variables].shape[1])]
    return vif

vif_multi_var = VIF(house_train, variables, dep_var)
vif_multi_var.head()


# # 2.4 Eliminating Features
# Here, we eliminate features based on VIF factors greater than 10 and correlations with Sale Price less than 60%

# In[14]:


# removing features based on VIF and correlation with Sale Price
single_var_stat = singleVarModelStat(house_train, variables, dep_var)
i = 0
for v in variables:
    if(vif_multi_var.iloc[i]['VIF'] > 10 and abs(single_var_stat.iloc[i]['Correlation']) < 0.60):
        variables = list(variables)
        variables.remove(v)
        i += 1
    elif(single_var_stat.iloc[i]['Correlation'] < 0.05):
        variables = list(variables)
        variables.remove(v)
        i += 1
    else:
        i += 1
        continue
house_train_new = house_train[variables]


# Eliminating features using PCA (method 2)

# In[15]:


# gather the best components using PCA 
def PCA_analysis(dataset):
    pca = PCA(n_components = 20)
    X_train = pca.fit_transform(dataset[variables])
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title("Explained variance by different principal components")
    plt.xlabel("Principal Component")
    plt.ylabel("Fractional Explained Variance")
    plt.show()
    return X_train

X_train_pca = PCA_analysis(house_train_new)


# # 3.1 Multivariate Linear Regression Prediction
# To start off simple, we're going to use linear regression as our first model for predicting on the test set using the training data

# In[16]:


# splitting data into testing and training data
#dataset = house_price_new
#X_train, X_test, y_train, y_test = train_test_split(dataset[variables],dataset[dep_var], test_size = 0.5, random_state = 42)

# we have to fill missing values in the test set
house_test_miss1 = fillMissingValues(house_test, 'object')
house_test_miss2 = fillMissingValues(house_test, 'int64')
house_test_miss3 = fillMissingValues(house_test, 'float64')

# first we have to convert the test set to numeric data as well
house_test_miss1= encodingCategorical(house_test_miss1)
house_test_new = replaceOldDataset(house_test, house_test_miss1)
house_test_new = replaceOldDataset(house_test_new, house_test_miss2)
house_test_new = replaceOldDataset(house_test_new, house_test_miss3)

# setting up training and testing sets
y_train = house_train['SalePrice']
X_test = house_test_new[variables]
X_train = house_train_new[variables]

lr = LinearRegression() # creating our linear model
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

plt.plot(y_pred) # visualizing our predictions
plt.title("Linear Regression House Sale Price Predictions")
plt.ylabel('Sale Price $')
plt.xlabel("House ID")
plt.show()

#print(np.sqrt(mean_squared_error(y_pred, y_test)))


# # 3.2 Random Forest Predictions

# In[17]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

plt.plot(y_pred_rf)
plt.title("Random Forest House Sale Price Predictions")
plt.ylabel('Sale Price $')
plt.xlabel("House ID")
plt.show()

