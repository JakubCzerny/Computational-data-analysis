import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

data = pd.read_csv('data/Case1_Data.csv')
print data.columns
print data.head(5)
print data.describe()

# Identify variables with missing values
print data.isnull().any()

# All of the columns have missing values
target_values = data.iloc[:99, 0]
data.iloc[:,1:100] = data.fillna(data.iloc[:,1:100].median(), inplace=True)
dummies = pd.get_dummies(data.iloc[:,-1])
data = pd.concat([data.iloc[:,0:100], dummies], axis=1)

# Split
train_data = data.iloc[:100,1:]
test_data = data.iloc[100:,1:]

# Scale the data
scaler = preprocessing.StandardScaler().fit(train_data.iloc[:,:-3])
scaled_train = np.concatenate((scaler.transform(train_data.iloc[:,:-3]), train_data.iloc[:,-3:].as_matrix()), axis=1)
scaled_test = np.concatenate((scaler.transform(test_data.iloc[:,:-3]), test_data.iloc[:,-3:].as_matrix()), axis=1)
