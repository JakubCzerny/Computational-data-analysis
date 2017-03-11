import pandas as pd
import numpy as np


data = pd.read_csv('data/Case1_Data.csv')
print data.columns
print data.head(5)
print data.describe()

# Identify variables with missing values
print data.isnull().any()
# All of the columns have missing values

target_values = data.iloc[:99, 0]
data.iloc[:,1:100] = data.fillna(data.iloc[:,1:100].median(), inplace=True)
