import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

def get_data():
    data = pd.read_csv('data/Case1_Data.csv')

    # All of the columns have missing values
    target_values = data.iloc[:99, 0]
    data.iloc[:,1:100] = data.fillna(data.iloc[:,1:100].median(), inplace=True)
    dummies = pd.get_dummies(data.iloc[:,-1])
    data = pd.concat([data.iloc[:,0:100], dummies], axis=1)

    # Detect outliers wtih boxplot
    plt.boxplot(data.iloc[:,1:].as_matrix())
    plt.show()
    
    # Split
    train_data = data.iloc[:99,1:]
    test_data = data.iloc[99:,1:]

    # Scale the data
    scaler = preprocessing.StandardScaler().fit(train_data.iloc[:,:-3])
    scaled_train = np.concatenate((scaler.transform(train_data.iloc[:,:-3]), train_data.iloc[:,-3:].as_matrix()), axis=1)
    scaled_test = np.concatenate((scaler.transform(test_data.iloc[:,:-3]), test_data.iloc[:,-3:].as_matrix()), axis=1)

    data.to_csv("data\preprocessed.csv")
    return {'train': scaled_train, 'test':test_data, 'target':target_values}
