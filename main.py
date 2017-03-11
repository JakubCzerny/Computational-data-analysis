import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.decomposition import PCA
import preprocessing
import matplotlib.pyplot as plt

obj = preprocessing.get_data()
train  = obj["train"]
output = obj["target"]
test   = obj["test"]
print train.shape
print output.shape

X_train, X_test, y_train, y_test = train_test_split(train, output,
                                    test_size=0.3, random_state=1)
'''
SPLIT TRAIN/TEST PARAMETER TUNING

model_EN = ElasticNet(l1_ratio=0.7)
alphas = np.logspace(-5,1,50)
train_scores = []
test_scores = []

for alpha in alphas:
    model_EN.set_params(alpha=alpha)
    model_EN.fit(X_train, y_train)
    train_scores.append(model_EN.score(X_train, y_train))
    test_scores.append(model_EN.score(X_test, y_test))
    # test_errors.append(model_EN)

plt.semilogx(alphas, train_scores, label="Train")
plt.semilogx(alphas, test_scores, label="Test")
plt.ylim([-1,1.2])
plt.show()
'''

folds = 5

models = ElasticNetCV(l1_ratio=np.linspace(0,1,5,endpoint=True), alphas=np.logspace(-5,1,20), verbose=1, cv=folds, n_jobs=-1)
models.fit(X_train, y_train)
model_EN = ElasticNet(l1_ratio=models.l1_ratio_, alpha=models.alpha_)
model_EN.fit(X_train, y_train)
print model_EN.score(X_test, y_test)

# TODO
'''
PCA for visualization
Bootstraping for understanding
Extend grid search
Outlier detection
One more interesting model
'''
