import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet, ElasticNetCV, LassoLars, LassoLarsCV
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


folds = 5

# Elastic-net
models = ElasticNetCV(l1_ratio=np.linspace(0,1,10,endpoint=True), alphas=np.logspace(-8,3,40), verbose=1, cv=folds, n_jobs=-1)
models.fit(X_train, y_train)
model_EN = ElasticNet(l1_ratio=models.l1_ratio_, alpha=models.alpha_)
model_EN.fit(X_train, y_train)
model_EN.score(X_train, y_train)
print "Score of Elastic-net on train data: ", model_EN.score(X_train, y_train)
print "Score of Elastic-net on test data: ", model_EN.score(X_test, y_test)
print "L1 ratio: ", models.l1_ratio_
print "Alpha: ", models.alpha_


# At this point we should be our final model on the entire dataset
# using previously tunned parameters


# Lasso-lars
models = LassoLarsCV(max_n_alphas=40, verbose=1, cv=folds)
models.fit(X_train, y_train)
model_LL = LassoLars(alpha=models.alpha_)
model_LL.fit(X_train, y_train)
print "Score of Lasso-Lars on train data: ", model_LL.score(X_train, y_train)
print "Score of Lasso-Lars on test data: ", model_LL.score(X_test, y_test)


'''
=============== PCA ==================
3 first components explain only ~33% of the variance when applied non-normalized data

You can test it by copying following code to preprocessing.py

from sklearn.decomposition import PCA
pca = PCA(3)
X = pca.fit_transform(data.iloc[:,1:])
print pca.explained_variance_ratio_


Actually I was surprised by how poorly it performed so I even wrote a simple MatLab script because it didn't feel right,
but it outputs exactly same numbers.

Then I tested it on scaled data but that results in even poorer results.
We can still mention that in the report but I'm not sure how that's the case.
Maybe we could dig into this to understand what may cause that.

from sklearn.decomposition import PCA
pca = PCA(3)
X = pca.fit_transform(X_train)
print pca.explained_variance_ratio_

X = pca.fit_transform(X_test)
print pca.explained_variance_ratio_

X = np.concatenate((X_train, X_test))
X = pca.fit_transform(X)
print pca.explained_variance_ratio_





=============== Elastic-net vs Lasso-Lars ==================
Interestingly it seems thtat EN performs better on much better on training data than LL,
but on unseen test data it's the other way around. Maybe it would be also nice to see how
they both perform on complete dataset





=============== Extend grid search ==================
I tried to make them more dense but I didn't really get better results.
Feel welcome to test them too :) maybe I omitted something interesting
Interestingly, for EN l1_ratio is 1 or very close to 1 (.98 or so) what makes it pure rigde regression




=============== Outlier detection ==================
We can see on a boxplot that there are actually some outliers
http://www.wikihow.com/Calculate-Outliers

http://matplotlib.org/api/pyplot_api.html
The boxplot returns dictionary where one of the keys is
`fliers`: points representing data that extend beyond the whiskers (fliers).
Maybe we could use this to get the outliers and for example map them to max/min
for being outlier? I don't know what would be the standard procedure and that's just
something I came up with
'''
