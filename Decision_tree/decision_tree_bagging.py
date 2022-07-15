from doctest import testfile
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt

df = pd.read_csv("winequality_red.csv")
print(df.head())

x = df.drop(columns = ["quality"])
y = df.quality

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state= 0, test_size= .20)

bag_dt = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)            #here we create 10 decision tree in one bag
bag_dt.fit(x_train, y_train)

print(bag_dt.predict(x_test))
print(bag_dt.classes_)             #to find the number of classes we have

bag_KNN = BaggingClassifier(KNeighborsClassifier(6), n_estimators=10)
bag_KNN.fit(x_train, y_train)

bag_KNN.predict(x_test)

rf = RandomForestClassifier(n_estimators=5)         #it use to create 100 decision tree with in a bag
rf.fit(x_train, y_train)

print(rf.score(x_test, y_test))
print(bag_dt.score(x_test, y_test))
print(bag_KNN.score(x_test, y_test))

print(rf.estimators_)

#print decision tree
plt.figure(figsize=(20,20))
tree.plot_tree(rf.estimators_[0], filled=True)
plt.show()

#to find the best parameter 
rf = RandomForestClassifier(n_estimators=5)

grid_param = {
    "n_estimators": [5,10,50,100,120,150],
    "criterion": ["gini", "entropy"],
    "max_depth": range(10),
    "min_sample_leaf" : range(10)
}
#only execute when system have high configuration
"""grid_search_rf = GridSearchCV(param_grid= grid_param, cv = 10, n_jobs=6, verbose=1, estimator=rf)
grid_search_rf.fit(x_train, y_train)
print(grid_search_rf.best_params_)"""

rf_new = RandomForestClassifier(criterion="entropy", max_depth=9, min_samples_leaf=1, n_estimators=100)
rf_new.fit(x_train, y_train)
print(rf_new.score(x_test, y_test))
