from doctest import testfile
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt


#load data using pandas
df = pd.read_csv("winequality_red.csv")
print(df.head(10))

#create a profiile report of dataset
"""pr = ProfileReport(df)
pr.to_file("winequality.html")"""       #to make it in a html file 

#separate the independent and dependent variable
x = df.drop(columns = "quality")
y = df["quality"]

#make training and testing dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=300)

#create a decision tree model
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
print(y_predict)

#to find the accuracy of model
print(model.score(x_test, y_test))

"""outfile = open("model.dot", "w")
tree.export_graphviz(model, out_file=outfile, feature_names= x.columns)"""

#to plot decision tree
plt.figure(figsize=(20,20))
tree.plot_tree(model, class_names=[str(i) for i in set(y_train)])
plt.show()


#to make a better prediction we have to provide the best ccp alpha value to our model
df1 = df.head(500)

x1 = df1.drop(columns = "quality")
y1 = df1.quality

dt_model1 = DecisionTreeClassifier()
dt_model1.fit(x1, y1)

path = dt_model1.cost_complexity_pruning_path(x1, y1)
ccp_alpha = path.ccp_alphas

dt_model2 = []
for ccp in ccp_alpha:
    dt_m = DecisionTreeClassifier(ccp_alpha=ccp)
    dt_m.fit(x1, y1)
    dt_model2.append(dt_m)

train_score = [i.score(x1, y1) for i in dt_model2]
test_score = [i.score(x_test, y_test) for i in dt_model2]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Alpha For training and testing sets")
ax.plot(ccp_alpha, test_score, marker = "o", label = "test", drawstyle = "steps-post")
ax.legend()
plt.show()


#creating a model with best ccp alpha value
dt_model_cpp = DecisionTreeClassifier(random_state=0, ccp_alpha=.014)
dt_model_cpp.fit(x1, y1)

plt.figure(figsize=(20,20))
tree.plot_tree(dt_model_cpp, filled=True)

print(dt_model_cpp.score(x1, y1))
print(dt_model_cpp.score(x_test, y_test))

#to make model with best parameter 
grid_pram = {"criterion" : ["gini", "entropy"],
    "splitter" : ["best", "random"],
    "max_depth": range(2,40,1),
    "min_sample_split": range(2,10,1),
    "min_sample_leaf": range(1,10,1)
}
#to find the best parameter use gridsearchcv as well as randomsearchcv
"""grid_ccp = GridSearchCV(estimator=dt_model_cpp, param_grid= grid_pram, cv = 5, n_jobs=-1)
grid_ccp.fit(x1, y1)

grid_ccp.best_params_"""             #to  get all best parameters


#create a confusion matrix
pred = dt_model_cpp.predict(x_train)
print(confusion_matrix(y_train, pred))

