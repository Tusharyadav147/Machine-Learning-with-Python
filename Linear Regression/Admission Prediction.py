from numpy.core.fromnumeric import var
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

df = pd.read_csv("Admission_Prediction.csv")
print(df)

#first create a profile report 
"""pr = ProfileReport(df)
pr.to_file("admission_report.html")"""

#dealing with missing values
df["GRE Score"] = df["GRE Score"].fillna(df["GRE Score"].mean())
df["TOEFL Score"] = df["TOEFL Score"].fillna(df["TOEFL Score"].mean())
df["University Rating"] = df["University Rating"].fillna(df["University Rating"].mean())

print(df.describe())
print(df.isnull().sum())

#droping unwanted column
df.drop(columns= ["Serial No."], inplace= True)

#separate independent as dependent features
x = df.drop(columns= "Chance of Admit")
y = df["Chance of Admit"]

#normalization and standardization (to make the same range of all the features)
scalar = StandardScaler()
arr = scalar.fit_transform(x)
df1 = pd.DataFrame(arr)

print(df1.describe())

#check multicollinarity
vif_df = pd.DataFrame()
vif_df["vif"] = [variance_inflation_factor(arr,i) for i in range (arr.shape[1])]
vif_df["feature"] = x.columns
print(vif_df)               #if we get vif value more then 10 then we have to drop that perticular column

#split the dataset into train and test
x_train, x_test, y_train, y_test = train_test_split(arr, y, test_size=.2)

#create a model
model = LinearRegression()
model.fit(x_train, y_train)

pickle.dump(model, open("admission_model.pkl", "wb"))

model.predict(x_test)   #for prediction through direct value first we need to convert the values though standard scalar technique

#to check the accuracy score of our model
print(model.score(x_test, y_test))

#get m and c
print(model.intercept_)
print(model.coef_)

#implement model through lasso
lassocv = LassoCV(cv = 10, max_iter=20000000, normalize=True)
lassocv.fit(x_train, y_train)
print(lassocv.alpha_)

lasso = Lasso(alpha= lassocv.alpha_)
lasso.fit(x_train, y_train)
print(lasso.score(x_test, y_test))

#implement model through lasso
ridgecv = RidgeCV(cv = 10, normalize= True, alphas = np.random.uniform(0, 10, 50))
ridgecv.fit(x_train, y_train)
print(ridgecv.alpha_)

ridge = Ridge(alpha = ridgecv.alpha_)
ridge.fit(x_train, y_train)
print(ridge.score(x_test, y_test))

#similarly create with elasticnet
