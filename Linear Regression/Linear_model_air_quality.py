import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.pandas.set_option("display.max_columns", None)
df = pd.read_csv("ai4i2020.csv")
print(df.head())
print(df.shape)

#droping unwanted columns
df = df.drop(columns=["UDI","Product ID", "Type"])

#create a profile report and save it in the html file
pr = ProfileReport(df)
#pr.to_file("profile_report.html")

sns.heatmap(df.corr(), annot=True)
plt.show()

print(df.describe())
print(df.info())
print(df.isnull().sum())

x = df.drop(columns="Air temperature [K]")
y = df["Air temperature [K]"]
print("x" , x)

scaler = StandardScaler()
scaler.fit(x)

scaler_feature = scaler.transform(x)
df_feat = pd.DataFrame(scaler_feature, columns=df.columns[1::])
print(df_feat.head())

print(df_feat.describe())

x_train, x_test, y_train, y_test = train_test_split(df_feat, y, test_size=.3)

model = LinearRegression()
model.fit(x_train, y_train)

print(model.predict(x_test))
print(y_test)
print(model.score(x_test, y_test)*100)
