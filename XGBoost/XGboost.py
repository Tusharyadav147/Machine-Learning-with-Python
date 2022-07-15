
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("Admission_Prediction.csv")
pd.set_option("display.max_columns", None)
print(df.head())
print(df.describe())

df["GRE Score"] = df["GRE Score"].fillna(df["GRE Score"].mean())
df["TOEFL Score"] = df["TOEFL Score"].fillna(df["TOEFL Score"].mean())
df["University Rating"] = df["University Rating"].fillna(df["University Rating"].mean())

x = df.drop(columns=["Serial No.", "Chance of Admit"])
y = df["Chance of Admit"]

std_sca = StandardScaler()
x = std_sca.fit_transform(x)

def objective(trail, data = x, target = y):
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.10, random_state= 30)
    param = {
        "lambda" : trail.suggest_loguniform("lambda", 1e-4, 10.0),
        "alpha": trail.suggest_loguniform("alpha", 1e-4, 10.0),
        "colsample_bytree" : trail.suggest_categorical("colsample_bytree", [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]),
        "subsample" : trail.suggest_categorical("subsample", [.1,.2,.3,.4,.5,.6,.7,.8,.9,1]),
        "learning_rate" : trail.suggest_categorical("learning_rate", [.00001,.0003,.008,.02,1, 10, 20]),
        "n_estimator": 30000, 
        "max_depth": trail.suggest_categorical("max_depth", [3,4,5,6,7,8,9,10,11,12]),
        "random_state": trail.suggest_categorical("random_state", [10,20,30,2000,3454,243123]),
        "min_child_weight": trail.suggest_int("min_child_weight", 1,200)
    }
    xgb_reg_model = xgb.XGBRFRegressor(**param)
    xgb_reg_model.fit(x_train, y_train)
    pred_xgb = xgb_reg_model.predict(x_test)
    rese = mean_squared_error(y_test, pred_xgb)
    return rese

find_param = optuna.create_study(direction="minimize")
find_param.optimize(objective,n_trials= 10)
print(find_param.best_trial.params)

optuna.visualization.plot_optimization_history(find_param)
plt.show()


best_param = find_param.best_trial.params

xgb_final_model = xgb.XGBRFRegressor(**best_param)
xgb_final_model.fit(x,y)
print(xgb_final_model.score(x,y))