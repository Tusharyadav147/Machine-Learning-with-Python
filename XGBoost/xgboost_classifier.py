import pandas as pd
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb

df = pd.read_csv("winequality_red.csv")
pd.set_option("display.max_columns", None)
print(df.head())

print(df.describe())

x = df.drop(columns="quality")
y = df.quality


def objective_classification(trial):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25)
    param = {
        "verbosity" : 3,
        "objective" : "binary:logistics",
        "booster" : trial.suggest_categorical('booster', ["dart", "gbtree", "bglinear"]),
        "lambda" : trial.suggest_float("lambda", 1e-4, 1),
        "alpha" : trial.suggest_float("alpha", 1e-4, 1),
        "subsample" : trial.suggest_float("subsample", .1,.5),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", .1,.5)
    }
    if param["booster"] in ["gbtree", "dart"]:
        param['gamma'] = trial.suggest_float("gamma", 1e-3, 4)
        param["eta"]= trial.suggest_float("eta", .001, 5)
    xgb_classification = xgb.XGBClassifier("**param")
    xgb_classification.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    pred = xgb_classification.predict(x_test)
    accuracy = xgb_classification.score(x_test, y_test)
    return accuracy

xgb_classification_optuna = optuna.create_study(direction="minimize")
xgb_classification_optuna.optimize(objective_classification, n_trials = 10)

print(xgb_classification_optuna.best_trial)

param = xgb_classification_optuna.best_trial.params
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.25)

xgb_final_class = xgb.XGBClassifier(**param)
xgb_final_class.fit(x_train, y_train)
print(xgb_final_class.score(x_test, y_test))