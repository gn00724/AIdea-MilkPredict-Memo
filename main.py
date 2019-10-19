#%%
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime as dt

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("./乳牛/data/train.csv", error_bad_lines = False)
df_report = pd.read_csv("./乳牛/data/report.csv", error_bad_lines = False)
df_birth = pd.read_csv("./乳牛/data/birth.csv", error_bad_lines = False)
df_submit = pd.read_csv("./乳牛/data/submission.csv", error_bad_lines = False)

#%%
#init
X = df_train
X_pred = df_report
X["11"] = X["11"].dropna(axis=0)
Y = pd.DataFrame(df_train["11"])

#%%
#缺值檢查
print(X.count(), "\n", Y.count())

#%%
#年份和乳量的關係
YearsFact = pd.Series(df_train.groupby(df_train["2"])["11"].mean())
sns.relplot(data=YearsFact)

#%%
d_YearsType = pd.qcut(YearsFact,4)
print(d_YearsType.values.codes)
d_yearsTypeDict = {}

for m in df_train["2"]:
    d_yearsTypeDict[m] = d_YearsType.values.codes[m-2013]

df_train["YearsType"] = df_train["2"].map(d_yearsTypeDict).astype("int")
df_train["YearsType"].describe()
df_byYears = pd.Series(df_train.groupby(df_train["YearsType"])["11"].mean())
sns.relplot(data=df_byYears)
#%%
#農場影響
FarmFact = pd.Series(df_train.groupby(df_train["4"])["11"].mean())
sns.relplot(data=FarmFact)


#%%
d_framA = df_train[df_train.loc[:,"4"] == 1]
d_framB = df_train[df_train.loc[:,"4"] == 2]
d_framC = df_train[df_train.loc[:,"4"] == 3]

d_framA.describe()
d_framB.describe()
d_framC.describe()




#%%
#預測與建模
b1 = ["4"]
b1_Model = RandomForestClassifier(random_state=10, n_estimators=250, min_samples_split=20, oob_score=True)
b1_Model.fit(X[b1],Y.astype('int'))
#%%
print(b1_Model.oob_score_)

#%%
