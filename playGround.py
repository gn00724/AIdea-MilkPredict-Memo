#%%
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime as dt
from datetime import timezone
import datetime

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#from toolBox import basicTool as bt

df_report = pd.read_csv("./乳牛/data/report.csv", error_bad_lines = False)
df_add_report = pd.read_csv("./乳牛/data/data_add/report.csv", error_bad_lines = False)
df_birth = pd.read_csv("./乳牛/data/birth.csv", error_bad_lines = False)
df_submit = pd.read_csv("./乳牛/data/submission.csv", error_bad_lines = False)
df_cow_spec = pd.read_csv("./乳牛/data/spec.csv", error_bad_lines = False)



#合併資料集
df_report = pd.concat([df_report, df_add_report], axis=0)

#%%
#把空值去掉當作訓練依據
df_train = df_report
df_train = df_train.dropna(how='any', subset=["11"])

#%%
#資料預處理成數字
df_report["4"] = df_report["4"]\
    .replace("A",1)\
    .replace("B",2)\
    .replace("C",3)
#%%
#轉換成日期格式
df_cow_spec["4"] = pd.to_datetime(df_cow_spec["4"], format="%Y/%m/%d %H:%M", utc=8)
#df_cow_spec[df_cow_spec["4"].between("1999/08", "1999/09")]
#%%
df_train["dataDate"] = df_train["2"].astype("str") +"/"+ df_train["3"].astype("str")
df_train["dataDate"] = pd.to_datetime(df_train["dataDate"], format="%Y/%m", utc=8)
df_train[["5", "dataDate"]]
#%%
#df_train["dataDate"] + datetime.timedelta(days = 10)
#%%

#%%
