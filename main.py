#%%
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime as dt
import matplotlib.pyplot as plt

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("./乳牛/data/train.csv", error_bad_lines = False)
df_report = pd.read_csv("./乳牛/data/report.csv", error_bad_lines = False)
df_birth = pd.read_csv("./乳牛/data/birth.csv", error_bad_lines = False)
df_submit = pd.read_csv("./乳牛/data/submission.csv", error_bad_lines = False)
splitMonth = 5

#%%
d_framA = df_train[df_train.loc[:,"4"] == 1]
d_framB = df_train[df_train.loc[:,"4"] == 2]
d_framC = df_train[df_train.loc[:,"4"] == 3]

print(d_framA.count(),
    d_framB.count(),
    d_framC.count(),
    df_train.count())

#%%
d_monthtype = []
for _t, m in enumerate(df_train["3"]):
    if m <= splitMonth:
        d_monthtype.append(1)
    elif m > splitMonth and (df_train["2"][_t] == 2015 or df_train["2"][_t] == 2016):
        d_monthtype.append(3)
    elif m > splitMonth:
        d_monthtype.append(2)
    else:
        None
        
df_train["MonthType"] = pd.Series(d_monthtype)
df_train["MonthType"].unique()

#%%
d_splitYears2type = []
for m in df_train["2"]:
    if m == 2015 or m == 2016:
        d_splitYears2type.append(1)      
    else:
        d_splitYears2type.append(2)
        
df_train["splitYears2type"] = pd.Series(d_splitYears2type)
#%%
d_year2013 = df_train[df_train.loc[:,"2"] == 2013]
d_year2014 = df_train[df_train.loc[:,"2"] == 2014]
d_year2015 = df_train[df_train.loc[:,"2"] == 2015]
d_year2016 = df_train[df_train.loc[:,"2"] == 2016]
d_year2017 = df_train[df_train.loc[:,"2"] == 2017]
d_year2018 = df_train[df_train.loc[:,"2"] == 2018]

yearFact_2013 = pd.Series(d_year2013.groupby(d_year2013["MonthType"])["10"].mean().to_dict()).values
yearFact_2014 = pd.Series(d_year2014.groupby(d_year2014["MonthType"])["10"].mean().to_dict()).values
yearFact_2015 = pd.Series(d_year2015.groupby(d_year2015["MonthType"])["10"].mean().to_dict()).values
yearFact_2016 = pd.Series(d_year2016.groupby(d_year2016["MonthType"])["10"].mean().to_dict()).values
yearFact_2017 = pd.Series(d_year2017.groupby(d_year2017["MonthType"])["10"].mean().to_dict()).values
yearFact_2018 = pd.Series(d_year2018.groupby(d_year2018["MonthType"])["10"].mean().to_dict()).values

months = [1,2]

#%%
plt.figure(figsize=(8,5),dpi=100,linewidth = 2)
plt.plot(months,yearFact_2013,'s-', label="2013")
plt.plot(months,yearFact_2014,'s-', label="2014")
plt.plot(months,yearFact_2015,'s-', label="2015")
plt.plot(months,yearFact_2016,'s-', label="2016")
plt.plot(months,yearFact_2017,'s-', label="2017")
plt.plot(months,yearFact_2018,'s-', label="2018")
plt.legend(loc = "best", fontsize=20)
plt.show()
#%%
#init
X = df_train
X_pred = df_report
X["10"] = X["10"].dropna(axis=0)
Y = pd.DataFrame(df_train["10"])

#%%
#缺值檢查
print(X.count(), "\n", Y.count())

#%%
#年份和乳量的關係
YearsFact = pd.Series(df_train.groupby(df_train["2"])["10"].mean())
sns.relplot(data=YearsFact)

#%%
d_YearsType = pd.qcut(YearsFact,4)
print(d_YearsType.values.codes)
d_yearsTypeDict = {}

for m in df_train["2"]:
    d_yearsTypeDict[m] = d_YearsType.values.codes[m-2013]

df_train["YearsType"] = df_train["2"].map(d_yearsTypeDict).astype("int")
df_train["YearsType"].describe()
df_byYears = pd.Series(df_train.groupby(df_train["YearsType"])["10"].mean())
sns.relplot(data=df_byYears)



#%%
#農場影響
FarmFact = pd.Series(df_train.groupby(df_train["4"])["10"].mean())
sns.relplot(data=FarmFact)

#%%
targetFarmDict = {
    "A":"FarmA/C0D390",
    "B":"FarmB/C0G770",
    "C":"FarmC/C0R510"}
pdbyFarmyear = []
#%%
for key in targetFarmDict.keys():
    _tmp = {}
    for y in range(2012, 2020):
        _tmp[y] = pd.read_csv("./乳牛/data/" + targetFarmDict[key] + "-" + str(y) + ".csv", error_bad_lines = False)
    pdbyFarmyear.append(_tmp)

#%%
X = df_train
X_raw = df_report
X["10"] = X["10"].dropna(axis=0)
Y_raw = pd.DataFrame(df_train["10"])

dyearsArray = X["2"].values
dFarmArray = X["4"].values
dMonthArray = X["3"].values
dArrayRH = []
dArrayTemperature = []
LenofData = len(dFarmArray)
bar = 0
#%%
for _t, dfram in enumerate(X["4"].values):
    if (_t / LenofData) >= bar + 0.1:
        print(_t / LenofData)
        bar = _t / LenofData
    _rh = pdbyFarmyear[int(dfram)-1]\
            [int(dyearsArray[_t])]\
            ["RH"]\
            [int(dMonthArray[_t])-1]
    _temp = pdbyFarmyear[int(dfram)-1]\
            [int(dyearsArray[_t])]\
            ["Temperature"]\
            [int(dMonthArray[_t])-1]
    dArrayRH.append(_rh)
    dArrayTemperature.append(_temp)

#%%
df_train["RH"] = pd.Series(dArrayRH)
df_train["Temperature"] = pd.Series(dArrayTemperature)   

df_rh = pd.Series(df_train.groupby(df_train["RH"])["10"].mean())
df_temp = pd.Series(df_train.groupby(df_train["Temperature"])["10"].mean())
sns.relplot(data=df_rh)
sns.relplot(data=df_temp)
#%%
dd = []
d_RHType = pd.qcut(df_rh,6)

dd2 = {}
for _t, key in enumerate(d_RHType.to_dict().keys()):
    dd2[key] = d_RHType.values.codes[_t]

for x in df_train["RH"]:
    dd.append(dd2[x])

df_train["RH_by7"] = pd.Series(dd)
dRH = pd.Series(df_train.groupby(df_train["RH_by7"])["10"].mean())
sns.relplot(data=dRH)

#%%
#d_byTemperature = pd.Series(df_train.groupby(df_train["Temperature"])["10"].mean())
#sns.relplot(data=d_byTemperature)

#%%
df_train["12"] = pd.to_datetime(df_train["12"], format="%Y/%m/%d %H:%M")
df_train["15"] = pd.to_datetime(df_train["15"], format="%Y/%m/%d %H:%M")
df_train["DayAfterBreed"] = df_train["15"] - df_train["12"]

tmpArray = []
for x in df_train["DayAfterBreed"]:
    tmpArray.append(x.days)

df_train["DayAfterBreed"] = pd.Series(tmpArray)
df_train["DayAfterBreed"].describe()

#%%
d_DayAfterBreed = pd.Series(df_train.groupby(df_train["DayAfterBreed"])["10"].mean())
d_DayAfterBreed.describe()
#%%
sns.relplot(data=d_DayAfterBreed)
#d_framAbyDayAfterBreed = pd.Series(d_framA.groupby(d_framA["12"])["10"].mean())
#sns.relplot(data=d_framAbyDayAfterBreed)

#%%
TypeDayAfterBreed = pd.qcut(df_train["DayAfterBreed"],10)
d_typedayAfterBreed = TypeDayAfterBreed.values.codes
TypeDayAfterBreed
#%%
df_train["DayAfterBreed"] = pd.Series(d_typedayAfterBreed).astype("int")
df_train["DayAfterBreed"].describe()
#%%
df = pd.Series(df_train.groupby(df_train["DayAfterBreed"])["10"].mean())
sns.relplot(data=df)
#%%
mancoeDict = {}
mancowArray = []
cowCount = 1
for mancowID in df_train["6"]:
    if mancowID not in mancoeDict:
        mancoeDict[mancowID] = cowCount
        mancowArray.append(cowCount)
        cowCount += 1
    else:
        mancowArray.append(mancoeDict[mancowID])

df_train["ManCow"] = pd.Series(mancowArray)
df_train["ManCow"]

#%%
df = pd.Series(df_train.groupby(df_train["ManCow"])["10"].mean())
sns.relplot(data=df)


#%%
#預測與建模
b1 = ["splitYears2type", "4", "MonthType", "DayAfterBreed"]
b1_Model = RandomForestClassifier(random_state=10, n_estimators=250, min_samples_split=20, oob_score=True)
b1_Model.fit(X[b1],Y.astype('int'))
#%%
print(b1_Model.oob_score_)

#%%
#預測
dpred_monthtype = []
for m in X_pred["3"]:
    if m <= splitMonth:
        dpred_monthtype.append(1)
    elif m > splitMonth:
        dpred_monthtype.append(2)
    else:
        None
        
X_pred["MonthType"] = pd.Series(dpred_monthtype)
#%%
d_predsplitYears2type = []
for m in X_pred["2"]:
    if m == 2015 or m == 2016:
        d_predsplitYears2type.append(1)      
    else:
        d_predsplitYears2type.append(2)
        
X_pred["splitYears2type"] = pd.Series(d_predsplitYears2type)
X_pred["splitYears2type"]
#%%
X_pred["10"]

#%%
YearsFact = pd.Series(X_pred.groupby(X_pred["2"])["10"].mean())
d_YearsType = pd.qcut(YearsFact,4)
d_predYearTypeDict = {}
for m in X_pred["2"]:
    try:
        d_predYearTypeDict[m] = d_YearsType.values.codes[m-2013]
    except:
        d_predYearTypeDict[m] = d_YearsType.values.codes[5]
X_pred["YearsType"] = X_pred["2"].map(d_predYearTypeDict).astype("int")
X_pred["YearsType"].describe()
#%%
bar = 0
dpredArrayTemperature = []


#%%
dPredyearsArray = X_pred["2"].values
dPredFarmArray = X_pred["4"].values
dPredMonthArray = X_pred["3"].values
dPredLenofData = len(dPredFarmArray)

for _t, dfram in enumerate(dPredFarmArray):
    if (_t / dPredLenofData) >= bar + 0.1:
        print(_t / dPredLenofData)
        bar = _t / dPredLenofData
    _temp = pdbyFarmyear[int(dfram)-1]\
            [int(dPredyearsArray[_t])]\
            ["RH"]\
            [int(dPredMonthArray[_t])-1]
    #print(_temp, dfram, dPredyearsArray[_t], dPredMonthArray[_t])
    dpredArrayTemperature.append(int(_temp))
#%%
X_pred["RH"] = pd.Series(dpredArrayTemperature)
X_pred["RH"]
#%%
X_pred["12"] = pd.to_datetime(X_pred["12"], format="%Y/%m/%d %H:%M")
X_pred["15"] = pd.to_datetime(X_pred["15"], format="%Y/%m/%d %H:%M")
X_pred["DayAfterBreed"] = X_pred["15"] - X_pred["12"]

tmpArray = []
for x in X_pred["DayAfterBreed"]:
    tmpArray.append(x.days)

X_pred["DayAfterBreed"] = pd.Series(tmpArray)
dPredTypeDayAfterBreed = pd.qcut(X_pred["DayAfterBreed"],10)
d_predtypedayAfterBreed = dPredTypeDayAfterBreed.values.codes
X_pred["DayAfterBreed"] = pd.Series(d_predtypedayAfterBreed).astype("int")

#%%
predmanCowArray = []
for mancowID in X_pred["6"]:
    if mancowID not in mancoeDict:
        cowCount += 1
        mancoeDict[mancowID] = cowCount
        predmanCowArray.append(cowCount)
    else:
        predmanCowArray.append(mancoeDict[mancowID])

X_pred["ManCow"] = pd.Series(predmanCowArray)
X_pred["ManCow"]


#%%
b1_pred = b1_Model.predict(X_pred[b1])
#%%
submit2 = pd.DataFrame({"ID": X_pred["1"], "預測乳量":b1_pred.astype(int)})
#%%
submit2.to_csv("submit" + ".csv", index=False)

#%%
print("Done")

#%%
