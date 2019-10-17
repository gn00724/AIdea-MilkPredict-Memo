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
X = df_train
X_raw = df_report
X["11"] = X["11"].dropna(axis=0)
Y_raw = pd.DataFrame(df_train["11"])

#%%
Y_raw.describe()
#%%
X["10"] = X["10"].fillna(0)
X["10"].describe()

#%%月份與平均乳量的關係 | 牧場關係很大
#觀察
d_fram = pd.Series(df_train.groupby(df_train["4"])["11"].mean())
d_fram.describe()
#%%
#只觀察牧場，看月份有無關係 | 好像有一點
d_framA = pd.DataFrame(df_train)
d_framA = d_framA[d_framA.loc[:,"4"] == 1]
d_framAbyMonthsMean = pd.Series(d_framA.groupby(d_framA["3"])["11"].mean())
d_framAbyMonthsMean.describe()
sns.relplot(data=d_framAbyMonthsMean)
#%%
#把南部的牧場攪和一起 | 預測分數會下降
tmpA = []
for x in df_train["4"]:
    code = x
    if x != 1:
        code = 2
    tmpA.append(code)
df_train["FramCode"] = pd.Series(tmpA)

#%%
MonthTypeCodeDict = {}
MonthType = pd.qcut(pd.Series(df_train.groupby(df_train["3"])["11"].mean()),5)
sns.relplot(data=pd.Series(d_framA.groupby(df_train["3"])["11"].mean()))
for m in range(12):
    MonthTypeCodeDict[m+1] = MonthType.values.codes[m]

#%%
#觀察圖表，把月份切成三個單位，好像比較好用 | 4個預測分數會下降
df_train["MonthCode"] = df_train["3"].map(MonthTypeCodeDict).astype("int")
sns.relplot(data=pd.Series(d_framA.groupby(df_train["MonthCode"])["11"].mean()))

#%%
#年份和乳量的關係
YearsFact = pd.Series(df_train.groupby(df_train["2"])["11"].mean())
sns.relplot(data=YearsFact)

#%%
d_YearsType = pd.qcut(YearsFact,3)
print(d_YearsType.values.codes)
yearsTypeDict = {}

for m in df_train["2"]:
    yearsTypeDict[m] = d_YearsType.values.codes[m-2013]

df_train["YearsType"] = df_train["2"].map(yearsTypeDict).astype("int")
df_train["YearsType"].describe()



#只觀察牧場，觀察泌乳時間分佈
#%%
d_framAbyMilkDuration = pd.Series(d_framA.groupby(d_framA["10"])["11"].mean())
d_framAbyMilkDuration.describe()
sns.relplot(data=d_framAbyMilkDuration)
#%%
d_MilkDuraiton = pd.qcut(d_framAbyMilkDuration,2)
d_MilkDuraiton.values.categories[1]
MilkDuraitonDict = {}

for m in df_train["10"]:
    if m in d_MilkDuraiton.values.categories[1]:
        MilkDuraitonDict[m] = 1
    else:
        MilkDuraitonDict[m] = 0

df_train["MilkDuraiton"] = df_train["10"].map(MilkDuraitonDict).astype("int")
df_train["MilkDuraiton"]

#%%
d_framAbyBreedTimesMean = pd.Series(d_framA.groupby(d_framA["18"])["11"].mean())
sns.relplot(data=d_framAbyBreedTimesMean)

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
d_DayAfterBreed = pd.Series(df_train.groupby(df_train["DayAfterBreed"])["11"].mean())
d_DayAfterBreed.describe()
#%%
sns.relplot(data=d_DayAfterBreed)
#d_framAbyDayAfterBreed = pd.Series(d_framA.groupby(d_framA["12"])["11"].mean())
#sns.relplot(data=d_framAbyDayAfterBreed)

#%%
TypeDayAfterBreed = pd.qcut(df_train["DayAfterBreed"],10)
d_typedayAfterBreed = TypeDayAfterBreed.values.codes


#%%
df_train["DayAfterBreed"] = pd.Series(d_typedayAfterBreed).astype("int")
df_train["DayAfterBreed"].describe()
#%%
df = pd.Series(df_train.groupby(df_train["DayAfterBreed"])["11"].mean())
sns.relplot(data=df)


#%%
#年齡資料後來的變異數太大
d_framAYearsOld = pd.Series(d_framA.groupby(d_framA["14"])["11"].mean())
d_framAYearsOld.describe()
sns.relplot(data=d_framAYearsOld)

#%%
#胎次好像可以有明顯分區
d_framABirthTimes = pd.Series(df_train.groupby(df_train["9"])["11"].mean())
d_framABirthTimes.describe()
sns.relplot(data=d_framABirthTimes)
#%%
d_BirthTimesType = pd.qcut(d_framABirthTimes,2)
d_BirthTimesType.values.categories[1]
BirthTimesDict = {}

for m in df_train["9"]:
    if m in d_BirthTimesType.values.categories[1]:
        BirthTimesDict[m] = 1
    else:
        BirthTimesDict[m] = 0

df_train["BirthTimesType"] = df_train["9"].map(BirthTimesDict).astype("int")
df_train["BirthTimesType"]

sns.relplot(data=pd.Series(df_train.groupby(df_train["BirthTimesType"])["11"].mean()))


#%%
b1 = ["YearsType","4","MonthCode","DayAfterBreed"]
b1_Model = RandomForestClassifier(random_state=10, n_estimators=250, min_samples_split=20, oob_score=True)
b1_Model.fit(X[b1],Y_raw.astype('int'))
#%%
print(b1_Model.oob_score_)

#---------Pred End

#%%
#把剛剛中途創建公式欄位的都加在預測的欄位中
X_raw["MonthCode"] = X_raw["3"].map(MonthTypeCodeDict).astype("int")

predBirthTimesDict = {}
predMilkDuraitonDict = {}
predYersTypeDict = {}
#%%
X_raw["10"] = X_raw["10"].fillna(0)
#%%
for m in X_raw["10"]:
    if m in d_MilkDuraiton.values.categories[1]:
        predMilkDuraitonDict[m] = 1
    else:
        predMilkDuraitonDict[m] = 0

for m in X_raw["9"]:
    if m in d_BirthTimesType.values.categories[1]:
        predBirthTimesDict[m] = 1
    else:
        predBirthTimesDict[m] = 0
#%%
for m in X_raw["2"]:
    try:
        predYersTypeDict[m] = d_YearsType.values.codes[m-2013]
    except:
        predYersTypeDict[m] = d_YearsType.values.codes[-1]

#%%
X_raw["12"] = pd.to_datetime(X_raw["12"], format="%Y/%m/%d %H:%M")
X_raw["15"] = pd.to_datetime(X_raw["15"], format="%Y/%m/%d %H:%M")
X_raw["DayAfterBreed"] = X_raw["15"] - X_raw["12"]

xrawTypeDayAfterBreed = pd.qcut(X_raw["DayAfterBreed"],10)
xraw_typedayAfterBreed = xrawTypeDayAfterBreed.values.codes

#%%
X_raw["DayAfterBreed"] = pd.Series(xraw_typedayAfterBreed).astype("int")
X_raw["DayAfterBreed"].describe()

#%%
X_raw["YearsType"] = X_raw["2"].map(predYersTypeDict).astype("int")
X_raw["YearsType"]

#%%
X_raw["MilkDuraiton"] = X_raw["10"].map(predMilkDuraitonDict).astype("int")
#%%
X_raw["BirthTimesType"] = X_raw["9"].map(predBirthTimesDict).astype("int")


#%%
b1_pred = b1_Model.predict(X_raw[b1])
#%%
submit2 = pd.DataFrame({"ID": X_raw["1"], "預測乳量":b1_pred.astype(int)})
#%%
submit2.to_csv("submit.csv", index=False)











#%%
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
df_train['11'].describe()
#%%
X = df_train['3'].values.reshape(-1,1)
Y = df_train['11'].values.reshape(-1,1)

#%%
regressor.fit(X, Y)
#%%
XR = df_report['3'].values.reshape(-1,1)
YR = df_report['11'].values.reshape(-1,1)

YR = YR.astype('int')
YR
#%%
regressor.score(XR, YR)
#%%
b2_pred = regressor.predict(XR)
#%%
df = pd.DataFrame({'ID': df_report['1'],'Actual': YR.flatten(), 'Predicted': b2_pred.flatten()})
df.to_csv("submit_line.csv", index=False)

#%%
"""

#%%
sns.countplot(df_report["11"], hue=df_report["3"])
#%%
#嘗試分析客艙死亡率與平均死亡率
SurvivedMean = data["Survived"].mean()
fig, ax = plt.subplots(figsize = (5,4))
Pclass_Percent = pd.DataFrame(train.groupby(data['Pclass'])["Survived"].mean())

Pclass_Percent.plot.bar(ax=ax)
ax.axhline(SurvivedMean,linestyle='dashed', c='black',alpha = .3)
#%%
#嘗試分析男女性與平均死亡率
fig, ax = plt.subplots(figsize = (5,4))
Pclass_Percent = pd.DataFrame(train.groupby(data['Sex'])["Survived"].mean())
Pclass_Percent.plot.bar(ax=ax)
ax.axhline(SurvivedMean,linestyle='dashed', c='black',alpha = .3)

#%%
#嘗試分析年齡對於生存率的關鍵轉折點，作為二分法特徵
fig, ax = plt.subplots(figsize = (5,4))
g = sns.FacetGrid(data, col="Survived")
g.map(sns.distplot, "Age", kde=True)
g.map(sns.distplot, "Age", kde=True, ax=ax)
#%%
g = sns.FacetGrid(data, col="Survived")
g.map(sns.distplot, "Fare", kde=False)
#%%
g = sns.FacetGrid(data, col="Survived")
g.map(sns.distplot, "Parch", kde=False)
#%%
g = sns.FacetGrid(data, col="Survived")
g.map(sns.distplot, "SibSp", kde=False)

#%%
data["Family_Size"] = data["Parch"] + data["SibSp"]
g = sns.FacetGrid(data, col="Survived")
g.map(sns.distplot, "Family_Size", kde=False)
#%%
data["Title1"] = data["Name"].str.split(", ", expand=True)[1]
data["Name"].str.split(", ", expand=True).head(3)
#%%
data["Title1"].head(3)
#%%
data["Title1"] = data["Name"].str.split(", ", expand=True)[0]
data["Name"].str.split(", ", expand=True).head(3)
data["Title1"].unique()
#%%
pd.crosstab(data["Title1"],data["Sex"]).T.style.background_gradient(cmap="summer_r")
#%%
pd.crosstab(data["Title1"],data["Survived"]).T.style.background_gradient(cmap="summer_r")

#%%
data.groupby(["Title1"])["Age"].mean()
#%%
data["Title2"] = data["Title1"].replace(["Mlle","Mme","Ms","Dr","Major","Lady","the Countess"],["Miss","Mrs","Miss","Mr","Mr","Mrs","Mrs"])
data["Title2"].unique()
#%%
pd.crosstab(data["Title2"],data["Sex"]).T.style.background_gradient(cmap="summer_r")
#%%
pd.crosstab(data["Title2"],data["Survived"]).T.style.background_gradient(cmap="summer_r")
#%%
data["Ticket_info"] = data["Ticket"].apply(lambda x : x.replace(".","").replace("/","").strip(" "))
data["Ticket_info"].unique()

#%%
data["Embarked"] = data["Embarked"].fillna("S")
data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
data["Cabin"].head(10)
data["Cabin"] = data["Cabin"].apply(lambda x : str(x)[0] if not pd.isnull(x) else "NoCabin")
data["Cabin"].unique()
#%%
sns.countplot(data["Cabin"], hue=data["Survived"])
data["Sex"] = data["Sex"].astype("category").cat.codes
data["Embarked"] = data["Embarked"].astype("category").cat.codes
data["Pclass"] = data["Pclass"].astype("category").cat.codes
data["Title1"] = data["Title1"].astype("category").cat.codes
data["Title2"] = data["Title2"].astype("category").cat.codes
data["Cabin"] = data["Cabin"].astype("category").cat.codes
data["Ticket_info"] = data["Ticket_info"].astype("category").cat.codes
#%%
dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))                     
                     ]
rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)
ageColumns = ['Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title1', 'Title2','Cabin','Ticket_info']
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(X= dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)
#%%
dataTrain = data[pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
dataTest = data[~pd.notnull(data['Survived'])].sort_values(by=["PassengerId"])
#%%
dataTrain.columns
dataTrain = dataTrain[['Survived', 'Age', 'Embarked', 'Fare',  'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTest = dataTest[['Age', 'Embarked', 'Fare', 'Pclass', 'Sex', 'Family_Size', 'Title2','Ticket_info','Cabin']]
dataTrain
#%%
from sklearn.ensemble import RandomForestClassifier
 
rf = RandomForestClassifier(criterion='gini', 
                             n_estimators=1000,
                             min_samples_split=12,
                             min_samples_leaf=1,
                             oob_score=True,
                             random_state=1,
                             n_jobs=-1) 

rf.fit(dataTrain.iloc[:, 1:], dataTrain.iloc[:, 0])
print("%.4f" % rf.oob_score_)

pd.concat((pd.DataFrame(dataTrain.iloc[:, 1:].columns, columns = ['variable']), 
           pd.DataFrame(rf.feature_importances_, columns = ['importance'])), 
          axis = 1).sort_values(by='importance', ascending = False)[:20]
rf_res =  rf.predict(dataTest)
submit['Survived'] = rf_res
submit['Survived'] = submit['Survived'].astype(int)
submit.to_csv('submit.csv', index= False)
submit

"""