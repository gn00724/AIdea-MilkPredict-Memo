#%%
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_train = pd.read_csv("./data/train.csv", error_bad_lines = False)
df_report = pd.read_csv("./data/report.csv", error_bad_lines = False)
df_birth = pd.read_csv("./data/birth.csv", error_bad_lines = False)
df_submit = pd.read_csv("./data/submission.csv", error_bad_lines = False)

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
#%%
d_framAbyMonthsMean.describe()
sns.relplot(data=d_framAbyMonthsMean)

#%%
d_framAbyBirTimesMean = pd.Series(d_framA.groupby(d_framA["18"])["11"].mean())
sns.relplot(data=d_framAbyBirTimesMean)

#%%
b1 = ["18","4","3"]
b1_Model = RandomForestClassifier(random_state=2, n_estimators=250, min_samples_split=20, oob_score=True)
b1_Model.fit(X[b1],Y_raw.astype('int'))
#%%
#print(b1_Model.oob_score_)

#%%
df_submit
#%%
b1_pred = b1_Model.predict(X_raw[b1])
#%%
submit2 = pd.DataFrame({"ID": X_raw["1"], "預測乳量":b1_pred.astype(int)})
#%%
submit2.to_csv("submit.csv", index=False)

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