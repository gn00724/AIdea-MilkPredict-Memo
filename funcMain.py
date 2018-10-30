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

train = pd.read_csv("./rawData/train.csv")
test = pd.read_csv("./rawData/test.csv")
submit = pd.read_csv("./rawData/gender_submission.csv")
#%%
#1.先觀察資料及有沒有缺損
#發現Total資料樹不一樣，需要補缺損內容
train.info()
test.info()
#%%
#2.觀察資料的大概數值分佈
train.describe()
test.describe()

#%%
#3.最重要的是找出有用的特徵值
sns.countplot(data["Survived"])
#%%
sns.countplot(data["Pclass"], hue=data["Survived"])
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

