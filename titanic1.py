# coding: utf-8
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from sklearn.ensemble import RandomForestRegressor

data_train = pd.read_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/train.csv")


#缺失值处理:Age, Cabin, Embarked
#Age：以平均值填补(714)
#data_train.Age = data_train.Age.fillna(data_train.Age.median())

#Age: 以RandomForest处理
"""
def set_missing_age(df):
    age_df = df[['Age','Fare','Parch','SibSp','Pclass']]
    #把乘客分成知道年龄的和不知道年龄的
    know_age = age_df[age_df.Age.notnull()].as_matrix()
    unknow_age = age_df[age_df.Age.isnull()].as_matrix()
    #y即目标年龄
    y = know_age[:,0]
    #x即特征属性值
    X = know_age[:,1:]
    #fit到RandomForestRegressor中
    rfr = RandomForestRegressor(random_state = 0, n_estimators = 2000,n_jobs = -1)
    rfr.fit(X,y)
    predictedAges = rfr.predict(unknow_age[:,1::])
    #用得到的预测结果填补原始数据
    df.loc[(df.Age.isnull()),'Age'] = predictedAges
    return df,rfr
"""
#Age:以平均值处理
def set_missing_age(df):
    #用split?分成Miss,Mrs,Mr
    
    return df,rfr

#把Cabin转化为Yes和No
def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()),'Cabin'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabin'] = 'No'  #必须要先Yes再No，否则有了No之后都不是null了
    return df

# 处理缺失的Age
data_train,rfr = set_missing_age(data_train)

#Embarked: 直接扔掉(889)
data_train = data_train.dropna(subset = ["Embarked"])

#把Cabin有无转化为Yes No
data_train = set_Cabin_type(data_train)

#构造新特征SP
def set_SP_type(df):
    df["SP"] = df["Parch"]+df["SibSp"]
    df.loc[(df.SP != 0),'SP'] = 'Yes'
    df.loc[(df.SP == 0),'SP'] = 'No'  #必须要先Yes再No，否则有了No之后都不是null了
    return df
data_train = set_SP_type(data_train)

#构造新特征OldChi(OldChild)
def set_OldChi_type(df):
    df["OldChi"] = 0
    df.loc[(df.Age <= 12),'OldChi'] = 1
    df.loc[(df.Age >= 60),'OldChi'] = 1
    return df    
data_train = set_OldChi_type(data_train) 
    
#用哑变量矩阵 get_dummy，全部转换成0,1数值类型
dummies_SP = pd.get_dummies(data_train["SP"],prefix = "SP")
dummies_Cabin = pd.get_dummies(data_train["Cabin"],prefix = "Cabin")
dummies_Embarked = pd.get_dummies(data_train["Embarked"],prefix = "Embarked")
dummies_Sex = pd.get_dummies(data_train["Sex"],prefix = "Sex")
dummies_Pclass = pd.get_dummies(data_train["Pclass"],prefix = "Pclass")
df = pd.concat([data_train,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex], axis =1)
df.drop(["Cabin","Embarked","Pclass","Sex","Name","Ticket"],axis = 1,inplace = True)

#归一化Age,Fare
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale)
fare_scale = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale)


#逻辑回归建模
from sklearn import linear_model
#取出需要用的属性并转成numpy
train_df = df.filter(regex = 'Survived|Age_.*|SP_.*|Fare_.*|Cabin_.*|Sex_.*|Embarked_.*|Pclass_.*|OldChi')
train_np = train_df.as_matrix()
y = train_np[:,0]
X = train_np[:,1:]
clf = linear_model.LogisticRegression(C = 1.0,penalty = 'l1', tol = 1e-6)
clf.fit(X,y)

#处理test.csv
data_test = pd.read_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/test.csv")
#注意test中有一个Fare缺失了
data_test.loc[(data_test.Fare.isnull(),"Fare")]=0
#处理Age
tmp_df = data_test[["Age","Fare","Parch","SibSp","Pclass"]]
null_age = tmp_df[tmp_df.Age.isnull()].as_matrix()
X = null_age[:,1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()),"Age"] = predictedAges
#处理Cabin
data_test = set_Cabin_type(data_test)
#处理SP
data_test = set_SP_type(data_test)
#处理OldChi
data_test = set_OldChi_type(data_test)

#化为0,1
dummies_SP = pd.get_dummies(data_test["SP"],prefix = "SP")
dummies_Cabin = pd.get_dummies(data_test["Cabin"],prefix = "Cabin")
dummies_Embarked = pd.get_dummies(data_test["Embarked"],prefix = "Embarked")
dummies_Sex = pd.get_dummies(data_test["Sex"],prefix = "Sex")
dummies_Pclass = pd.get_dummies(data_test["Pclass"],prefix = "Pclass")
df_test = pd.concat([data_test,dummies_Cabin,dummies_Embarked,dummies_Pclass,dummies_Sex], axis =1)
df_test.drop(["Cabin","Embarked","Pclass","Sex","Name","Ticket"],axis = 1,inplace = True)
#归一化
df_test["Age_scaled"] = scaler.fit_transform(df_test.Age,age_scale)
df_test["Fare_scaled"] = scaler.fit_transform(df_test.Fare,fare_scale)

#预测结果
test = df_test.filter(regex = 'Age_.*|SP_.*|Fare_.*|Cabin_.*|Sex_.*|Embarked_.*|Pclass_.*|OldChi')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/predictions1.csv",index = False)

#交叉验证
from sklearn import cross_validation
clf = linear_model.LogisticRegression(C = 1.0,penalty = 'l1', tol = 1e-6)
all_data = df.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|OldChi')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print (cross_validation.cross_val_score(clf,X,y,cv=5))

#分割数据
split_train,split_cv = cross_validation.train_test_split(df,test_size = 0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|OldChi')
clf = linear_model.LogisticRegression(C = 1.0,penalty = 'l1', tol = 1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
#对cross validation进行预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|OldChi')
predictions = clf.predict(cv_df.as_matrix()[:,1:])

original_data_train = pd.read_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/train.csv")
bad_cases = original_data_train.loc[original_data_train["PassengerId"].isin(split_cv[predictions!=cv_df.as_matrix()[:,0]]["PassengerId"].values)]



