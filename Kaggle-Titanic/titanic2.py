# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:36:08 2017

@author: Phoenix11
"""

# coding: utf-8
import numpy as np
import pandas as pd
import re
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
    df_Miss = df.loc[(df['Name'].str.contains('Miss')),['Age']]
    df_Mrs = df.loc[(df['Name'].str.contains('Mrs')),['Age']]
    df_Mr = df.loc[(df['Name'].str.contains('Mr')),['Age']]
    if "Miss" in df.Name:
        df['Age'] = df["Age"].fillna(df_Miss['Age'].median())
    elif "Mrs" in df.Name:
        df['Age'] = df["Age"].fillna(df_Mrs['Age'].median())
    else: 
        df['Age'] = df["Age"].fillna(df_Mr['Age'].median())
    return df
# 处理缺失的Age
#data_train,rfr = set_missing_age(data_train)
data_train = set_missing_age(data_train)

#处理Cabin
#根据Cabin有无把Cabin转化为Yes和No
def set_Cabinyn_type(df):
    df["Cabinyn"] = ''
    df.loc[(df.Cabin.notnull()),'Cabinyn'] = 'Yes'
    df.loc[(df.Cabin.isnull()),'Cabinyn'] = 'No'  #必须要先Yes再No，否则有了No之后都不是null了
    return df
data_train = set_Cabinyn_type(data_train)

#根据Cabin A B C D E F G 多个类型
#注意有些字母同时存在
def set_Cabinletter_type(df):
    #####注意有一个Cabin是T，非常特别，在这里做0处理了
    df["Cabinletter"] = 0
    df.loc[(df['Cabin'].str.startswith('A') == True),'Cabinletter'] = 1
    df.loc[(df['Cabin'].str.startswith('B') == True),'Cabinletter'] = 2
    df.loc[(df['Cabin'].str.startswith('C') == True),'Cabinletter'] = 3
    df.loc[(df['Cabin'].str.startswith('D') == True),'Cabinletter'] = 4
    df.loc[(df['Cabin'].str.startswith('E') == True),'Cabinletter'] = 5
    df.loc[(df['Cabin'].str.startswith('F') == True),'Cabinletter'] = 6
    df.loc[(df['Cabin'].str.startswith('G') == True),'Cabinletter'] = 7
    return df  
data_train = set_Cabinletter_type(data_train)

#根据Cabin的数值大小
def set_Cabinnum_type(df):
    df.loc[(df.Cabin.isnull()),'Cabin'] = '0'
    df["Cabinnum"] = 0
    i = 0
    #如果只有一个字母，注意这个时候不能用group(),返回的是None即不能用group()
    while (i<len(df)):
        Cabin_num = re.search('(\d+)',df.Cabin[i])
        if Cabin_num == None:
            df.Cabinnum[i] = 0
        else:
            df.Cabinnum[i] = re.search('(\d+)',df.Cabin[i]).group()
        i = i+1
    #把Cabin转化成int
    df.Cabinnum = df.Cabinnum.convert_objects(convert_numeric=True)
    return df
data_train = set_Cabinnum_type(data_train)

#Embarked: 直接扔掉(889)
#data_train = data_train.dropna(subset = ["Embarked"])

#构造新特征SP
def set_SP_type(df):
    df["SP"] = df["Parch"]+df["SibSp"]
    df.loc[(df.SP != 0),'SP'] = 'Yes'
    df.loc[(df.SP == 0),'SP'] = 'No'  #必须要先Yes再No，否则有了No之后都不是null了
    return df
data_train = set_SP_type(data_train)

#构造新特征Chi(Child)
def set_Chi_type(df):
    df["Chi"] = 0
    df.loc[(df.Age <= 12),'Chi'] = 1
    return df    
data_train = set_Chi_type(data_train) 

#构造新特征Old
def set_Old_type(df):
    df["Old"] = 0
    df.loc[(df.Age >= 50),'Old'] = 1
    return df    
data_train = set_Old_type(data_train) 
    
#构造新特征Mother
def set_Mother_type(df):
    df['Mother'] = 0
    df.loc[(df.Name.str.contains('Mrs') & (df.Parch>=1)),['Mother']]=1
    return df    
data_train = set_Mother_type(data_train) 

#构造新特征PS(Pclass+Sex)
def set_PS_type(df):
    df['PS'] = 0
    df.loc[((df['Pclass'] == 1) & (df['Sex'] == "female")),'PS'] = 1
    df.loc[((df['Pclass'] == 2) & (df['Sex'] == "female")),'PS'] = 2
    df.loc[((df['Pclass'] == 3) & (df['Sex'] == "female")),'PS'] = 3
    df.loc[((df['Pclass'] == 1) & (df['Sex'] == "male")),'PS'] = 4
    df.loc[((df['Pclass'] == 2) & (df['Sex'] == "male")),'PS'] = 5
    df.loc[((df['Pclass'] == 3) & (df['Sex'] == "male")),'PS'] = 6
    return df
data_train = set_PS_type(data_train)  


#用哑变量矩阵 get_dummy，全部转换成0,1数值类型
dummies_PS = pd.get_dummies(data_train["PS"],prefix = "PS")
dummies_Cabinyn = pd.get_dummies(data_train["Cabinyn"],prefix = "Cabinyn")
dummies_Cabinletter = pd.get_dummies(data_train["Cabinletter"],prefix = "Cabinletter")
dummies_SP = pd.get_dummies(data_train["SP"],prefix = "SP")
dummies_Embarked = pd.get_dummies(data_train["Embarked"],prefix = "Embarked")
dummies_Sex = pd.get_dummies(data_train["Sex"],prefix = "Sex")
dummies_Pclass = pd.get_dummies(data_train["Pclass"],prefix = "Pclass")
df = pd.concat([data_train,dummies_PS,dummies_Cabinyn,dummies_Cabinletter,dummies_SP,dummies_Embarked,dummies_Pclass,dummies_Sex], axis =1)
df.drop(["Embarked","Cabin","Cabinyn","Cabinletter","Pclass","Sex","Name","Ticket","PS"],axis = 1,inplace = True)

#归一化Age,Fare,Cabin
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()
age_scale = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'],age_scale)
fare_scale = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'],fare_scale)
cabinnum_scale = scaler.fit(df['Cabinnum'])
df['Cabinnum_scaled'] = scaler.fit_transform(df['Cabinnum'],cabinnum_scale)

#逻辑回归建模
from sklearn import linear_model
#取出需要用的属性并转成numpy
train_df = df.filter(regex = 'Survived|Age_.*|SP_.*|Fare_.*|Cabinnum_.*|Cabinyn_.*|Cabinletter_.*|PS_.*|Sex_.*|Embarked_.*|Pclass_.*|Chi|Mother|Old')
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
#tmp_df = data_test[["Name","Age","Fare","Parch","SibSp","Pclass"]]
#tmp_df = set_missing_age(tmp_df)
#null_age = tmp_df[tmp_df.Age.isnull()].as_matrix()
#X = null_age[:,1:]
#predictedAges = rfr.predict(X)
#data_test.loc[(data_test.Age.isnull()),"Age"] = predictedAges
data_test = set_missing_age(data_test)
#处理Cabin
data_test = set_Cabinyn_type(data_test)
data_test = set_Cabinletter_type(data_test)
data_test = set_Cabinnum_type(data_test)
#处理SP
data_test = set_SP_type(data_test)
#处理Chi
data_test = set_Chi_type(data_test)
#处理Old
data_test = set_Old_type(data_test)
#处理Mother
data_test = set_Mother_type(data_test)
#处理PS
data_test = set_PS_type(data_test)

#化为0,1
dummies_PS = pd.get_dummies(data_test["PS"],prefix = "PS")
dummies_Cabinyn = pd.get_dummies(data_test["Cabinyn"],prefix = "Cabinyn")
dummies_Cabinletter = pd.get_dummies(data_test["Cabinletter"],prefix = "Cabinletter")
dummies_SP = pd.get_dummies(data_test["SP"],prefix = "SP")
dummies_Embarked = pd.get_dummies(data_test["Embarked"],prefix = "Embarked")
dummies_Sex = pd.get_dummies(data_test["Sex"],prefix = "Sex")
dummies_Pclass = pd.get_dummies(data_test["Pclass"],prefix = "Pclass")
df_test = pd.concat([data_test,dummies_PS,dummies_Cabinyn,dummies_Cabinletter,dummies_SP,dummies_Embarked,dummies_Pclass,dummies_Sex], axis =1)
df_test.drop(["Embarked","Cabin","Cabinyn","Cabinletter","Pclass","Sex","Name","Ticket","PS"],axis = 1,inplace = True)
#归一化
df_test["Age_scaled"] = scaler.fit_transform(df_test.Age,age_scale)
df_test["Fare_scaled"] = scaler.fit_transform(df_test.Fare,fare_scale)
df_test["Cabinnum_scaled"] = scaler.fit_transform(df_test.Cabinnum,cabinnum_scale)

#预测结果
test = df_test.filter(regex = 'Age_.*|SP_.*|Fare_.*|Cabinyn_.*|Cabinletter_.*|Cabinnum_.*|PS_.*|Sex_.*|Embarked_.*|Pclass_.*|Chi|Mother|Old')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(),'Survived':predictions.astype(np.int32)})
result.to_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/predictions2.csv",index = False)

'''
############################################################################
#模型融合
from sklearn.ensemble import BaggingRegressor
train_df = df.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabinyn_.*|Cabinletter_.*|Cabinnum_.*|PS_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Chi|Old')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True, bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SP_.*|Fare_.*|Cabinyn_.*|Cabinletter_.*|Cabinnum_.*|PS_.*|Sex_.*|Embarked_.*|Pclass_.*|Chi|Mother|Old')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/predictions2.csv", index=False)

###############################################################################
'''
#看一看特征的贡献，把model系数和feature关联一下
pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})

#交叉验证
from sklearn import cross_validation
clf = linear_model.LogisticRegression(C = 1.0,penalty = 'l1', tol = 1e-6)
all_data = df.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabinyn_.*|Cabinletter_.*|Cabinnum_.*|PS_.*|Embarked_.*|Sex_.*|Pclass_.*|Chi|Mother|Old')
X = all_data.as_matrix()[:,1:]
y = all_data.as_matrix()[:,0]
print (cross_validation.cross_val_score(clf,X,y,cv=5))

#分割数据
split_train,split_cv = cross_validation.train_test_split(df,test_size = 0.3,random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabinyn_.*|Cabinletter_.*|Cabinnum_.*|PS_.*|Embarked_.*|Sex_.*|Pclass_.*|Chi|Mother|Old')
clf = linear_model.LogisticRegression(C = 1.0,penalty = 'l1', tol = 1e-6)
clf.fit(train_df.as_matrix()[:,1:],train_df.as_matrix()[:,0])
#对cross validation进行预测
cv_df = split_cv.filter(regex='Survived|Age_.*|SP_.*|Fare_.*|Cabinyn_.*|Cabinletter_.*|Cabinnum_.*|PS_.*|Embarked_.*|Sex_.*|Pclass_.*|Chi|Mother|Old')
predictions = clf.predict(cv_df.as_matrix()[:,1:])

original_data_train = pd.read_csv("D:/YCR/Study/Statistic&Machine Learning/Kaggle/Titanic/train.csv")
bad_cases = original_data_train.loc[original_data_train["PassengerId"].isin(split_cv[predictions!=cv_df.as_matrix()[:,0]]["PassengerId"].values)]

#画学习率曲线
from sklearn.learning_curve import learning_curve
import matplotlib.pyplot as plt
def plot_learning_curve(estimator, title, X, y , ylim = None, cv = None, n_jobs = 1, train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"m")
        plt.ylabel(u"score")
        plt.gca().invert_yaxis()
        plt.grid()
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"train")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"cv")
        plt.legend(loc="best")
        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()
        
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
plot_learning_curve(clf, u"learning curve", X, y)



