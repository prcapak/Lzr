#-*- coding: UTF-8 -*-
#!/usr/bin/python
#-*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
import sklearn.preprocessing as pp
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import csv

data_train = pd.read_csv('D:\\train.csv')

### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']] # 把已有的数值型特征取出来丢进Random Forest Regressor中
    known_age = age_df[age_df.Age.notnull()].as_matrix()  # 乘客分成已知年龄和未知年龄两部分
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    y = known_age[:, 0] # y即目标年龄
    X = known_age[:, 1:] # X即特征属性值
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1) # fit到RandomForestRegressor之中
    rfr.fit(X, y)
    predictedAges = rfr.predict(unknown_age[:, 1::]) # 用得到的模型进行未知年龄结果预测
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges # 用得到的预测结果填补原缺失数据
    return df, rfr


def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')
df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)


scaler = pp.StandardScaler()
age_scale_param = scaler.fit(df['Age'].reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].reshape(-1, 1), age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].reshape(-1, 1), fare_scale_param)

#df.to_csv('D:\\prepro.csv', index=False)



train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*') # 用正则取出我们要的属性值
train_np = train_df.as_matrix()
y = train_np[:, 0] # y即Survival结果
X = train_np[:, 1:] # X即特征属性值
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6) # fit到RandomForestRegressor之中
clf.fit(X, y)



data_test = pd.read_csv("D:\\test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0

tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']] # 接着我们对test_data做和train_data中一致的特征变换
null_age = tmp_df[data_test.Age.isnull()].as_matrix() # 首先用同样的RandomForestRegressor模型填上丢失的年龄
X = null_age[:, 1:] # 根据特征属性X预测年龄并补上
predictedAges = rfr.predict(X)
data_test.loc[ (data_test.Age.isnull()), 'Age' ] = predictedAges
data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].reshape(-1, 1), age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].reshape(-1, 1), fare_scale_param)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].as_matrix(), 'Survived':predictions.astype(np.int32)})
result.to_csv("D:\\predictions.csv", index=False)