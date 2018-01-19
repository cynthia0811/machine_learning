# -*- coding:utf-8 -*-
import numpy as np 
import pandas as pd 
import xgboost as xgb 
import math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer,LabelEncoder
from sklearn.metrics import mean_squared_error


def data_preprocessing():
    """
    数据预处理
    """

    pd_train = pd.read_csv('./data/titanic/train.csv')
    pd_test = pd.read_csv('./data/titanic/test.csv')
    pd_gender = pd.read_csv('./data/titanic/gender_submission.csv')
    print(pd_train.shape, pd_test.shape)
    
    # test和gender 合并
    pd_test = pd.merge(pd_test, pd_gender, on='PassengerId')
    print(pd_train.shape, pd_test.shape)

    # 性别 将性别字段Sex中的值 female用0，male用1代替,类型 int
    pd_train['Sex'] = pd_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
    pd_test['Sex'] = pd_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
    # print(pd_test.columns)

    # 将类型变量转换位连续变量
    for f in pd_train.columns:
        if pd_train[f].dtype == 'object':
            label = LabelEncoder()
            label.fit(list(pd_train[f].values))
            pd_train[f] = label.transform(list(pd_train[f].values))

    for f in pd_test.columns:
        if pd_test[f].dtype == 'object':
            label = LabelEncoder()
            label.fit(list(pd_test[f].values))
            pd_test[f] = label.transform(list(pd_test[f].values))

    # 统计缺失的列
    na_train = pd_train.isnull().sum().sort_values(ascending=False)
    print(na_train)

    # 使用均值填充缺失值
    train_data= pd_train.values
    imput = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imput = imput.fit(train_data)
    train_data = imput.fit_transform(train_data)

    # 使用均值填充缺失值
    test_data= pd_test.values
    imput = Imputer(missing_values="NaN", strategy="mean", axis=0)
    imput = imput.fit(test_data)
    test_data = imput.fit_transform(test_data)

    pd_train = pd.DataFrame(train_data, index=None, columns=['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                                             'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'])

    pd_test = pd.DataFrame(train_data, index=None, columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
                                                            'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived'])
                                                           
    na_train = pd_train.isnull().sum().sort_values(ascending=False)
    na_test = pd_test.isnull().sum().sort_values(ascending=False)
    print("缺失值处理后：")
    print(na_train)
    print(na_test)
    # print(pd_train.head())

    # 保存新数据
    pd_train.to_csv('./data/titanic/new_train.csv')
    pd_train.to_csv('./data/titanic/new_test.csv')


def load_data():
    train_data = pd.read_csv('./data/titanic/new_train.csv')
    test_data = pd.read_csv('./data/titanic/new_test.csv')

    X = train_data.drop(['Survived', 'PassengerId'], 1)
    y = train_data['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

    return X_train, X_test, y_train, y_test


def train_logreistic():
    """
    逻辑回归
    """

    X_train, X_test, y_train, y_test = load_data()

    model = LogisticRegression(penalty='l2')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rfc_rate, rmse = calc_accuracy(y_pred, y_test)
    total = total_survival(y_pred)

    print("LogisticRegression：{0},RMS:{1},存活：{3}".format(rfc_rate, rmse, total))



def calc_accuracy(y_pred, y_true):
    """
    计算精度
    """
    acc = y_pred.ravel() == y_true.ravel()
    acc_rate = 100 * float(acc.sum()) / y_pred.size
    # rmse=np.sqrt(mean_squared_error(y_pred,y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    return acc_rate, rmse


def total_survival(y_pred):
    """
    存活人数
    """
    total = 0
    for value in y_pred:
        if value == 1:
            total += 1
    
    return total



if __name__ == '__main__':
    
    # data_preprocessing()
    # load_data()

    train_logreistic()
