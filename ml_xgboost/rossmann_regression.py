# -*- coding:utf-8 -*-
import os
import datetime
import operator
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, scale
import matplotlib.pyplot as plt

np.random.seed(19260817)
plt.rcParams['font.sans-serif'] = ['simHei']
plt.rcParams['axes.unicode_minus'] = False


def parse_time(data):
    DD = datetime.datetime.strptime(data, "%Y-%m-%d")
    day = DD.day
    month = DD.month
    return day, month


def parse_sh(data):
    if data == "0":
        return "0"
    else:
        return "1"


def parse_year(data):
    return 2015 - data


def main():
    pass


def pre_train():

    pd_train = pd.read_csv('./data/rossmann/train.csv')
    pd_test = pd.read_csv('./data/rossmann/test.csv')
    # 先统计缺失的列
    na_train = pd_train.isnull().sum().sort_values(ascending=False)
    na_test = pd_test.isnull().sum().sort_values(ascending=False)
    print(na_train)
    print(na_test)
    # Open   test 有列缺失
    pd_test['Open'].fillna(1, inplace=True)

    # 处理时间列
    pd_train['Day'], pd_train['Month'] = zip(
        *pd_train['Date'].apply(parse_time))
    pd_test['Day'], pd_test['Month'] = zip(*pd_test['Date'].apply(parse_time))

    # 处理 StateHoliday
    pd_train['SH'] = pd_train['StateHoliday'].apply(parse_sh)
    pd_test['SH'] = pd_test['StateHoliday'].apply(parse_sh)

    # 删除原来的值
    pd_train.drop(['Date', 'StateHoliday'], inplace=True, axis=1)
    pd_test.drop(['Date', 'StateHoliday'], inplace=True, axis=1)

    # 保存处理后的数据
    pd_train.to_csv("./data/rossmann/train2.csv", index=False)
    pd_test.to_csv("./data/rossmann/test2.csv", index=False)
    print("saved")

    # # 缺失值处理
    # imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
    # imp.fit(train)
    # train = imp.transform(train)


def pre_store():
    pd_store = pd.read_csv('./data/rossmann/store.csv')
    # 先统计缺失的列
    na_store = pd_store.isnull().sum().sort_values(ascending=False)
    print(na_store)

    # CompetitionDistance 填充0,然后标准化
    pd_store['CompetitionDistance'].fillna(0, inplace=True)
    scale(pd_store['CompetitionDistance'], copy=False)

    # CompetitionOpenSinceYear 填充0,然后更改
    pd_store['CompetitionOpenSinceYear'].fillna(0, inplace=True)
    pd_store['Cosyc'] = pd_store['CompetitionOpenSinceYear'].apply(parse_year)

    # StoreType/Assortment 变哑变量
    pd_store = pd.get_dummies(pd_store['StoreType'], prefix='StoreType').join(pd_store)
    pd_store = pd.get_dummies(pd_store['Assortment'], prefix='Assortment').join(pd_store)

    # 删除旧列
    pd_store.drop(['StoreType', 'Assortment', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'Promo2SinceWeek',
                   'Promo2SinceYear', 'PromoInterval'], inplace=True, axis=1)

    # print(pd_store.head())
    pd_store.to_csv("./data/rossmann/store2.csv", index=False)


def load_data():

    pd_train = pd.read_csv('./data/rossmann/train2.csv')
    pd_test = pd.read_csv('./data/rossmann/test2.csv')
    pd_store = pd.read_csv('./data/rossmann/store2.csv')

    print(pd_train.columns)
    print(pd_test.columns)
    # # 数据合并
    if not os.path.exists("./data/rossmann/merge.csv"):
        X_train = pd.merge(pd_store, pd_train, on='Store')
        X_train.to_csv('./data/rossmann/merge.csv')

    data = pd.read_csv('./data/rossmann/merge.csv')
    x = data.drop(['Sales', 'Store'], 1)
    y = data['Sales']

    x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=7)

    return x_train, x_test, y_train, y_test


def feature_selection():
    X_train, X_test, y_train, y_test = load_data()

    print("load.......")
    print(X_train.columns)
    params = {
        # 节点的最少特征数 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
        'min_child_weight': 60,
        'eta': 0.02,        # 如同学习率 [默认0.3]
        'colsample_bytree': 0.7,   # 用来控制每棵随机采样的列数的占比(每一列是一个特征)。 典型值：0.5-1
        # 这个值为树的最大深度。 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。 需要使用CV函数来进行调优。 典型值：3-10
        'max_depth': 7,
        'subsample': 0.7,   # 采样训练数据，设置为0.7
        'alpha': 1,         # L1正则化项 可以应用在很高维度的情况下，使得算法的速度更快。
        'gamma': 1,         # Gamma指定了节点分裂所需的最小损失函数下降值。 这个参数的值越大，算法越保守。
        'silent': 1,        # 0 打印正在运行的消息，1表示静默模式。
        'verbose_eval': True,
        'seed': 12
    }

    xgtrain = xgb.DMatrix(X_train, label=y_train)
    bst = xgb.train(params, xgtrain, num_boost_round=10)
    # 特征
    features = [x for x in X_train.columns if x not in ['Sales', 'Store']]
    create_feature_map(features)
    # 获得每个特征的重要性
    importance = bst.get_fscore(fmap='./data/rossmann/xgb.fmap')
    # 重要性排序
    importance = sorted(importance.items(), key=operator.itemgetter(1))

    df = pd.DataFrame(importance, columns=["feature", "fscore"])
    df["fscore"] = df["fscore"] / df['fscore'].sum()
    print(df)
    label = df['feature'].T.values
    xtop = df['fscore'].T.values
    idx = np.arange(len(xtop))
    fig = plt.figure(figsize=(12, 6))
    plt.barh(idx, xtop, alpha=0.8)
    plt.yticks(idx, label,)
    plt.grid(axis='x')              # 显示网格
    plt.xlabel('重要性')
    plt.ylabel('特征')
    plt.title('XGBoost 特征选择图示')
    plt.show()



def transform_data(train):
    for f in train.columns:
        if train[f].dtype == 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(train[f].values))
            train[f] = lbl.transform(list(train[f].values))

    return train


def create_feature_map(features):
    outfile = open('./data/house/xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()


def train():

    X_train, X_test, y_train, y_test = load_data()
    eval_set = [(X_test, y_test)]
    print(X_train.shape)
    print(y_train.shape)

    model = xgb.XGBRegressor(max_depth=6, learning_rate=0.1, n_estimators=500, objective="reg:linear",
                             nthread=4, silent=True, subsample=0.8, colsample_bytree=0.8)
    bst = model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=30, verbose=True)

    preds = bst.predict(X_test)
    rmse = np.sqrt(np.mean((preds - y_test)**2))
    print("rms:", rmse)


if __name__ == '__main__':
    # main()
    # pre_train()
    # pre_store()
    # feature_selection()
    # load_data()

    train()
