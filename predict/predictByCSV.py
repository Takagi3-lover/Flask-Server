import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# --------自定义函数--------
def MaxMinNormalization(x):  # 数据归一化
    """[0,1] normaliaztion"""
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x


def r2(y_true, y_pred):
    a = np.square(y_true - y_pred)
    b = np.sum(a)
    c = np.mean(y_true)
    d = np.square(y_true - c)
    e = np.sum(d)
    f = 1 - b / e
    return f


# --------模型训练--------
# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")


def getHOMO(PredictFilePath):
    df = pd.read_excel(r'predict/chem+E+DA.xlsx', sheet_name='HOMO NEWTOP 23FEATURES')
    Y_a = df['HOMO']
    X_a = df.iloc[:, 3:]
    a = pd.concat([Y_a, X_a], axis=1)
    a.dropna(inplace=True)
    X_0 = a.iloc[:, 1:]
    Y_0 = a['HOMO'].values

    model = joblib.load("predict/models/HOMO_Model.pkl")

    # 预测新材料代码部分
    GTL = pd.read_csv(PredictFilePath)
    gtl_x = GTL.iloc[:, 1:]
    for column in np.arange(X_0.shape[1]):
        gtl_x.iloc[:, column] = (gtl_x.iloc[:, column] - np.min(X_0.iloc[:, column])) / (
                np.max(X_0.iloc[:, column]) - np.min(X_0.iloc[:, column]))
    gtl_y_pred = model.predict(gtl_x)
    gtl_y_pred_Original = gtl_y_pred * (np.max(Y_0) - np.min(Y_0)) + np.min(Y_0)
    return gtl_y_pred_Original[0]


def getLUMO(PredictFilePath):
    df = pd.read_excel(r'predict/chem+E+DA.xlsx', sheet_name='LUMO NEWTOP 20 FEATURES')
    Y_a = df['LUMO']
    X_a = df.iloc[:, 3:]
    a = pd.concat([Y_a, X_a], axis=1)
    a.dropna(inplace=True)
    X_0 = a.iloc[:, 1:]
    Y_0 = a['LUMO'].values

    model = joblib.load("predict/models/LUMO_Model.pkl")

    # 预测新材料代码部分
    GTL = pd.read_csv(PredictFilePath)
    gtl_x = GTL.iloc[:, 1:]
    for column in np.arange(X_0.shape[1]):
        gtl_x.iloc[:, column] = (gtl_x.iloc[:, column] - np.min(X_0.iloc[:, column])) / (
                np.max(X_0.iloc[:, column]) - np.min(X_0.iloc[:, column]))
    gtl_y_pred = model.predict(gtl_x)
    gtl_y_pred_Original = gtl_y_pred * (np.max(Y_0) - np.min(Y_0)) + np.min(Y_0)

    return gtl_y_pred_Original[0]


def trainAndPredict(PredictFilePath):
    df = pd.read_excel(r'predict/chem+E+DA.xlsx', sheet_name='HOMO NEWTOP 23FEATURES')
    Y_a = df['HOMO']
    X_a = df.iloc[:, 3:]
    a = pd.concat([Y_a, X_a], axis=1)
    a.dropna(inplace=True)
    X_0 = a.iloc[:, 1:]
    Y_0 = a['HOMO'].values
    X = MaxMinNormalization(X_0)
    Y = MaxMinNormalization(Y_0)

    indices = np.arange(X.shape[0])

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=0.3,
                                                                                     random_state=5)  # 数据集分割

    # 使用网格搜索法调整参数
    # from sklearn.model_selection import GridSearchCV
    # param_grid = {'n_estimators': range(10, 150, 10), 'max_features': range(1, 23, 1)}
    # rf = RandomForestRegressor()
    # grid_search = GridSearchCV(rf, param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_params_)

    rf2 = RandomForestRegressor(oob_score=True, n_estimators=100)  # 一般来说n_estimators越大越好，运行结果呈现出的两种结果该值分别是10和1000
    model = rf2.fit(X_train, y_train)

    # plt.figure(figsize=(12, 9))
    # importances = rf2.feature_importances_
    # fea_label = X.columns.tolist()
    # Feature_importances = [round(x, 4) for x in importances]
    # F2 = pd.Series(Feature_importances, index=fea_label)
    # F2 = F2.sort_values(ascending=True)  # 排序
    # f_index = F2.index
    # f_values = F2.values

    # x_index = list(range(0, 23))
    # x_index = [x / 23 for x in x_index]
    # plt.rcParams['figure.figsize'] = (10, 10)
    # plt.barh(x_index, f_values, height=0.028, align="center", color='tan', tick_label=f_index)
    # plt.xlabel('importances')
    # plt.ylabel('features')
    # plt.savefig('img/feature_importances.png')
    # plt.show()

    # explainer = shap.TreeExplainer(model)
    # shap_values = explainer.shap_values(X_train)
    # shap.force_plot(explainer.expected_value, shap_values, X_train)
    # shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=500, show=True)
    # 如果要保存图片，需要将show设置为False
    # plt.savefig('img/shap_values1.png')
    # shap.summary_plot(shap_values, X_train, max_display=500, show=True)
    # plt.savefig('img/shap_values2.png')

    # shap_values2 = explainer(X_train)
    # shap.plots.bar(shap_values2, max_display=30, show=True)
    # plt.savefig('img/shap_values3.png')

    # 预测新材料代码部分
    GTL = pd.read_csv(PredictFilePath)
    gtl_x = GTL.iloc[:, 1:]
    for column in np.arange(X_0.shape[1]):
        gtl_x.iloc[:, column] = (gtl_x.iloc[:, column] - np.min(X_0.iloc[:, column])) / (
                np.max(X_0.iloc[:, column]) - np.min(X_0.iloc[:, column]))
    gtl_y_pred = model.predict(gtl_x)
    gtl_y_pred_Original = gtl_y_pred * (np.max(Y_0) - np.min(Y_0)) + np.min(Y_0)

    # 逐行输出预测结果
    return gtl_y_pred_Original[0]
    # for i in range(len(gtl_y_pred_Original)):
    #     print('分子名称: ' + GTL.iloc[i, 0])
    #     print('HOMO预测值: ' + str(gtl_y_pred_Original[i]))
    #     print('------------------------')
