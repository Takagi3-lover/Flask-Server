import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import r2_score
from sklearn import metrics
from matplotlib import pyplot
from sklearn import datasets, ensemble


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


def plot_logloss(model):
    results = model.evals_result_
    # print(results)
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    # print(epochs)
    # plot log loss
    fig, ax = plt.subplots()
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
    # print(results['validation_1']['logloss'])
    ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
    ax.legend()
    plt.ylabel('Log Loss')
    plt.title('XGboost Log Loss')
    plt.show()


# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")

df = pd.read_excel(r'chem+E+DA.xlsx', sheet_name='HOMO NEWTOP 23FEATURES')
# df.dropna(inplace=True)#删除行
Y_a = df['HOMO']
X_a = df.iloc[:, 3:]
a = pd.concat([Y_a, X_a], axis=1)
a.dropna(inplace=True)
X_0 = a.iloc[:, 1:]
Y_0 = a['HOMO'].values
X = MaxMinNormalization(X_0)
Y = MaxMinNormalization(Y_0)

indices = np.arange(X.shape[0])  # help for check the index after split

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(X, Y, indices, test_size=0.3,
                                                                                 random_state=5)  # 数据集分割
rf2 = RandomForestRegressor(oob_score=True, n_estimators=100)  # 一般来说n_estimators越大越好，运行结果呈现出的两种结果该值分别是10和1000
model = rf2.fit(X_train, y_train)
print('oob score ' + str(rf2.oob_score_))  # 评估模型泛化能力

y_pred1 = model.predict(X_train)
print('train r2: ', r2(y_train, y_pred1))
Reverse_y_pred1 = y_pred1 * ((np.max(Y_0) - np.min(Y_0))) + np.min(Y_0)
Reverse_y_train = y_train * ((np.max(Y_0) - np.min(Y_0))) + np.min(Y_0)
print('train_reverse_rmse: ', metrics.mean_squared_error(Reverse_y_pred1, Reverse_y_train) ** 0.5)

y_pred2 = model.predict(X_test)
print('test r2: ', r2(y_test, y_pred2))
Reverse_y_pred2 = y_pred2 * ((np.max(Y_0) - np.min(Y_0))) + np.min(Y_0)
Reverse_y_test = y_test * ((np.max(Y_0) - np.min(Y_0))) + np.min(Y_0)
print('test_reverse_rmse: ', metrics.mean_squared_error(Reverse_y_pred2, Reverse_y_test) ** 0.5)

fig = plt.figure(figsize=(12, 9))
plt.scatter(Reverse_y_train, Reverse_y_pred1, c='r', s=10, label='RF-train')
plt.scatter(Reverse_y_test, Reverse_y_pred2, c='b', s=10, label='RF-test')
plt.xlabel('True')
plt.ylabel('Pred')
plt.legend()
plt.xlim(4, 7)
plt.ylim(4, 7)
plt.show()
# #test RMSE
# sum_mean=0
# for i in range(len(y_rf2test)):
#     sum_mean=(sum_mean+(y_rf2test[i] - y_test.values[i]) ** 2).astype(float)
# sum_erro=np.sqrt((sum_mean / len(y_rf2test)))
#
# # calculate RMSE by hand
# print ("test_RMSE :",sum_erro)
#
# y_rf2train=rf2.fit(X_train, y_train).predict(X_train)
#
# #train RMSE
# sum_mean=0
# for i in range(len(y_rf2train)):
#     sum_mean=(sum_mean+(y_rf2train[i] - y_train.values[i]) ** 2).astype(float)
# sum_erro2=np.sqrt((sum_mean / len(y_rf2train)))
#
# # calculate RMSE by hand
# print ("train_RMSE :",sum_erro2)
#


# Reverse_y_train=Reverse_y_train.reset_index(drop=True)
# Reverse_y_train=np.array(Reverse_y_train)
# for i in range(60):
#     if Reverse_y_pred1[i]-Reverse_y_train[i]>=5:
#         print('train误差大于5对应的Y： ',Reverse_y_train[i])
#         print('train误差：',Reverse_y_pred1[i]-Reverse_y_train[i])
#
# Reverse_y_test=Reverse_y_test.reset_index(drop=True)
# Reverse_y_test=np.array(Reverse_y_test)
# for i in range(26):
#     if Reverse_y_pred2[i]-Reverse_y_test[i]>=5:
#         print('test误差大于5对应的Y： ',Reverse_y_test[i])
#         print('test误差：',Reverse_y_pred2[i]-Reverse_y_test[i])
fig2 = plt.figure(figsize=(12, 9))
importances = rf2.feature_importances_
fea_label = X.columns.tolist()
Feature_importances = [round(x, 4) for x in importances]
F2 = pd.Series(Feature_importances, index=fea_label)
F2 = F2.sort_values(ascending=True)  # 排序
f_index = F2.index
f_values = F2.values

# 输出
# print('f_index:', f_index)
# print('f_values:', f_values)

x_index = list(range(0, 23))
x_index = [x / 23 for x in x_index]
plt.rcParams['figure.figsize'] = (10, 10)
plt.barh(x_index, f_values, height=0.028, align="center", color='tan', tick_label=f_index)
plt.xlabel('importances')
plt.ylabel('features')
plt.show()

# plot_logloss(model)


# 预测新材料代码部分
import shap

# TODO: 当前数据集中的特征名称数量、顺序与训练模型时的特征名称数量、顺序不一致。

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value, shap_values, X_train)
shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=500, show=True)
shap.summary_plot(shap_values, X_train, max_display=500, show=True)

shap_values2 = explainer(X_train)
shap.plots.bar(shap_values2, max_display=30)

from scipy.stats import pearsonr

print('train r: ', pearsonr(y_train, y_pred1))
print('test r: ', pearsonr(y_test, y_pred2))

GTL = pd.read_excel(r'TestFile.xlsx', sheet_name='HOMO NEWTOP 23 FEATURES')
gtl_x = GTL.iloc[:, 1:]
for column in np.arange(X_0.shape[1]):
    gtl_x.iloc[:, column] = (gtl_x.iloc[:, column] - np.min(X_0.iloc[:, column])) / (
            np.max(X_0.iloc[:, column]) - np.min(X_0.iloc[:, column]))
gtl_y_pred = model.predict(gtl_x)
gtl_y_pred_Original = gtl_y_pred * (np.max(Y_0) - np.min(Y_0)) + np.min(Y_0)
print('预测结果: ' + str(gtl_y_pred_Original))

#
# shap_interaction_values = explainer.shap_interaction_values(X_train)
# shap.summary_plot(shap_interaction_values, X_train,show=True)
