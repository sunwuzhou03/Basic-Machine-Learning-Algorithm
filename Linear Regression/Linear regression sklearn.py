import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_diabetes
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

#导入sklearn中的数据集
diabetes=load_diabetes()
data,target=diabetes.data,diabetes.target
x,y=shuffle(data,target,random_state=13)
offset=int(x.shape[0]*0.8)
x_train,y_train=x[:offset],y[:offset]
x_test,y_test=x[offset:],y[offset:]
y_train=y_train.reshape((-1,1))
y_test=y_test.reshape((-1,1))

#自定义性能衡量指标计算函数
def CalculateError(y_true,y_predicion):
    MSE=np.sum( (y_predicion-y_true) **2,axis=0) /y_predicion.shape[0]
    RMSE=(np.sum((y_predicion-y_true) ** 2, axis=0)/ y_predicion.shape[0])**(0.5)
    MAE=np.sum(abs(y_predicion-y_true), axis=0) / y_predicion.shape[0]
    return MSE.item(),RMSE.item(),MAE.item()

#定义模型实例
regr = linear_model.LinearRegression()
#模型拟合训练数据
regr.fit(x_train,y_train)
#模型预测值
y_pred=regr.predict(x_test)
#真实值
y_true=y_test
#计算MSE,RMSE,MAE
MSE,RMSE,MAE=CalculateError(y_true,y_pred)
#输出结果
print("MSE of prediction:", MSE)
print("RMSE of prediction:", RMSE)
print("MAE of prediction:", MAE)
