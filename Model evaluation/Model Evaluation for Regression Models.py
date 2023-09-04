import csv
import numpy as np
#导入sklearn计算和计算结果进行对比
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 读取CSV文件
with open('experiment_01_dataset_01.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    data = [row for row in csvreader]

# 将数据转换为NumPy数组
data = np.array(data).astype(float)

#计算误差度量的函数
def CalculateError(data):
    MSE=[]
    RMSE=[]
    MAE=[]
    #第2列到第4列分别对应三种方法的结果
    for i in range(2,5):
        #计算MSE
        MSE.append(np.sum( (data[:,i]-data[:,1]) **2,axis=0) /data.shape[0])
        #计算RMSE
        RMSE.append((np.sum((data[:,i]-data[:,1]) ** 2, axis=0)/ data.shape[0])**(0.5) )
        #计算MAE
        MAE.append(np.sum(abs(data[:,i]-data[:,1]), axis=0) / data.shape[0])
    return MSE,RMSE,MAE


MSE,RMSE,MAE=CalculateError(data)
print(MSE,RMSE,MAE,sep='\n')
print("my calculate:")
print("MSE of prediction 1:", MSE[0])
print("MSE of prediction 2:", MSE[1])
print("MSE of prediction 3:", MSE[2])

print("RMSE of prediction 1:", RMSE[0])
print("RMSE of prediction 2:", RMSE[1])
print("RMSE of prediction 3:", RMSE[2])

print("MAE of prediction 1:", MAE[0])
print("MAE of prediction 2:", MAE[1])
print("MAE of prediction 3:", MAE[2])

print("----------------------------------")

print("sklearn calculate:")
# 定义真实标签和多个算法的预测结果
y_true = data[:,1]
prediction_1 = data[:,2]
prediction_2 = data[:,3]
prediction_3 = data[:,4]

# 计算MSE
mse_1 = mean_squared_error(y_true, prediction_1)
mse_2 = mean_squared_error(y_true, prediction_2)
mse_3 = mean_squared_error(y_true, prediction_3)

print("MSE of prediction 1:", mse_1)
print("MSE of prediction 2:", mse_2)
print("MSE of prediction 3:", mse_3)

# 计算RMSE
rmse_1 = np.sqrt(mean_squared_error(y_true, prediction_1))
rmse_2 = np.sqrt(mean_squared_error(y_true, prediction_2))
rmse_3 = np.sqrt(mean_squared_error(y_true, prediction_3))

print("RMSE of prediction 1:", rmse_1)
print("RMSE of prediction 2:", rmse_2)
print("RMSE of prediction 3:", rmse_3)

# 计算MAE
mae_1 = mean_absolute_error(y_true, prediction_1)
mae_2 = mean_absolute_error(y_true, prediction_2)
mae_3 = mean_absolute_error(y_true, prediction_3)

print("MAE of prediction 1:", mae_1)
print("MAE of prediction 2:", mae_2)
print("MAE of prediction 3:", mae_3)
