import csv
import numpy as np
from numpy import mat
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

# 读取CSV文件
with open('experiment_01_dataset_02.csv') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    data = [row for row in csvreader]

# 将数据转换为NumPy数组
data = np.array(data).astype(float)


def CaculatePermance(data):
    accuracy = []#精度
    error_rate=[]#错误率
    precision=[]#查准率
    recall=[]#查全率
    F1=[]#F1
    confusionmatrix=[]#混淆矩阵
    tp=[]#真正例
    fn=[]#假反例
    fp=[]#假正例
    tn=[]#真反例
    for i in range(2,5):
        #利用逻辑运算进行错误率和准确率的计算
        accuracy.append(np.sum(data[:, i] == data[:, 1]) / data.shape[0])
        error_rate.append(np.sum(data[:, i] != data[:, 1]) / data.shape[0])
        #利用位运算来计算混淆矩阵的参数
        tp.append(np.sum((data[:, 1]==1) & (data[:, i]==1)))
        fn.append(np.sum((data[:, 1]==1) & (data[:, i]==0)))
        fp.append(np.sum((data[:, 1]==0) & (data[:, i]==1)))
        tn.append(np.sum((data[:, 1]==0) & (data[:, i]==0)))
        #根据混淆矩阵的参数计算查准率，查全率，F1
        precision.append((tp[i-2])/(tp[i-2]+fp[i-2]))
        recall.append((tp[i-2])/(tp[i-2]+fn[i-2]))
        F1.append((2*precision[i-2]*recall[i-2])/(precision[i-2]+recall[i-2]))
        confusionmatrix.append([[tn[i-2],fp[i-2]],
                                 [fn[i-2],tp[i-2]]])
    return error_rate,accuracy,precision,recall,F1,confusionmatrix

error_rate,accuracy,precision,recall,F1,confusionmatrix=CaculatePermance(data)

print(error_rate,accuracy,precision,recall,F1,confusionmatrix,sep='\n')

#输出结果
print("[[tn fp]\n[fn tp]]")#混淆矩阵格式
print("Confusion matrix (prediction 1):\n", mat(confusionmatrix[0]))
print("Confusion matrix (prediction 2):\n", mat(confusionmatrix[1]))
print("Confusion matrix (prediction 3):\n", mat(confusionmatrix[2]))

print("Accuracy rate (prediction 1):", accuracy[0])
print("Accuracy rate (prediction 2):", accuracy[1])
print("Accuracy rate (prediction 3):", accuracy[2])

print("Error rate (prediction 1):", error_rate[0])
print("Error rate (prediction 2):", error_rate[1])
print("Error rate (prediction 3):", error_rate[2])

print("Precision (prediction 1):", precision[0])
print("Precision (prediction 2):", precision[1])
print("Precision (prediction 3):", precision[2])

print("Recall (prediction 1):", recall[0])
print("Recall (prediction 2):", recall[1])
print("Recall (prediction 3):", recall[2])

print("F1 score (prediction 1):", F1[0])
print("F1 score (prediction 2):", F1[1])
print("F1 score (prediction 3):", F1[2])


print("-------------------------------------")
print("sklearn caculate")

y_true = data[:,1]
predictions = data[:,1:5]
# 将真实标签转化为二元格式
y_true = np.array(y_true)

# 将算法的预测结果转化为二元格式
# prediction 1的预测结果
pred1 = data[:,2]
# prediction 2的预测结果
pred2 = data[:,3]
# prediction 3的预测结果
pred3 = data[:,4]
# 计算混淆矩阵
cm_pred1 = confusion_matrix(y_true, pred1)
cm_pred2 = confusion_matrix(y_true, pred2)
cm_pred3 = confusion_matrix(y_true, pred3)
#print(cm_pred1,cm_pred2,cm_pred3,sep='\n')

# 计算错误率
acc_pred1 =  accuracy_score(y_true, pred1)
acc_pred2 =  accuracy_score(y_true, pred2)
acc_pred3 =  accuracy_score(y_true, pred3)

# 计算错误率
err_pred1 = 1 - accuracy_score(y_true, pred1)
err_pred2 = 1 - accuracy_score(y_true, pred2)
err_pred3 = 1 - accuracy_score(y_true, pred3)

# 计算精度
prec_pred1 = precision_score(y_true, pred1)
prec_pred2 = precision_score(y_true, pred2)
prec_pred3 = precision_score(y_true, pred3)

# 计算查全率
rec_pred1 = recall_score(y_true, pred1)
rec_pred2= recall_score(y_true, pred2)
rec_pred3 = recall_score(y_true, pred3)

#计算 F1 分数
f1_pred1 = f1_score(y_true, pred1)
f1_pred2 = f1_score(y_true, pred2)
f1_pred3 = f1_score(y_true, pred3)

#输出结果
print("Confusion matrix (prediction 1):\n", cm_pred1)
print("Confusion matrix (prediction 2):\n", cm_pred2)
print("Confusion matrix (prediction 3):\n", cm_pred3)


print("Accuracy rate (prediction 1):", acc_pred1)
print("Accuracy rate (prediction 2):", acc_pred2)
print("Accuracy rate (prediction 3):", acc_pred3)


print("Error rate (prediction 1):", err_pred1)
print("Error rate (prediction 2):", err_pred2)
print("Error rate (prediction 3):", err_pred3)

print("Precision (prediction 1):", prec_pred1)
print("Precision (prediction 2):", prec_pred2)
print("Precision (prediction 3):", prec_pred3)

print("Recall (prediction 1):", rec_pred1)
print("Recall (prediction 2):", rec_pred2)
print("Recall (prediction 3):", rec_pred3)

print("F1 score (prediction 1):", f1_pred1)
print("F1 score (prediction 2):", f1_pred2)
print("F1 score (prediction 3):", f1_pred3)






