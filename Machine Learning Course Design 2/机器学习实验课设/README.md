If u want run this project, u need to extract the "data.rar" and the "save_model.rar" to the current folder



# 模型训练介绍
- 简单学习
- 分别采用**0.001，0.01，0.05，0.1**四种不同的学习率以及**ASGD,SGD，Adam**等8种不同的优化器进行训练，训练其他超参数**EPOCHS=10,batch_size=64**。

# 模型预测方式
- 卷积神经网络预测

# 执行步骤
- 模型调整：调整model_adjust各个py文件的超参数，再执行。已经调试好了，可直接进行执行。
- 模型训练执行model_train各个文件，训练结果如正确率曲线在accuracy_curve文件夹，损失曲线在loss_curve文件中
- 模型测试执行model_test各个py文件，结果其子文件夹CNN中
