# 神经网络
# BP、RPROP、GRPROP 神经网络
# install.packages("neuralnet")
library(neuralnet)
data=read.csv("E:/讲课/机器学习_R语言实现/data_breasttumor1.csv")
summary(data)
# data$良.恶=as.factor(data$良.恶)
# 数据标准化
data1=scale(data[,3:62],center=TRUE,scale=TRUE)
data[,3:62]=data1
data=data[,-1]
# 划分数据集：训练集和测试集
set.seed(123) #设置个随机数，使结果具有可重复性
train_sub=sample(nrow(data),7/10*nrow(data)) #7/10表示70%用于训练
train_data=data[train_sub,]
test_data=data[-train_sub,]
# 构建初级神经网络模型
net1=neuralnet(良.恶~., data=train_data)
plot(net1)
# 训练集和测试集分别进行模型效果测试
net1_predict1=predict(net1,train_data,type="class") #net1——predict1 训练集
net1_predict2=predict(net1,test_data,type="class") #net1——predict2 测试练集
prediction1=c("恶","良")[apply(net1_predict1,1,which.max)] # 将概率赋给各自分类
prediction2=c("恶","良")[apply(net1_predict2,1,which.max)]
pre_table1=table(train_data$良.恶,prediction1)
pre_table2=table(test_data$良.恶,prediction2)
pre_table1
pre_table2
# install.packages("caret")
library(caret)
confusionMatrix(pre_table1)
confusionMatrix(pre_table2)
#
# 调整模型参数，提高性能

# hidden: 隐藏神经元的个数，一般隐藏神经元越多，越精细，计算复杂度越高
#threshold: 停止计算的阈值
#stepmax: 最大迭代次数
#rep: 重复训练次数，与稳定性有关系
#algorithm: 网络类型，可用的有参数有：‘backprop':BP, 
#'rprop-','rprop+': BP with or without weight backtracking，即rprop网络
#'sag','slr':在rporp基础上引入了globall convergent algorithm,即grprop网络
#err.fct: 误差函数，‘sse','ce'分别对应用方差和以及cross-entropy作为误差函数
#act.fct:激活函数，’logistic', 'tanh'
net2=neuralnet(良.恶~., data=train_data,hidden=2)
plot(net2)
# 训练集和测试集分别进行模型效果测试
net2_predict1=predict(net2,train_data,type="class") #net1——predict1 训练集
net2_predict2=predict(net2,test_data,type="class") #net1——predict2 测试练集
prediction3=c("恶","良")[apply(net2_predict1,1,which.max)] # 将概率赋给各自分类
prediction4=c("恶","良")[apply(net2_predict2,1,which.max)]
pre_table3=table(train_data$良.恶,prediction3)
pre_table4=table(test_data$良.恶,prediction4)
pre_table3
pre_table4
confusionMatrix(pre_table3)
confusionMatrix(pre_table4)
# 模型表示ROC曲线和校准曲线
# 训练集ROC曲线
# y 是字符型，需要转换成数值型
y1=as.factor(train_data$良.恶)
y1=as.numeric(y1)
y2=as.factor(test_data$良.恶)
y2=as.numeric(y2)
y1_predict=as.factor(prediction3)
y1_predict=as.numeric(y1_predict)
y2_predict=as.factor(prediction4)
y2_predict=as.numeric(y2_predict)
library(pROC)
roc1=roc(y1,y1_predict)
roc2=roc(y2,y2_predict)
plot(roc1)
title("训练集ROC曲线")
plot(roc2)
title("测试集ROC曲线")