# SVM
install.packages("e1071")
library(e1071) 
data1=read.csv("E:/讲课/机器学习_R语言实现/data_breasttumor1.csv")
View(data1)
data1$良.恶=as.factor(data1$良.恶) #将字符型变量转换为因子型
# data1$良.恶=as.numeric(data1$良.恶) 
#
# 在做SVM之前，把数据随机划分为训练集和测试集
set.seed(123) #设置个随机数，使结果具有可重复性
train_sub=sample(nrow(data1),7/10*nrow(data1)) #7/10表示70%用于训练
train_data=data1[train_sub,]
test_data=data1[-train_sub,]
# 下面建立两种形式的SVM
# 第一种形式
mode1=svm(良.恶~., data=train_data)
#
#第二种形式
y=train_data$良.恶
x=subset(train_data,select=-良.恶)
y2=test_data$良.恶
x2=subset(test_data,select=-良.恶)
mode2=svm(x,y)

# 上边两种形式等效，任何一种习惯的形式都可以
summary(mode1)
summary(mode2)

# 模型建立后，对训练集和测试集数据进行预测，并用校准曲线和ROC曲线评价模型
predict1=predict(mode2,newdata=x,type='log-odds') # 训练集预测
table(predict1,y)
predict2=predict(mode2,newdata=x2,type='prob')
table(predict2,y2)
# 绘制ROC曲线
library(pROC)
y=as.numeric(y)
predict1=as.numeric(predict1)
ROC1 <-roc(y,predict1) # 训练集ROC
plot(ROC1)
y2=as.numeric(y2)
predict2=as.numeric(predict2)
ROC2 <-roc(y2,predict2) #测试集ROC
plot(ROC2)
round(auc(ROC1),3) #查看曲线下面积
round(ci(auc(ROC1)),3) # 查看auc的至信区间
round(auc(ROC2),3)
round(ci(auc(ROC2)),3) 
summary(mode2)
# 绘制 校准曲线
#绘制训练集calibrate curve
library(gbm)
mode3=svm(x,y,probability=TRUE)
predict3=predict(mode3,x,probability=TRUE)
y1=as.numeric(y)-1
prob1=attr(predict3,"probability")[,1]
calibrate.plot(y1,y1)
predict4=predict(mode3,x2,probability=TRUE)
y2=as.numeric(y2)-1
prob2=attr(predict4,"probabilities")[,1]
calibrate.plot(y2,prob2)
# 查看变量重要性
#rminer 包
install.packages('rminer')
library(rminer)
M=fit(良.恶~.,data=train_data,model='svm')
mode.importance=Importance(M,data=train_data)
plot(mode.importance$value)


