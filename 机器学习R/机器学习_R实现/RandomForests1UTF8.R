# 随机森林
#install.packages("randomForest")
#install.packages("pROC")
#install.packages("dplyr")
#install.packages("ggplot2")

library(randomForest)
library(pROC)
library(ggplot2)
data1=read.csv("C:\\Users\\Administrator\\Desktop\\机器学习会议\\data_breasttumor1.csv")

data1$良.恶 <-as.factor(data1$良.恶)

data1=data1[,-1]
# dd=datadist(data1) #  对于给定的数据框架，确定影响和绘图范围的变量摘要、要调整的值和Predict的总体范围
# options(datadist="dd") #允许用户设置和检查各种影响R计算和显示结果的方式的全局选项 
set.seed(123) #设置个随机数，使结果具有可重复性
#将数据随机分为训练集和测试集
train_sub=sample(nrow(data1),7/10*nrow(data1))
train_data=data1[train_sub,]
test_data=data1[-train_sub,]
mode1 <-randomForest(良.恶~ .,data=train_data)
mode1 
plot(mode1)
mode3 <-randomForest(良.恶~ .,data=train_data,ntree=50000,mtry=8) # ntree 数的棵树(obs的10倍)，mtry 每棵树(obs开根号)使用的特征个数
mode3
varImpPlot(mode3)  #变量重要性
x=data.frame(importance(mode3)) 
library(dplyr) 
x=arrange(x,desc(MeanDecreaseGini)) #排序
imp=data.frame(MeanDecreaseGini=x[1:10,],row.names=rownames(x)[1:10])
p1=ggplot(imp,aes(x=MeanDecreaseGini,y=reorder(rownames(imp),MeanDecreaseGini)))
  p1+geom_bar(position = position_dodge(),
           width = 0.5,
           stat = "identity",
           fill="steelblue")

##绘制RF的ROC曲线
  #对训练集进行预测和测试集进行预测
predict1=predict(mode3,train_data)
predict2 <-predict(mode3,test_data)
help(predict)
#输出混淆后的矩阵
table(train_data$良.恶,predict1,dnn=c("真实值","预测值"))
table(test_data$良.恶,predict2,dnn=c("真实值","预测值"))
roc1 <- roc(train_data$良.恶,as.numeric(predict1))
roc2=roc(test_data$良.恶, as.numeric(predict2))
plot(roc1, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),grid.col=c("green", "red"), max.auc.polygon=TRUE,auc.polygon.col="skyblue", print.thres=TRUE,main='随机森林模型训练集ROC曲线,mtry=8,ntree=5000')
plot(roc2,print.auc=TRUE,main='随机森林模型测试集ROC曲线,mtry=8,ntree=5000')

## RF实现生多分类预测
library(randomForest)
data2=read.csv("C:\\Users\\Administrator\\Desktop\\机器学习会议\\multiclasstest.csv")
data2=data2[,-1]
data2=na.omit(data2)
View(data2)
data2$Classes<-factor(data2$Classes, levels=c(1,2,3),labels=c("HC","MCI","AD"))
set.seed(71)
train_sub=sample(nrow(data2),7/10*nrow(data2))
train_data=data2[train_sub,]
test_data=data2[-train_sub,]
classes.rf<- randomForest(Classes~ ., data=train_data,importance=TRUE,proximity=TRUE)
print(classes.rf)
classes.rf$predicted
# 多分类的ROC曲线
pre2=predict(classes.rf,test_data,type="prob")
roc2 <-multiclass.roc(test_data$Classes, pre2,)
roc2
plot(roc2$rocs$`HC/MCI`[[1]],col='green',lwd=5)
plot(roc2$rocs$`HC/AD`[[1]],add=TRUE,col='blue',lty=3,lwd=3)
plot(roc2$rocs$`MCI/AD`[[1]],add=TRUE, col='red',lty=2,lwd=1)

round(auc(roc2$rocs$`HC/MCI`[[1]]),2)
round(auc(roc2$rocs$`HC/AD`[[1]]),2)
round(auc(roc2$rocs$`MCI/AD`[[1]]),2)




## RF实现连续型变量预测：生存时间预测

data=read.csv("C:\\Users\\Administrator\\Desktop\\机器学习会议\\lung.csv")
summary(data)
# 数据预处理，异常值\缺失值处理
data1=data[,-1] # 将ID从数据总删除
summary(data1$time)

# 对time中的异常值进行检测
boxplot(data1$time,boxwex=0.7)
title("生存时间异常值检测箱线图")
stats=boxplot.stats(data1$time) # 计算箱线图检测结果
id=which(data1$time %in% stats$out)
data2=data1
data2[id,1]=NA
#使用missForest进行缺失值填补
#install.packages("missForest")
library(missForest)
data3=missForest(data2,ntree=1000)
data4=data3$ximp
set.seed(131)
train_sub=sample(nrow(data4),9/10*nrow(data4))
train_data=data4[train_sub,]
test_data=data4[-train_sub,]
mode5<- randomForest(time ~ ., data=train_data, ntree=50000, mtry=3,
                     importance=TRUE, na.action=na.omit)

predict <-predict(mode5,newdata=test_data)
error=predict-test_data[,1]

plot(c(1:23),test_data[,1],col='blue',pch=1,lwd=4,type='b')
lines(c(1:23),predict,col='red',pch=1,lwd=4,type='b')
legend("topright",cex=0.5,inset=0.05,title="RFprediction",c("real","predicted"),
       lty=c(1,2),pch=c(1,2),col=c("blue","red"),lwd=2,text.font=4)
x=data4$time
#install.packages("vioplot")
library(vioplot)
vioplot(x,ylim=c(0,1000))
# 
# 带有截尾的生存资料特性，需要用到比例风险模型
# 下面利用randomForestSRC做生存分析
#install.packages("rms")
library(rms)
dd=datadist(data4)
options(datadist="dd")

f1 <- coxph(Surv(time,status) ~age+sex+ph.ecog+ph.karno+pat.karno+meal.cal+wt.loss, data=train_data)
summary(f1)
# 构建最终cox模型
f2=psm(Surv(time,status) ~sex+ph.ecog+pat.karno+wt.loss+meal.cal, data=train_data,dist='lognormal')
#方程f1是用的rms包中的参数回归模型，假设生存函数是lognormal分布，然后f2是cox等比例风险模型，不注重生存函数的分布，只关注暴露因素的效应，流行病学中主要采用f2拟合方程。
med <- Quantile(f2) # 计算中位生存时间
surv <- Survival(f2) # 构建生存概率函数
# 绘制诺莫图
nom <- nomogram(f2, fun=function(x) med(lp=x),funlabel="Median Survival Time")
plot(nom)
# 绘制带生存概率的诺莫图
nom2 <- nomogram(f2, fun=list(function(x) surv(365, x),
                              function(x) surv(1825, x),
                              function(x) med(lp=x)),
                 funlabel=c("1-year Survival Probability", "5-year Survival Probability","Median Survival Time"))
plot(nom2)
# 预测
a=exp(predict(f2,type="lp"))
abeta=f2$coefficients
x=matrix(data=NA,nrow=205,ncol=6)
prediction1=matrix(data=NA,nrow=205,ncol=1)
x[,1]=1
x[,2]=train_data$sex
x[,3]=train_data$ph.ecog
x[,4]=train_data$pat.karno
x[,5]=train_data$wt.loss
x[,6]=train_data$meal.cal
predict1=matrix(data=NA,205,6)
prediction2=matrix(data=NA,205,1)
for (i in 1:205){
  predict1[i,]=t(abeta)*x[i,]
  prediction2[i]=exp(sum(predict1[i,]))}
plot(c(1:205),train_data[,1],col='blue',pch=1,lwd=4,type='b')
lines(c(1:205),exp(prediction2),col='red',pch=1,lwd=4,type='b')
# 绘制KM曲线
# 用KM曲线反映模型精度：分别做出训练集KM曲线，测试集KM曲线，与预测结果的KM曲线进行比较
# 训练集KM曲线
library(ggplot2)
library(survival)
install.packages("survminer") #首次需要安装survminer, ggpubr, Rcpp包
install.packages("ggpubr")
library(ggpubr)
library(survminer)

# 训练集K-M
fit1=survfit(Surv(time,status) ~1,data=train_data)
ggsurvplot(fit1, color = "blue",surv.median.line="hv",
                      ggtheme = theme_minimal())
# 预测的K-M曲线
new1=train_data
new1$time=prediction2
fit2=survfit(Surv(time,status) ~1,data=new1)
ggsurvplot(fit2, color = "green",surv.median.line="hv",
           ggtheme = theme_minimal())
# 危险因素-性别的KM曲线
fit3=survfit(Surv(time,status)~sex,data=train_data)
ggsurvplot(fit3, data=train_data,
           surv.median.line="hv",
           conf.int=TRUE,
           pval=TRUE)
# pat.karno:注意连续型变量
res.cut <- surv_cutpoint(train_data, time="time",event="status",variables="pat.karno")
res.cat <-surv_categorize(res.cut)
fit4 <-survfit(Surv(time,status) ~pat.karno,data=res.cat)
ggsurvplot(fit4, data=res.cat,surv.median.line="hv",
           conf.int=TRUE,
           pval=TRUE,
           risk.table=TRUE)
