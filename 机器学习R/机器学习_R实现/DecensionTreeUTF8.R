##Decision Tree
install.packages("rpart")
install.packages("rpart.plot")
library(rpart)      # 决策树算法实现
library(rpart.plot) # 图视化决策树

data1=read.csv("E:/讲课/机器学习_R语言实现/data_breasttumor1.csv")
View(data1)
data1$良.恶 <-as.factor(data1$良.恶)
View(data1)
data1=data1[,-1]
# dd=datadist(data1) #  对于给定的数据框架，确定影响和绘图范围的变量摘要、要调整的值和Predict的总体范围
# options(datadist="dd") #允许用户设置和检查各种影响R计算和显示结果的方式的全局选项 
set.seed(123) #设置个随机数，使结果具有可重复性
#将数据随机分为训练集和测试集
train_sub=sample(nrow(data1),9/10*nrow(data1))
train_data=data1[train_sub,]
test_data=data1[-train_sub,]
#
# 1、构建一个初步的决策树
tree1 <- rpart(良.恶~ .,data=train_data, control=rpart.control(cp=.0001))
# 2、寻找最优的cp值，对tree进行修剪
best <- tree1$cptable[which.min(tree1$cptable[,"xerror"]),"CP"]
pruned_tree <- prune(tree1, cp=best)
prp(pruned_tree,
    faclen=0,   # 使用完整标签名称
    extra=1,    # 显示每个终端节点数量
    roundint=F, # 输出数值不近似为整数
    digits=5)   # 输出显示小数位数5位 
# 使用
install.packages("rattle")
library(rattle)
library(rpart.plot)
library(RColorBrewer)
fancyRpartPlot(pruned_tree)
# 预测
pred<-predict(pruned_tree,test_data,type="class")
table(pred,test_data$良.恶)
# ROC曲线 自己画
