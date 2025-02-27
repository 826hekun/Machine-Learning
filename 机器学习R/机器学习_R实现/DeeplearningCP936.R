## Deeplearning：CNN
install.packages("torch")
library(torch)
#训练集
dfminist2<-read.csv("mnist_train.csv",header=F)
dfminist2[,1]<-dfminist2[,1]+1
xminist2<-as.matrix(dfminist2[,2:785])
xminist2<-xminist2/255
xarray2<-array(0,dim=c(60000,1,28,28))
for(i in 1:60000){
  xarray2[i,1,,]<-matrix(xminist2[i,],nrow=28,byrow=T)}

#测试集
dfminist<-read.csv("mnist_test.csv",header=F)
dfminist[,1]<-dfminist[,1]+1
xminist<-as.matrix(dfminist[,2:785])
xminist<-xminist/255
xarray<-array(0,dim=c(10000,1,28,28))
for(i in 1:10000){
  xarray[i,1,,]<-matrix(xminist[i,],nrow=28,byrow=T)}

minist_dataset<-dataset(
  
  initialize=function(xarray,label){
    
    self$x<-torch_tensor(xarray,dtype=torch_float())
    
    self$y<-torch_tensor(label,dtype=torch_long())},
  
  .getitem=function(index){
    
    list(x=self$x[index,,,],y=self$y[index])},
  
  .length=function(){length(self$y)})

# 3.生成训练集和测试集的dataset数据索引
ministdsta<-minist_dataset(xarray2,label=dfminist2[,1])
ministdste<-minist_dataset(xarray,label=dfminist[,1])
ministdlta<-dataloader(ministdsta,batch_size=32,shuffle=T)
ministdlte<-dataloader(ministdste,batch_size=32,shuffle=T)

5.创建神经网络结构

net <- nn_module(
  
  initialize = function() {
    
    self$conv1 <- nn_conv2d(1,32, kernel_size=3)#卷积层
    
    self$conv2 <- nn_conv2d(32,64,kernel_size=3)
    
    self$conv3 <- nn_conv2d(64,128, kernel_size=3)
    
    self$conv4 <- nn_conv2d(128,256,kernel_size=3)
    
    self$fc1 <- nn_linear(4*4*256, 128)#线性层
    
    self$fc2 <- nn_linear(128,10)
    
    self$dropout1<-nn_dropout(0.25)#随机丢失，用来防止过拟合
    
    self$dropout2<-nn_dropout(0.25)
    
    self$dropout3<-nn_dropout(0.25)
    
  },
  
  forward = function(x) {
    
    x %>%
      self$conv1() %>%
      nnf_relu() %>%
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      self$dropout1()%>%
      self$conv3() %>%
      nnf_relu() %>%
      self$conv4() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      self$dropout2()%>%
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_selu() %>%
      self$dropout3()%>%
      self$fc2()
  })
model<-net()
# 6.模型训练

# torch神经网络的训练主要有以下4步：

# 1. 将梯度设置为零。

# 2. 定义和计算成本和优化器

# 3. 在网络上传播错误。

# 4. 应用梯度优化。

optimizer <- optim_adam(model$parameters)#优化器

n_epochs <-10#迭代步数

model$train()# 设置成训练模型

for(epoch in 1:n_epochs) {
  
  train_losses <- c()
  coro::loop(for(b in ministdlta) {
    optimizer$zero_grad()
    output <- model(b[[1]])
    loss <- nnf_cross_entropy(output, b[[2]])
    loss$backward()
    optimizer$step()
    train_losses <- c(train_losses, loss$item())
  })
  cat(sprintf("Epoch %d: train loss: %3f\n",
              
              epoch, mean(train_losses)))
}

# 7.模型评价：

# 训练集的表现：

# Evaluate
model$eval()#设置预测模型
pre<-c()
true<-c()
coro::loop(for(b in ministdlta) {#测试集的情况
  output <- model(b[[1]])
  pred <- torch_max(output, dim = 2)[[2]]
  pre<-c(pre,as.numeric(pred))
  true<-c(true,as.numeric(b[[2]]))
  
})
ma<-table(true,pre)
ma
sum(diag(ma))/sum(ma)
测试集表现：

# Evaluate

model$eval()#设置预测模型
pre<-c()
true<-c()
coro::loop(for(b in ministdlte) {#训练集的情况
  output <- model(b[[1]])
  pred <- torch_max(output, dim = 2)[[2]]
  pre<-c(pre,as.numeric(pred))
  true<-c(true,as.numeric(b[[2]]))
})
ma<-table(true,pre)
ma
sum(diag(ma))/sum(ma)
