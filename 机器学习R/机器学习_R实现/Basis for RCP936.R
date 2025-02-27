# Basis for R a1=3
getwd() #查看当前工作目录
list.files() #列出当前目录下的文件及文件夹
setwd("E:/讲课/机器学习_R语言实现")
getwd()
list.files()
# R语言的变量：字母，数字，以及点号.或下划线组成，但得是字母或点开头
# R中的赋值方式：
var_2.=1
var_name=1
2vare_name=1 # 只能以字母或点开头
.var_name1=1
var.name=1
_var_name=1  # 只能以字母或点开头
a1=3
a1
a2=c(3)
a3 <-c(3)
a4 <-3
a1
a2
a3
a4
b=c(1,2,3)
b
rm(b) #删除变量：remove
b
# 数学运算：加、减、乘、除
c1=1
c2=2
c3=3
c4=c1+c2*c3
c5=c1/c2
c6=c1%/%c3

# R中的逻辑：TRUE, FALSE
1>2 #判断1是否大于2，如果大于则返回TRUE,否则返回FALSE
1<2
1==2
#与，或，非
# 与
a=c(1,2,TRUE,2+3i)
b=c(0,2,FALSE,0)
a&b
# 或
a|b
# 非
!b&!a
# R中的基本数据类型：数字，逻辑，文本
d1=1
str(d1)
d2=TRUE
str(d2)
d3='abcd'
str(d3)
# R中的变量类型：向量，列表，矩阵，数组，因子，数据框
# 向量：由数字构成的一行或一列数字
e1=c(1,2,3) 
e2=max(e1) # 还有一些向量操作，感兴趣自己学习
#字符串：由字符构成的一串字符
e3='abcd123' # 还有一些字符串操作，感兴趣自己学习
# 矩阵：M*N的数字构成
f1=matrix(c(1,2,3,4),2,2)
# 数据框：二维列表，数据每一列都有唯一列名，同一列的数据类型要求一致
# 不同列的数据类型可以不一样
g1=data.frame(ID=c(1,2,3),性别=c('F','M','F'),年龄=c(37,40,35))
# 因子（factor)
h1=c('男','女','男','女')
h2=factor(h1,levels=c('男','女'),labels=c('1','2'))
#
# 数据类型转换
# factor to vector
h3=as.numeric(h2)
# vector to factor
h4=factor(h3,levels=c('1','2'),labels=c('F','M'))
h5=as.factor(h3)
#
# frame to matrix
str(g1)
g2=as.matrix(g1)
str(g2)
g3=g1$性别
g4=as.factor(g3)
g5=as.numeric(g4)
g1$性别=g5
g1
g6=as.matrix(g1)
# 循环语句 repeat, while, for
# repeat
v <- c("Zhang Shan","Li Si")
count <- 1

repeat {
  print(v)
  count <- count+1
  
  if(count > 10) {
    break
  }
}

#while 循环
cnt <- 2

while (cnt < 7) {
  print(v)
  cnt = cnt + 1
}

# for 循环
for (i in c(1:5)) {
  print(v)
}

# 工具包（library)
# 工具包的概念
#查看已安装工具包
library()
# 
#下载工具包
# install.packages("工具包名称")
# 加载工具包
# library(工具包名称) 


#导入CSV文件
data=read.csv("E:/讲课/机器学习_R语言实现/data_breasttumor1.csv")
View(data)
names(data)
str(data)
#写出CSV文件
list.files()
write.csv(data,'E:/讲课/机器学习_R语言实现/newdata1.csv')

write.table(data,"newdata.csv",row.name=TRUE,col.name=TRUE,sep=",")
getwd()
list.files()
