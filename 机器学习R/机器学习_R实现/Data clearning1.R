# data clearning
# 寮傚父鍊兼娴嬩笌澶勭悊
# 寮傚父鍊兼娴?
data=read.csv('E:/璁茶/鏈哄櫒瀛︿範_R璇█瀹炵幇/lung.csv')
data1=data[,-1]
summary(data1$time)

# 瀵箃ime涓殑寮傚父鍊艰繘琛屾娴?
boxplot(data1$time,boxwex=0.7)
title("鐢熷瓨鏃堕棿寮傚父鍊兼娴嬬绾垮浘")
stats=boxplot.stats(data1$time)
# 鎵惧埌寮傚父鍊硷紝骞跺鐞?
id=which(data1$time %in% stats$out)
# 1銆佺洿鎺ュ垹闄?
data2=data1[-id,]
max(data2$time) # 鏌ョ湅鏈€澶у€?
# 2銆佸皢瓒呭嚭鍊肩敤鏈€澶у€兼垨鑰呮渶灏忓€?
data3=data1
data3[id,1]=740
# 3銆? 灏嗗紓甯稿€艰祴鍊间负绌哄€?
data4=data1
data4[id,1]=NA

# 缂哄け鍊兼娴嬩笌澶勭悊,鎺ata4缁х画澶勭悊
sum(complete.cases(data1))
sum(!complete.cases(data1))
mean(!complete.cases(data1))
# 缂哄け鍊煎鐞?
 #1銆佽鍒犻櫎娉?
data5=na.omit(data4)
 #2銆佸潎鍊兼浛浠ｆ硶
data6=data4
n=dim(data4)
for (i in c(1:n[2])){
  id=which(is.na(data4[,i]))
  data6[id,i]=mean(data4[,i],na.rm=T)
  rm(id)
}
# 3銆佷腑浣嶆暟鏇夸唬娉? 
data7=data4
for (i in c(1:n[2])){
  id=which(is.na(data4[,i]))
  data7[id,i]=median(data4[,i],na.rm=T)
  rm(id)
}
# 4銆佸閲嶆彃琛?
library(lattice)  #璋冨叆鍑芥暟鍖?
library(MASS)
library(nnet)
install.packages("mice")  # 涓嬭浇鍖咃紝鍓嶄笁涓寘鏄痬ice鐨勫熀纭€
library(mice) #鍔犺浇鍖?
imp<-mice(data4,m=4) # 4閲嶆彃琛ワ紝鍗崇敓鎴?4涓棤缂哄け鏁版嵁闆?
data8=complete(imp)

# 5銆佹満鍣ㄥ涔犵畻娉曞閲嶆彃琛?
# install.packages("missForest")
library(missForest)
data9=missForest(data4,ntree=1000)
data10=data9$ximp

write.csv(data9$ximp,"E:/璁茶/鏈哄櫒瀛︿範_R璇█瀹炵幇/lungpostprocess.csv")
