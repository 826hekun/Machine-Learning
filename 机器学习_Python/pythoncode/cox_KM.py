# cox regression
import pandas as pd                
import numpy as np                 
import matplotlib.pyplot as plt 
import seaborn as sns    #初次使用需要下载 cmd中py -m pip install "seaborn"
from lifelines import KaplanMeierFitter, CoxPHFitter        
from lifelines.statistics import logrank_test, multivariate_logrank_test 
from sklearn.model_selection import train_test_split, GridSearchCV
from lifelines.utils.sklearn_adapter import sklearn_adapter
from sklearn import metrics
import time

plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False  
import warnings
warnings.filterwarnings("ignore")
# 读取数据
data = pd.read_csv('D:/Python/WA_Fn-UseC_-Telco-Customer-Churn.csv')
data1=pd.read_csv('D:/Python/lung.csv')
#预处理数据
# 删除ID列
data1.drop("inst",axis=1, inplace=True)
# data.drop("customerID", axis=1, inplace=True)
# 转换成连续型变量
#data['TotalCharges'] = pd.to_numeric(data.TotalCharges, errors='coerce')
# 去掉缺失值
data2=data1.dropna()
# 重置索引
# data1 = data1.reset_index().drop('index',axis=1)
# 转换成类别变量
# data.SeniorCitizen = data.SeniorCitizen.astype("str")
# 将是否流失转换为01分布
# data.Churn = data.Churn.map({'No':0,'Yes':1})
# 生成哑变量
# df = pd.get_dummies(data1) 
# 查看整体K-M曲线
fig, ax = plt.subplots(figsize=(10,8)) 
kmf = KaplanMeierFitter()
kmf.fit(data2.time,  # 代表生存时长
        event_observed=data2.status,  # 代表事件的终点
        )

kmf.plot_survival_function(at_risk_counts=True,ax=ax)
plt.show()
## Cox模型
train, test = train_test_split(data2, test_size=0.2)
cph = CoxPHFitter(penalizer = 0.01)
cph.fit(train,  duration_col='time', event_col='status')
cph.print_summary()
## 绘制各个因素的重要性
fig,ax = plt.subplots(figsize=(12,9))
cph.plot(ax=ax)
plt.show()
## 绘制某个因素的K-M曲线
fig,ax = plt.subplots(figsize=(12,9))
cph.plot_partial_effects_on_outcome('sex', values=[1, 2], cmap='coolwarm', ax=ax)
plt.show()
# 利用对数秩检验得到两个曲线的差异性
p_value = multivariate_logrank_test(event_durations = data.tenure,  # 代表生存时长
                                            groups=data[feature],  # 代表检验的组别
                                            event_observed=data.Churn  # 代表事件的终点
                                           ).p_value
