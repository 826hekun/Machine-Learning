import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split,cross_val_score, KFold, RepeatedKFold
from scipy.stats import pearsonr,ttest_ind, levene
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
# 读入数据
data=pd.read_excel(r'E:\class\data_breasttumor.xlsx')
data=data.drop(['ID','良/恶'],axis=1)
#拆分为训练集和测试集
 predictors=data.columns[1:]
x_train1,x_test1,y_train,y_test=model_selection.train_test_split(data[predictors],data.classes,test_size=0.2,random_state=1234)
x_train=StandardScaler().fit_transform(x_train1)
x_test=StandardScaler().fit_transform(x_test1)
# randomforest
model_rf=RandomForestClassifier(n_estimators=500).fit(x_train,y_train)
score_rf=model_rf.score(x_test,y_test)
print(score_rf)
# ROC 曲线
from sklearn.metrics import roc_curve, roc_auc_score
y_probs=model_rf.predict_proba(x_train)
fpr,tpr, thresholds=roc_curve(y_train,y_probs[:,1],pos_label=1)
plt.figure()
plt.plot(tpr, fpr, marker='o')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
# 计算AUC值
auc_score=roc_auc_score(y_train,model_rf.predict(x_train))
print(auc_score)
# 绘制校准曲线，参考：https://scikit-learn.org.cn/view/104.html
from sklearn.calibration import calibration_curve
prob_true,prob_pred=calibration_curve(y_train,y_probs[:,1],normalize=False,n_bins=5,strategy='uniform')
plt.figure()
plt.plot([0,1],[0,1],'k:',label="Perfectly calibrated")
plt.plot(prob_true,prob_pred)
plt.show()

