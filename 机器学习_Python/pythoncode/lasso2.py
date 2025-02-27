import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
# 璇诲彇鏁版嵁
data=pd.read_excel(r'E:\class\data_breasttumor.xlsx')
data=data.drop(['ID','鑹?/鎭?'],axis=1)
#鎷嗗垎涓鸿缁冮泦鍜屾祴璇曢泦
 predictors=data.columns[1:]
x_train1,x_test1,y_train,y_test=model_selection.train_test_split(data[predictors],data.classes,test_size=0.2,random_state=1234)
x_train=StandardScaler().fit_transform(x_train1)
x_test=StandardScaler().fit_transform(x_test1)
#鏋勯€犱笉鍚岀殑lambda鍊?
Lambdas=np.logspace(-2,3,200)
#璁剧疆浜ゅ弶楠岃瘉鐨勫弬鏁帮紝浣跨敤鍧囨柟璇樊璇勪及
lasso_cv=LassoCV(alphas=Lambdas,normalize=False,cv=10,max_iter=10000)
lasso_cv.fit(x_train,y_train)
# 鐢诲嚭璇樊闅忕潃lambdas鐨勫彉鍖栧浘
MSEs=lasso_cv.mse_path_
MSEs_mean=np.apply_along_axis(np.mean,1,MSEs)
MSEs_std=np.apply_along_axis(np.std,1,MSEs)
plt.figure(dpi=300) 
plt.errorbar(lasso_cv.alphas_,MSEs_mean      # x , y 
             ,yerr=MSEs_std         #璇樊妫?
             ,fmt='o'        # 鏁版嵁鐐硅鍙?
             ,ms=3           #璁板彿澶у皬
             ,mfc='r'        #棰滆壊
             ,mec='b'       # 杈圭紭棰滆壊
             ,ecolor='lightblue'  #璇樊妫掗鑹?
             ,elinewidth=2
             ,capsize=4      #璇樊杈圭晫绾块暱搴?
             ,capthick=1)      #璇樊杈圭晫绾垮帤搴?
plt.semilogx()  #鎹㈡垚log鍧愭爣
plt.axvline(lasso_cv.alpha_,color='black',ls='--') #鏍囧嚭鏈€浼樼殑鐐?
plt.xlabel('Lambda')
plt.ylabel('MSE')
ax=plt.gca()
# y_major_locator=MutipleLocator(1) #y 鍛ㄩ棿闅?
# ax.yaxis.set_major_locator(y_major_locator)
plt.show()
## 

#鐢诲嚭鍚勪釜鐗瑰緛绯绘暟闅忕潃lambdas鐨勫彉鍖?
coefs=lasso_cv.path(x_train, y_train,alphas=Lambdas
                   ,max_iter=10000)[1].T
plt.figure()
plt.semilogx(lasso_cv.alphas_, coefs,'-')
plt.axvline(lasso_cv.alpha_,color='black',ls='--')
plt.xlabel('Lambda')
plt.ylabel('Coefficients')
plt.show()
# 鎵撳嵃鍑烘渶浼榓lpha鍙婂悇涓壒寰佺郴鏁?
print(lasso_cv.alpha_)
coef=pd.Series(lasso_cv.coef_, index=x_train1.columns)
print(coef)
print("Lasso picked:"  +str(sum(coef !=0)))
## 灏嗘暟鎹繘琛岄噸鏁达紝鍙繚鐣欑瓫閫夊嚭鐨勭壒寰?
print(coef[coef!=0])

##  鏋勫缓閫昏緫鍥炲綊妯″瀷
from sklearn.linear_model import LogisticRegressionCV
# 瀹氫箟鍚戝墠鍚戝悗娉曪紝鍙傝€?:https://blog.csdn.net/weixin_46649052/article/details/114855669
def froward_back_select(train_data, test_data, target):
    """
   
    :param data: 鏁版嵁
    :param target:鐩爣鍊?
    :return:
    """
    variate = list(set(train_data.columns))
    variate.remove(target)
    selected = []  # 鍌ㄥ瓨鎸戦€夌殑鍙橀噺
    selected_h = []  # 瀛樺偍鍒犻櫎鐨勫彉閲?
    # 鍒濆鍖?
    # 鍒濆鍖栧喅瀹氱郴鏁癛^2,瓒婅繎浜?1瓒婂ソ
    cur_score_f, best_score_f = 0.0, 0.0
    cur_score_h, best_score_h = 0.0, 0.0
    # 寰幆鍒犻€夊彉閲?,鐩磋嚦瀵规墍鏈夊彉閲忚繘琛屼簡閫夋嫨
    # 鍙屽悜鎸戦€夆€斿厛涓ゆ鍓嶅悜鍐嶄竴姝ュ悗鍚?
    while variate:
        variate_r2_f = []
        variate_r2_h = []
        # 鎵惧埌灞€閮ㄦ渶浼?
        # 鍏堜袱姝ュ墠鍚?
        for i in range(2):
            for var in variate:
                selected.append(var)
                if len(selected) == 1:
                    model = Lasso().fit(train_data[selected[0]].values.reshape(-1, 1), train_data[target])
                    y_pred = model.predict(test_data[selected[0]].values.reshape(-1, 1))
                    R2 = r2_score(test_data[target], y_pred)
                    variate_r2_f.append((R2, var))
                    selected.remove(var)
                else:
                    model = Lasso().fit(train_data[selected], train_data[target])
                    y_pred = model.predict(test_data[selected])
                    R2 = r2_score(test_data[target], y_pred)
                    variate_r2_f.append((R2, var))
                    selected.remove(var)
            variate_r2_f.sort(reverse=False)  # 闄嶅簭鎺掑簭r2锛岄粯璁ゅ崌搴?
            best_score_f, best_var_f = variate_r2_f.pop()  # pop鐢ㄤ簬绉婚櫎鍒楄〃涓殑涓€涓厓绱狅紙榛樿鏈€鍚庝竴涓厓绱狅級锛屽苟涓旇繑鍥炶鍏冪礌鐨勫€?
            if cur_score_f < best_score_f:  # 璇存槑浜嗗姞浜嗚鍙橀噺鏇村ソ浜嗗氨涓嶇Щ闄や簡,鍚﹀垯灏辩Щ闄?
                selected.append(best_var_f)
                cur_score_f = best_score_f
                print("R2_f={},continue!".format(cur_score_f))
            else:
                variate.remove(best_var_f)
                break
        # 鍐嶄竴姝ュ悗鍚?
        for var in variate:
            variate.remove(var)
            if len(variate) == 1:
                model = Lasso().fit(train_data[variate[0]].values.reshape(-1, 1), train_data[target])
                y_pred = model.predict(test_data[variate[0]].values.reshape(-1, 1))
                R2 = r2_score(test_data[target], y_pred)
                variate_r2_h.append((R2, var))
                variate.append(var)
            else:
                model = Lasso().fit(train_data[variate], train_data[target])
                y_pred = model.predict(test_data[variate])
                R2 = r2_score(test_data[target], y_pred)
                variate_r2_h.append((R2, var))
                variate.append(var)
        variate_r2_h.sort(reverse=False)  # 鍗囧簭鎺掑簭r2锛岄粯璁ゅ崌搴?
        best_score_h, best_var_h = variate_r2_h.pop()  # pop鐢ㄤ簬绉婚櫎鍒楄〃涓殑涓€涓厓绱狅紙榛樿鏈€鍚庝竴涓厓绱狅級锛屽苟涓旇繑鍥炶鍏冪礌鐨勫€?
        if cur_score_h < best_score_h:  # 璇存槑浜嗙Щ闄や簡璇ュ彉閲忔洿濂戒簡
            variate.remove(best_var_h)
            selected_h.append(best_var_h)
            cur_score_h = best_score_h
            print("R2_h={},continue!".format(cur_score_h))
        else:
            print('for selection over!')
            selected = [var for var in set(train_data.columns) if var not in selected_h]
            selected_features = '+'.join([str(i) for i in selected])
            print(selected_features)
            break
# 鍒╃敤涓婅堪瀹氫箟鐨勫悜鍓嶅悜鍚庢硶杩涜鍙橀噺绛涢€?
index=coef[coef!=0].index
#index=np.where(coef!=0)
data1=data[:,index]
x_train=pd.DataFrame(data1, columns=index)
x_train['labels']= data.classes
    train_data=x_train
    test_data=x_train
 froward_back_select(train_data, test_data, 'labels')
 
## 鏋勫缓閫昏緫鍥炲綊妯″瀷
from sklearn.linear_model import LogisticRegressionCV
y=data.classes
model1=LogisticRegressionCV(cv=10,max_iter=100000).fit(data1,y)
probal=model1.predict_proba(data1)
#鐢籖OC鏇茬嚎鍜屾牎鍑嗘洸绾?
from sklearn.metrics import roc_curve, roc_auc_score
y_probs=probal
fpr,tpr, thresholds=roc_curve(y,y_probs[:,1],pos_label=1)
plt.figure()
plt.plot(tpr, fpr, marker='o')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()
# 璁＄畻AUC鍊?
auc_score=roc_auc_score(y,model1.predict(data1))
print(auc_score)
# 缁樺埗鏍″噯鏇茬嚎锛屽弬鑰冿細https://scikit-learn.org.cn/view/104.html
from sklearn.calibration import calibration_curve
prob_true,prob_pred=calibration_curve(y,y_probs[:,1],normalize=False,n_bins=5,strategy='uniform')
plt.figure()
plt.plot([0,1],[0,1],'k:',label="Perfectly calibrated")
plt.plot(prob_true,prob_pred)
plt.show()
