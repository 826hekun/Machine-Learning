#t检验
from scipy.stats import levene, ttest_ind
x1=[110,121,115,118,125,115,113,118,119,117,119]
x2=[112,123,135,125,129,124,130,128,121,123,124,131,132]
levene(x1,x2)  #方差齐次性
if levene(x1,x2)[1]>0.05:
    res=ttest_ind(x1,x2)
else:
    res=ttest_ind(x,y,equal_var=False)
print(res)
print(res[1])
