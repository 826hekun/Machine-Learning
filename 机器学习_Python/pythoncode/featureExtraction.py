## 加载相应的工具包
import radiomics
from radiomics import featureextractor
import pandas as pd
## 数据输入
dataDir='E:/class/data/mri/'
folderList=['001','002','003']
extractor=featureextractor.RadiomicsFeatureExtractor()
df=pd.DataFrame()
for folder in folderList:
  imageName=dataDir+folder+'/data.nii'
  maskName=dataDir+folder+'/mask.nii.gz'
  featureVector=extractor.execute(imageName,maskName)
  df_add=pd.DataFrame.from_dict(featureVector.values()).T
  df_add.columns=featureVector.keys()
  df=pd.concat([df,df_add])
df.toexcel(dataDir+'results.xlsx')
  
  
