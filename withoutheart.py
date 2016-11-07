# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:39:23 2016

@author: sanjeev sukumaran
"""
#again
import pandas as pd
import numpy as np
import seaborn as sns


withoutheart=pd.read_csv('withoutheart.csv')


#replacing for categorical variable
withoutheart['OC005'].replace(5,0,inplace=True)        
withoutheart['OC010'].replace(5,0,inplace=True)   
withoutheart['OC037'].replace(5,0,inplace=True)   
withoutheart['OC107'].replace(5,0,inplace=True)   
withoutheart['OC280'].replace(5,0,inplace=True)   
withoutheart['OC240'].replace(5,0,inplace=True)   
withoutheart['OC070'].replace(5,0,inplace=True)    
withoutheart['OC053'].replace(5,0,inplace=True) 
withoutheart['OC030'].replace(5,0,inplace=True) 
withoutheart['OC128'].replace(5,0,inplace=True) 
withoutheart['OC134'].replace(5,0,inplace=True) 


withoutheart['Totalcost']=withoutheart["ON204"]+withoutheart["ON205"]+withoutheart["ON206"]+withoutheart["ON207"]+withoutheart["ON209"]+withoutheart["ON210"]+withoutheart["ON064"]
withoutheart=withoutheart.dropna()
withoutheart=withoutheart[withoutheart.Totalcost!=0]


withoutheart.drop("ON204",axis=1,inplace=True)
withoutheart.drop("ON205",axis=1,inplace=True)
withoutheart.drop("ON206",axis=1,inplace=True)
withoutheart.drop("ON207",axis=1,inplace=True)
withoutheart.drop("ON209",axis=1,inplace=True)
withoutheart.drop("ON210",axis=1,inplace=True)
withoutheart.drop("ON064",axis=1,inplace=True)
#gender 0 female 1 male
withoutheart['OX060_MC'].replace(2,0,inplace=True)

#for variable OC005
withoutheart['OC005'].replace([4,5],0,inplace=True)
withoutheart['OC005'].replace([1,3],1,inplace=True)
withoutheart['OC005'].replace([8,9],np.nan,inplace=True)

#for varibale OC010
withoutheart['OC010'].replace([4,5],0,inplace=True)
withoutheart['OC010'].replace([1,3],1,inplace=True)
withoutheart['OC010'].replace([8,9],np.nan,inplace=True)




#for variable OC037
withoutheart['OC037'].replace([5],0,inplace=True)
withoutheart['OC037'].replace([8,9],np.nan,inplace=True)

#FOR variable OC280
withoutheart['OC280'].replace([5],0,inplace=True)
withoutheart['OC280'].replace([8,9],np.nan,inplace=True)

#for variable OC240

withoutheart['OC240'].replace([5],0,inplace=True)
withoutheart['OC240'].replace([8,9],np.nan,inplace=True)


#For variable OC070
withoutheart['OC070'].replace([4,5],0,inplace=True)
withoutheart['OC070'].replace([1,3],1,inplace=True)
withoutheart['OC070'].replace([8,9],np.nan,inplace=True)

#for variable OC053
withoutheart['OC053'].replace([4,5],0,inplace=True)
withoutheart['OC053'].replace([1,3],1,inplace=True)
withoutheart['OC053'].replace([8,9],np.nan,inplace=True)

#For variable OC030

withoutheart['OC030'].replace([4,5],0,inplace=True)
withoutheart['OC030'].replace([1,3],1,inplace=True)
withoutheart['OC030'].replace([8,9],np.nan,inplace=True)

#for variable OC257
withoutheart['OC257'].replace([5],0,inplace=True)
withoutheart['OC257'].replace([8,9],np.nan,inplace=True)

#for variable OC263

withoutheart['OC263'].replace([5],0,inplace=True)
withoutheart['OC263'].replace([8,9],np.nan,inplace=True)


#for variable OC269

withoutheart['OC269'].replace([5],0,inplace=True)
withoutheart['OC269'].replace([8,9],np.nan,inplace=True)

#for variable OC051 

withoutheart['OC051'].replace([5],0,inplace=True)
withoutheart['OC051'].replace([8,9],np.nan,inplace=True)

#for variable OC052'

withoutheart['OC052'].replace([5],0,inplace=True)
withoutheart['OC052'].replace([8,9],np.nan,inplace=True)

#for variable OC128

withoutheart['OC128'].replace([5],0,inplace=True)
withoutheart['OC128'].replace([8,9],np.nan,inplace=True)

#for variable OC134
withoutheart['OC134'].replace([5],0,inplace=True)
withoutheart['OC134'].replace([8,9],np.nan,inplace=True)

#same case with final heart dataset
withoutheart.OC141=[i*12 for i in withoutheart.OC141]
withoutheart['height']=withoutheart['OC141']+withoutheart['OC142']
withoutheart['height'].fillna(withoutheart['height'].mean(),inplace=True)
withoutheart.drop(['OC141','OC142'],axis=1,inplace=True)

withoutheart=withoutheart.dropna()
withoutheart.hist()
withoutheart[withoutheart['Totalcost']>100]
g=sns.distplot(withoutheart.Totalcost)
g.set(xlim=(0,10000))

withoutheart.isnull().sum()

import seaborn as sns
corr = withoutheart.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# add insurance variable
insurance=pd.read_csv('insurance.csv')
merged=pd.merge(withoutheart,insurance,on='HHID',how='inner')
merged.drop('HHID')
merged=merged.dropna()
merged=merged.drop(['OC036'],axis=1)
merged=merged.drop(['Unnamed:0'],axis=1)

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing,cross_validation
X=np.array(merged.drop(['Totalcost'],1))
X=preprocessing.scale(X)
y=np.array(merged['Totalcost'])



from __future__ import print_function
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std


#model = sm.OLS(formula='Totalcost~height+Sex+Age+WEIGHT IN POUNDS+heartsurgery+EVER DRINK ALCOHOL+Heart Treatment+ABNORMAL HEART RHYTHM+EVER HAD HEART FAILURE+Heart Attack+LUNG DISEASE+STROKE+ARTHRITIS+Shingles+HAS OSTEOPOROSIS+OTHER MEDICAL CONDITIONS+Heart Medication+DIABETES+HIGH BLOOD PRESSURE+Life Statisfaction', data=heartgroup2)
model=sm.OLS(y,X)
model = sm.OLS(formula='Totalcost~height+Sex+Age', data=heartgroup2)
results = model.fit()
print(results.summary())






