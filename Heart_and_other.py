# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:39:23 2016

@author: sanjeev sukumaran
"""
#again
import pandas as pd
import numpy as np
import seaborn as sns


heartandothers=pd.read_csv('heartandothers.csv')


#replacing for categorical variable
heartandothers['OC005'].replace(5,0,inplace=True)        
heartandothers['OC010'].replace(5,0,inplace=True)   
heartandothers['OC036'].replace(5,0,inplace=True)   
heartandothers['OC037'].replace(5,0,inplace=True)   
heartandothers['OC107'].replace(5,0,inplace=True)   
heartandothers['OC280'].replace(5,0,inplace=True)   
heartandothers['OC240'].replace(5,0,inplace=True)   
heartandothers['OC070'].replace(5,0,inplace=True)    
heartandothers['OC053'].replace(5,0,inplace=True) 
heartandothers['OC030'].replace(5,0,inplace=True) 
heartandothers['OC257'].replace(5,0,inplace=True)           
heartandothers['OC263'].replace(5,0,inplace=True)             
heartandothers['OC269'].replace(5,0,inplace=True)    
heartandothers['OC051'].replace(5,0,inplace=True)    
heartandothers['OC052'].replace(5,0,inplace=True) 
heartandothers['OC052'].replace(5,0,inplace=True) 
heartandothers['OC128'].replace(5,0,inplace=True) 
heartandothers['OC134'].replace(5,0,inplace=True) 


heartandothers['Totalcost']=heartandothers["ON204"]+heartandothers["ON205"]+heartandothers["ON206"]+heartandothers["ON207"]+heartandothers["ON209"]+heartandothers["ON210"]+heartandothers["ON064"]
heartandothers=heartandothers.dropna()
heartandothers=heartandothers[heartandothers.Totalcost!=0]


heartandothers.drop("ON204",axis=1,inplace=True)
heartandothers.drop("ON205",axis=1,inplace=True)
heartandothers.drop("ON206",axis=1,inplace=True)
heartandothers.drop("ON207",axis=1,inplace=True)
heartandothers.drop("ON209",axis=1,inplace=True)
heartandothers.drop("ON210",axis=1,inplace=True)
heartandothers.drop("ON064",axis=1,inplace=True)
#gender 0 female 1 male
heartandothers['OX060_MC'].replace(2,0,inplace=True)

#for variable OC005
heartandothers['OC005'].replace([4,5],0,inplace=True)
heartandothers['OC005'].replace([1,3],1,inplace=True)
heartandothers['OC005'].replace([8,9],np.nan,inplace=True)

#for varibale OC010
heartandothers['OC010'].replace([4,5],0,inplace=True)
heartandothers['OC010'].replace([1,3],1,inplace=True)
heartandothers['OC010'].replace([8,9],np.nan,inplace=True)




#for variable OC037
heartandothers['OC037'].replace([5],0,inplace=True)
heartandothers['OC037'].replace([8,9],np.nan,inplace=True)

#FOR variable OC280
heartandothers['OC280'].replace([5],0,inplace=True)
heartandothers['OC280'].replace([8,9],np.nan,inplace=True)

#for variable OC240

heartandothers['OC240'].replace([5],0,inplace=True)
heartandothers['OC240'].replace([8,9],np.nan,inplace=True)


#For variable OC070
heartandothers['OC070'].replace([4,5],0,inplace=True)
heartandothers['OC070'].replace([1,3],1,inplace=True)
heartandothers['OC070'].replace([8,9],np.nan,inplace=True)

#for variable OC053
heartandothers['OC053'].replace([4,5],0,inplace=True)
heartandothers['OC053'].replace([1,3],1,inplace=True)
heartandothers['OC053'].replace([8,9],np.nan,inplace=True)

#For variable OC030

heartandothers['OC030'].replace([4,5],0,inplace=True)
heartandothers['OC030'].replace([1,3],1,inplace=True)
heartandothers['OC030'].replace([8,9],np.nan,inplace=True)

#for variable OC257
heartandothers['OC257'].replace([5],0,inplace=True)
heartandothers['OC257'].replace([8,9],np.nan,inplace=True)

#for variable OC263

heartandothers['OC263'].replace([5],0,inplace=True)
heartandothers['OC263'].replace([8,9],np.nan,inplace=True)


#for variable OC269

heartandothers['OC269'].replace([5],0,inplace=True)
heartandothers['OC269'].replace([8,9],np.nan,inplace=True)

#for variable OC051 

heartandothers['OC051'].replace([5],0,inplace=True)
heartandothers['OC051'].replace([8,9],np.nan,inplace=True)

#for variable OC052'

heartandothers['OC052'].replace([5],0,inplace=True)
heartandothers['OC052'].replace([8,9],np.nan,inplace=True)

#for variable OC128

heartandothers['OC128'].replace([5],0,inplace=True)
heartandothers['OC128'].replace([8,9],np.nan,inplace=True)

#for variable OC134
heartandothers['OC134'].replace([5],0,inplace=True)
heartandothers['OC134'].replace([8,9],np.nan,inplace=True)

#same case with final heart dataset
heartandothers.OC141=[i*12 for i in heartandothers.OC141]
heartandothers['height']=heartandothers['OC141']+heartandothers['OC142']
heartandothers['height'].fillna(heartandothers['height'].mean(),inplace=True)
heartandothers.drop(['OC141','OC142'],axis=1,inplace=True)

heartandothers=heartandothers.dropna()
heartandothers.hist()
heartandothers[heartandothers['Totalcost']>100]
g=sns.distplot(heartandothers.Totalcost)
g.set(xlim=(0,10000))

heartandothers.isnull().sum()

import seaborn as sns
corr = heartandothers.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

# add insurance variable
insurance=pd.read_csv('insurance.csv')
merged=pd.merge(heartandothers,insurance,on='HHID',how='inner')
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






