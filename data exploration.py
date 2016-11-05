# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 23:21:00 2016

@author: sanjeev sukumaran
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 
from sklearn import preprocessing,cross_validation
import numpy as np
controlgroup=pd.read_csv("finaldatasetcontrol1.csv")
heartgroup=pd.read_csv("finaldatasetheart1.csv")
g=sns.distplot(heartgroup.Totalcost)
g.set(xlim=(0,10000))
'''
heartgroup.hist()
controlgroup.hist()
#correlation matrix
plt.matshow(heartgroup.corr())


def plot_corr(df,size=10):
    

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    labels=['id','RR','C_S','U_U_C','A_D_R_R','a_d_i_r','a_d_a_r_r','a_u_d_a_r_r','mb_s','mb_e','mb_sub','mb_esec','mb_inp','mb_insec','mb_uneng','mb_idles']
    ax.set_xticklabels(labels,fontsize=10)
    ax.set_yticklabels(labels,fontsize=6)
    ax.matshow(corr)
    
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);






plot_corr(heartgroup,15)

#correlation plot#3
correlations = heartgroup.corr()
names=[]
names=list(heartgroup)
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

#check for outliers in the cost variable:

plt.boxplot(heartgroup.Totalcost)
plt.ylim(0, 10000)


heartgroup[np.abs(heartgroup.Totalcost-heartgroup.Totalcost.mean())<=(2*heartgroup.Totalcost.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
heartgroup[np.abs(heartgroup.Totalcost-heartgroup.Totalcost.mean())>(2*heartgroup.Totalcost.std())] #


sns.boxplot(heartgroup.Totalcost) 
sns.set(ylim=(10, 40))
sns.despine()
'''
'''
#checking for nulls
heartgroup.drop("HAD 12+ DRINKS OF ALCOHOL OVER ENTIRE LIFE",axis=1,inplace=True)
heartgroup.isnull().sum()
heartgroup["Life Statisfaction"].fillna(heartgroup["Life Statisfaction"].mean(),inplace=True)
heartgroup["Heart Treatment"].fillna(heartgroup["Heart Treatment"].mode()[0],inplace=True)
heartgroup["HAS OSTEOPOROSIS"].fillna(heartgroup["HAS OSTEOPOROSIS"].mode()[0],inplace=True)
heartgroup["Heart Attack"].fillna(heartgroup["Heart Attack"].mode()[0],inplace=True)
heartgroup["EVER HAD HEART FAILURE"].fillna(heartgroup["EVER HAD HEART FAILURE"].mode()[0],inplace=True)
heartgroup["ABNORMAL HEART RHYTHM"].fillna(heartgroup["ABNORMAL HEART RHYTHM"].mode()[0],inplace=True)
heartgroup["heartsurgery"].fillna(heartgroup["heartsurgery"].mode()[0],inplace=True)
#heartgroup["HAD 12+ DRINKS OF ALCOHOL OVER ENTIRE LIFE"].fillna(heartgroup["HAD 12+ DRINKS OF ALCOHOL OVER ENTIRE LIFE"].mean,inplace=True)
heartgroup["WEIGHT IN POUNDS"].fillna(heartgroup["WEIGHT IN POUNDS"].mean(),inplace=True)
heartgroup.set_index('HHID',inplace=True)
X=np.array(heartgroup.drop(['Totalcost'],1))
y=np.array(heartgroup['Totalcost'])
heartgroup1=preprocessing.scale(heartgroup)
heartgroup.dtypes
'''




#check for outliers in the cost variable:

plt.boxplot(heartgroup.Totalcost)
plt.ylim(0, 10000)


heartgroup[np.abs(heartgroup.Totalcost-heartgroup.Totalcost.mean())<=(2*heartgroup.Totalcost.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
heartgroup[np.abs(heartgroup.Totalcost-heartgroup.Totalcost.mean())>(2*heartgroup.Totalcost.std())] #

heartgroup=heartgroup[heartgroup['Heart Treatment']==1]
heartgroup
lm=sns.boxplot(heartgroup.Totalcost) 
axes = lm.axes
axes.set(xlim=(0, 5000))
sns.despine()
#heartgroup2=heartgroup[heartgroup['Totalcost']<=3200]
merged2=merged[merged['Totalcost']<=3200]


correlations = heartgroup2.corr()
names=[]
names=list(heartgroup)
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()
plt.matshow(heartgroup.corr())


import seaborn as sns
corr = heartgroup.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
            
merged2.isnull().sum()     
merged2=merged2.dropna()  
merged2=merged2.drop('HHID')   
from sklearn.linear_model import LinearRegression
X=np.array(merged2.drop(['Totalcost'],1))
X=preprocessing.scale(X)
y=np.array(merged2['Totalcost'])

#y1=np.array(heartgroup2['Totalcost'].apply(lambda x: (x - x.mean()) / (x.max() - x.min())))
#y1=(y - np.mean(y))/np.std(y)

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)



clf=LinearRegression()
clf.fit(X_train,y_train)

clf.score(X_test,y_test)



#-0.027 which is very bad


#using statsmodel
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




# normalize the target variable and remove outliers.

#checking only heart

onlyheart=pd.read_csv('only_heart.csv')
onlyheart.set_index('HHID',inplace=True)

onlyheart.rename(columns = {'OB000':'Life Statisfaction','OC005':'HIGH BLOOD PRESSURE',
'OC010':"DIABETES",'OC030':"LUNG DISEASE",'OC053':'STROKE','OC070':'ARTHRITIS','OC107':"OTHER MEDICAL CONDITIONS","OC128":"EVER DRINK ALCOHOL","OX060_MC":"Sex","OC280":"HAS OSTEOPOROSIS","OC139":"WEIGHT IN POUNDS","OC134":"HAD 12+ DRINKS OF ALCOHOL OVER ENTIRE LIFE"},inplace=True)
onlyheart.rename(columns = {'OA019':'Age','OC052':'heartsurgery','OC051':'Heart Treatment','OC037':'Heart Medication','OC240':'Shingles','OC257':'Heart Attack','OC263':'EVER HAD HEART FAILURE','OC269':'ABNORMAL HEART RHYTHM',},inplace=True)

#X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)



#No significant correlation between variables
#