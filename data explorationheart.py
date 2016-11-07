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

#check for distribution of heartandothers dataset
heartgroup.hist()


#check for correlation in data
import seaborn as sns
corr = heartgroup.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)



#create boxplot to identify outliers and remove them
lm=sns.boxplot(heartgroup.Totalcost) 
axes = lm.axes
axes.set(xlim=(0, 5000))
sns.despine()
#heartgroup2=heartgroup[heartgroup['Totalcost']<=3200]
merged2=merged[merged['Totalcost']<=3200]




