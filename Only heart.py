# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:16:53 2016

@author: sanjeev sukumaran
"""

import numpy as np

filter=pd.read_csv('Combined.csv')

#only heart
filter=filter[filter['OC036']==1]
len(filter.index)
filter=filter[filter['OC018']==5]
#removing variables related to other disease
filter=filter[(filter['OC030']==5) & (filter['OC053']==5) & (filter['OC070']==5)]
filter=filter[(filter['OC240']==5) & (filter['OC280']==5) & (filter['OC098']==5)]
filter=filter[(filter['OC101']==5) & (filter['OC107']==5)]

filter=filter.drop(['OC018','OC030','OC053','OC070','OC240','OC280','OC098','OC101','OC107','OC018'],axis=1)
len(filter.index)
filter.to_csv('san.csv')
frac=len(filter)*0.8
filter1=filter.dropna(axis=1,thresh=frac)
filter1.to_csv('san3.csv')
filter1=filter1.drop(['OC270M1'],axis=1)
filter1
filter1.to_csv('only_heart.csv')







                  
