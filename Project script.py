# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 16:16:53 2016

@author: sanjeev sukumaran
"""

import numpy as np
'''
filter=pd.read_csv('Combined.csv')
'''
#only heart
'''
filter=filter[filter['OC036']==1]
len(filter.index)
filter=filter[filter['OC018']==5]

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
'''
'''
#heart with other disease
filter=filter[filter['OC036']==1]
len(filter.index)
filter.to_csv("heartandothers.csv")
'''
'''
#withoutheart disease
filter=filter[filter['OC036']==5]
len(filter.index)
filter.to_csv("without heart.csv")
'''
'''
cost=pd.read_csv('F:\\Capstone\\Dataset\\cost2.csv')
onlyheart=pd.read_csv('F:\\Capstone\\Dataset\\only_heart.csv')
heart_only=pd.merge(onlyheart,cost,on='HHID',how='left')
heart_only=heart_only.drop_duplicates(['HHID'],keep='last')
len(heart_only.index)
heart_only.to_csv("cost+heart.csv")
'''
'''
cost=pd.read_csv('F:\\Capstone\\Dataset\\cost2.csv')
heartandothers=pd.read_csv('F:\\Capstone\\Dataset\\heartandothers.csv')
heartandothers=pd.merge(heartandothers,cost,on='HHID',how='left')
heartandothers=heartandothers.drop_duplicates(['HHID'],keep='last')
len(heartandothers.index)
heartandothers.to_csv("heart+others.csv")
'''
'''
cost=pd.read_csv('F:\\Capstone\\Dataset\\cost2.csv')
withoutheart=pd.read_csv('F:\\Capstone\\Dataset\\without heart.csv')
withoutheart=pd.merge(withoutheart,cost,on='HHID',how='left')
withoutheart=withoutheart.drop_duplicates(['HHID'],keep='last')
len(withoutheart.index)
frac=len(withoutheart)*0.8
withoutheart=withoutheart.dropna(axis=1,thresh=frac)
withoutheart=withoutheart.drop(['OC270M1'],axis=1)
withoutheart.to_csv("cost+heart.csv")
'''
'''
cost=pd.read_csv('F:\\Capstone\\Dataset\\cost2.csv')
withoutheart=pd.read_csv('F:\\Capstone\\Dataset\\without heart.csv')
withoutheart=pd.merge(withoutheart,cost,on='HHID',how='left')
withoutheart=withoutheart.drop_duplicates(['HHID'],keep='last')
len(withoutheart.index)
frac=len(withoutheart)*0.8
withoutheart=withoutheart.dropna(axis=1,thresh=frac)
withoutheart=withoutheart.drop(['OC270M1'],axis=1)
withoutheart.to_csv("cost+heart.csv")
'''

'''
cost=pd.read_csv('F:\\Capstone\\Dataset\\cost2.csv')
heartandothers=pd.read_csv('F:\\Capstone\\Dataset\\heartandothers.csv')
heartandothers=pd.merge(heartandothers,cost,on='HHID',how='left')
heartandothers=heartandothers.drop_duplicates(['HHID'],keep='last')
len(heartandothers.index)
frac=len(heartandothers)*0.8
heartandothers=heartandothers.dropna(axis=1,thresh=frac)
heartandothers=heartandothers.drop(['OC270M1'],axis=1)
heartandothers.to_csv("heartandothers.csv")
'''

'''
heartandothers=pd.read_csv('F:\\Capstone\\Dataset\\final\\Perform regression\\heartandothers.csv')
additionalcolomns=pd.read_csv('F:\\Capstone\\Dataset\\final\\Perform regression\\Additional Colomns.csv')
heartandothers=pd.merge(heartandothers,additionalcolomns,on='HHID',how='left')
heartandothers=heartandothers.drop_duplicates(['HHID'],keep='last')
heartandothers.to_csv('F:\\Capstone\\Dataset\\final\\Perform regression\\Final1.csv')
'''
'''
withoutheart=pd.read_csv('C:\\temp\\Dataset\\final\\Perform regression\\without heart.csv')
additionalcolomns=pd.read_csv('C:\\temp\\Dataset\\final\\Perform regression\\Additional Colomns.csv')
withoutheart=pd.merge(withoutheart,additionalcolomns,on='HHID',how='left')
withoutheart=withoutheart.drop_duplicates(['HHID'],keep='last')
withoutheart.to_csv('C:\\temp\\Dataset\\final\\Perform regression\\withoutheart.csv')
'''
'''
heartandothers=pd.read_csv('C:\\temp\\Dataset\\final\\Perform regression\\subset\\heartandothers.csv')
withoutheart=pd.read_csv('C:\\temp\\Dataset\\final\\Perform regression\\subset\\withoutheart.csv')

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

# some value of 8 also found hence converted to 0.
heartandothers['OC005'].replace(8,0,inplace=True)        
heartandothers['OC010'].replace(8,0,inplace=True)   
heartandothers['OC036'].replace(8,0,inplace=True)   
heartandothers['OC037'].replace(8,0,inplace=True)   
heartandothers['OC107'].replace(8,0,inplace=True)   
heartandothers['OC280'].replace(8,0,inplace=True)   
heartandothers['OC240'].replace(8,0,inplace=True)   
heartandothers['OC070'].replace(8,0,inplace=True)    
heartandothers['OC053'].replace(8,0,inplace=True) 
heartandothers['OC030'].replace(8,0,inplace=True) 
heartandothers['OC257'].replace(8,0,inplace=True)           
heartandothers['OC263'].replace(8,0,inplace=True)             
heartandothers['OC269'].replace(8,0,inplace=True)    
heartandothers['OC051'].replace(8,0,inplace=True)    
heartandothers['OC052'].replace(8,0,inplace=True) 
heartandothers['OC052'].replace(8,0,inplace=True) 
heartandothers['OC128'].replace(8,0,inplace=True) 
heartandothers['OC134'].replace(8,0,inplace=True)


#similarlly for without heart

withoutheart['OC005'].replace(5,0,inplace=True)        
withoutheart['OC010'].replace(5,0,inplace=True)   
heartandothers['OC036'].replace(5,0,inplace=True)  #remove 
withoutheart['OC037'].replace(5,0,inplace=True)   #notpresent
withoutheart['OC107'].replace(5,0,inplace=True)   
withoutheart['OC280'].replace(5,0,inplace=True)   
withoutheart['OC240'].replace(5,0,inplace=True)   
withoutheart['OC070'].replace(5,0,inplace=True)    
withoutheart['OC053'].replace(5,0,inplace=True) 
withoutheart['OC030'].replace(5,0,inplace=True) 
#heartandothers['OC257'].replace(5,0,inplace=True)           
#heartandothers['OC263'].replace(5,0,inplace=True)             
#heartandothers['OC269'].replace(5,0,inplace=True)    
#heartandothers['OC051'].replace(5,0,inplace=True)    
#heartandothers['OC052'].replace(5,0,inplace=True) 
#heartandothers['OC052'].replace(5,0,inplace=True) 
withoutheart['OC128'].replace(5,0,inplace=True) 
withoutheart['OC134'].replace(5,0,inplace=True) 


withoutheart['OC005'].replace(8,0,inplace=True)        
withoutheart['OC010'].replace(8,0,inplace=True) 
withoutheart['OC107'].replace(8,0,inplace=True)   
withoutheart['OC280'].replace(8,0,inplace=True)   
withoutheart['OC240'].replace(8,0,inplace=True)   
withoutheart['OC070'].replace(8,0,inplace=True)    
withoutheart['OC053'].replace(8,0,inplace=True) 
withoutheart['OC030'].replace(8,0,inplace=True) 
withoutheart['OC128'].replace(8,0,inplace=True) 
withoutheart['OC134'].replace(8,0,inplace=True) 


#sum all the cost related in both the datasets
heartandothers['Totalcost']=heartandothers["ON204"]+heartandothers["ON205"]+heartandothers["ON206"]+heartandothers["ON207"]+heartandothers["ON209"]+heartandothers["ON210"]+heartandothers["ON064"]
withoutheart['Totalcost']=withoutheart["ON204"]+withoutheart["ON205"]+withoutheart["ON206"]+withoutheart["ON207"]+withoutheart["ON209"]+withoutheart["ON210"]+withoutheart["ON064"]

heartandothers=heartandothers[heartandothers.Totalcost!=0]
heartandothers    
withoutheart=withoutheart[withoutheart.Totalcost!=0]


#removing cost related variables
heartandothers.drop("ON204",axis=1,inplace=True)
heartandothers.drop("ON205",axis=1,inplace=True)
heartandothers.drop("ON206",axis=1,inplace=True)
heartandothers.drop("ON207",axis=1,inplace=True)
heartandothers.drop("ON209",axis=1,inplace=True)
heartandothers.drop("ON210",axis=1,inplace=True)
heartandothers.drop("ON064",axis=1,inplace=True)

withoutheart.drop("ON204",axis=1,inplace=True)
withoutheart.drop("ON205",axis=1,inplace=True)
withoutheart.drop("ON206",axis=1,inplace=True)
withoutheart.drop("ON207",axis=1,inplace=True)
withoutheart.drop("ON209",axis=1,inplace=True)
withoutheart.drop("ON210",axis=1,inplace=True)
withoutheart.drop("ON064",axis=1,inplace=True)

heartandothers.to_csv("C:\\temp\\Dataset\\final\\Perform regression\\subset\\heartandothers2.csv")

withoutheart.to_csv("C:\\temp\\Dataset\\final\\Perform regression\\subset\\withoutheart2.csv")

'''
'''
heartandothers=pd.read_csv('F:\\Capstone\\Dataset\\final\\Perform regression\\subset\\heartandothers2.csv')
withoutheart=pd.read_csv('F:\\Capstone\\Dataset\\final\\Perform regression\\subset\\withoutheart2.csv')


withoutheart['OX060_MC'].replace(2,0,inplace=True)  
heartandothers['OX060_MC'].replace(2,0,inplace=True)

#for variable OC005
heartandothers['OC005'].replace([4,5,8,9],0,inplace=True)
heartandothers['OC005'].replace([1,3],1,inplace=True)

withoutheart['OC005'].replace([4,5,8,9],0,inplace=True)
withoutheart['OC005'].replace([1,3],1,inplace=True)


#for variable OC010

heartandothers['OC010'].replace([4,5,8,9],0,inplace=True)
heartandothers['OC010'].replace([1,3],1,inplace=True)

withoutheart['OC010'].replace([4,5,8,9],0,inplace=True)
withoutheart['OC010'].replace([1,3],1,inplace=True)

#FOR variable OC107 and OC037

heartandothers['OC107'].replace([5,8,9],0,inplace=True)

withoutheart['OC107'].replace([5,8,9],0,inplace=True)

heartandothers['OC037'].replace([5,8,9],0,inplace=True)

#FOR variable OC280
heartandothers['OC280'].replace([5,8,9],0,inplace=True)
withoutheart['OC280'].replace([5,8,9],0,inplace=True)
#FOR variable OC240
heartandothers['OC240'].replace([5,8,9],0,inplace=True)
withoutheart['OC240'].replace([5,8,9],0,inplace=True)


#For variable OC070
heartandothers['OC070'].replace([4,5,8,9],0,inplace=True)
heartandothers['OC070'].replace([1,3],1,inplace=True)

withoutheart['OC070'].replace([4,5,8,9],0,inplace=True)
withoutheart['OC070'].replace([1,3],1,inplace=True)

#For variable OC053
heartandothers['OC070'].replace([4,5,8,9],0,inplace=True)
heartandothers['OC070'].replace([1,2,3],1,inplace=True)

withoutheart['OC070'].replace([4,5,8,9],0,inplace=True)
withoutheart['OC070'].replace([1,2,3],1,inplace=True)

#For variable OC030
heartandothers['OC030'].replace([4,5,8,9],0,inplace=True)
heartandothers['OC030'].replace([1,3],1,inplace=True)

withoutheart['OC030'].replace([4,5,8,9],0,inplace=True)
withoutheart['OC030'].replace([1,3],1,inplace=True)

#for variable OC257

heartandothers['OC257'].replace([5,8,9],0,inplace=True)

withoutheart['OC257'].replace([5,8,9],0,inplace=True)

#for variable OC263

heartandothers['OC263'].replace([5,8,9],0,inplace=True)

withoutheart['OC263'].replace([5,8,9],0,inplace=True)

#for variable OC269
heartandothers['OC269'].replace([5,8,9],0,inplace=True)

withoutheart['OC269'].replace([5,8,9],0,inplace=True)

#for variable OC051 
heartandothers['OC051'].replace([5,8,9],0,inplace=True)

withoutheart['OC051'].replace([5,8,9],0,inplace=True)


#for variable OC052
heartandothers['OC052'].replace([5,8,9],0,inplace=True)
withoutheart['OC052'].replace([5,8,9],0,inplace=True)



#for variable OC128
heartandothers['OC128'].replace([5,8,9],0,inplace=True)
withoutheart['OC128'].replace([5,8,9],0,inplace=True)

#for variable OC134
heartandothers['OC134'].replace([5,6,8,9],0,inplace=True)
heartandothers['OC134'].replace([1,2],1,inplace=True)

withoutheart['OC134'].replace([5,6,8,9],0,inplace=True)
withoutheart['OC134'].replace([1,2],1,inplace=True)


heartandothers.to_csv("F:\\Capstone\\Dataset\\final\\Perform regression\\subset\\heartandothersALMOST.csv")

withoutheart.to_csv("F:\\Capstone\\Dataset\\final\\Perform regression\\subset\\WITHOUTHEARTalmost.csv")








withoutheart['OC010'].replace(3,0,inplace=True) 
withoutheart['OC107'].replace(3,0,inplace=True)   
withoutheart['OC280'].replace(3,0,inplace=True)   
withoutheart['OC240'].replace(3,0,inplace=True)   
withoutheart['OC070'].replace(3,0,inplace=True)    
withoutheart['OC053'].replace(3,0,inplace=True) 
withoutheart['OC030'].replace(3,0,inplace=True) 
withoutheart['OC128'].replace(3,0,inplace=True) 
withoutheart['OC134'].replace(3,0,inplace=True) 
'''
'''
# FOR VARIABLE OC005
withoutheart.ix[withoutheart['OC005'] >3,'0C005'] = 0
withoutheart.ix[withoutheart['OC005']<=3,'0C005'] = 1

heartandothers.ix[heartandothers['OC005'] >3,'0C005'] = 0
heartandothers.ix[heartandothers['OC005']<=3,'0C005'] = 1

#for variable OC010

withoutheart['OC010'] = (withoutheart['OC010']<=3).astype(int)
withoutheart.ix[withoutheart['OC010']>3,'0C010']=0
withoutheart.ix[withoutheart['OC010']<=3,'0C010']=1

heartandothers.ix[heartandothers['OC010'] >3,'0C010'] = 0
heartandothers.ix[heartandothers['OC010']<=3,'0C010'] = 1

#for variable OC036

heartandothers[heartandothers['OC036']>3]=0
heartandothers[heartandothers['OC036']<=3]=1

withoutheart[withoutheart['OC036']>3]=0
withoutheart[withoutheart['OC036']<=3]=1
'''

import pandas as pd
heartandothers=pd.read_csv("F:\\Capstone\\Dataset\\final\\Perform regression\\subset\\ALMOST\\heartandothersALMOST.csv")
withoutheart=pd.read_csv("F:\\Capstone\\Dataset\\final\\Perform regression\\subset\\ALMOST\\WITHOUTHEARTalmost.csv")

#final datapreparation without heart

withoutheart.drop('OC036',axis=1,inplace=True)
withoutheart.isnull().sum()

withoutheart['OB000']=withoutheart.fillna(withoutheart['OB000'].mean())

#time to match
type(heartandothers)
heartandothers
matches=[1]
matchesb=[1]
matched=pd.DataFrame()
for index,persons in heartandothers.iterrows():
    for age_allowance in [0,1]:
        for index,person2 in withoutheart.iterrows():
            with open('sanTEST.txt','a') as f:
                if ((persons['OX060_MC']==person2['OX060_MC']) & (abs(persons['OA019']-person2['OA019'])<age_allowance)):
                        i=persons['HHID']
                        j=person2['HHID']
                        if ((not i in matches) & (not j in matchesb) & (i!=j)):
                            string=str(persons['HHID'])+","+str(person2['HHID'])+","+str(persons['OX060_MC'])+","+str(person2['OX060_MC'])+","+str(persons['OA019'])+","+str(person2['OA019'])+"\n"
                            print(string)
                            f.write(string)
                            matches.append(i)
                            matchesb.append(j)
                            
#creating new subset of datasets based on matches
matching=pd.read_csv("Matching.csv")
#DONE BY USING COMBINATION OF EXCEL AND PYTHON
controlgroup1=pd.read_csv("WITHOUTHEARTalmost.csv")
heartgroup1=pd.read_csv("heartandothersALMOST.csv")
heartgroup=pd.merge(heartgroup1,matching,on="HHID",how="inner")
# changing the colomn name to HHID instead of control group in excel
controlgroup=pd.merge(controlgroup1,matching,on="HHID",how="inner")   
#removing variables:

controlgroup.drop('OC036',axis=1,inplace=True)
controlgroup.drop(['Heart',],axis=1,inplace=True)



heartgroup.drop('OC036',axis=1,inplace=True)
controlgroup.drop('OC036',axis=1,inplace=True)
heartgroup.drop(['HeartGender','ControlGender','HeartAge','ControlAge','ControlID'],axis=1,inplace=True)
controlgroup.drop(['HeartGender','ControlGender','HeartAge','ControlAge','HeartID'],axis=1,inplace=True)
#cleaning control group dataset
controlgroup.OC141=[i*12 for i in controlgroup.OC141]
#create height variable
controlgroup['height']=controlgroup['OC141']+controlgroup['OC142']
controlgroup['height'].fillna(controlgroup['height'].mean(),inplace=True)
controlgroup.drop(['OC141','OC142'],axis=1,inplace=True)

controlgroup.drop(['Unnamed:0'],axis=1,inplace=True)
controlgroup.drop(['HeartGender'],axis=1,inplace=True)
controlgroup.drop(['HeartAge'],axis=1,inplace=True)
controlgroup.drop(['ControlGender'],axis=1,inplace=True)
controlgroup.drop(['ControlAge'],axis=1,inplace=True)
controlgroup.drop(['Unnamed:0'],axis=1,inplace=True)
#checking null
controlgroup.isnull().sum()
controlgroup.OB000.fillna(controlgroup.OB000.mean(),inplace=True)


#same case with final heart dataset
heartgroup.OC141=[i*12 for i in heartgroup.OC141]
heartgroup['height']=heartgroup['OC141']+heartgroup['OC142']
heartgroup['height'].fillna(heartgroup['height'].mean(),inplace=True)
heartgroup.drop(['OC141','OC142'],axis=1,inplace=True)


heartgroup=pd.read_csv("finaldatasetheart.csv")
heartgroup.rename(columns = {'OB000':'Life Statisfaction','OC005':'HIGH BLOOD PRESSURE',
'OC010':"DIABETES",'OC030':"LUNG DISEASE",'OC053':'STROKE','OC070':'ARTHRITIS','OC107':"OTHER MEDICAL CONDITIONS","OC128":"EVER DRINK ALCOHOL","OX060_MC":"Sex","OC280":"HAS OSTEOPOROSIS","OC139":"WEIGHT IN POUNDS","OC134":"HAD 12+ DRINKS OF ALCOHOL OVER ENTIRE LIFE"},inplace=True)
heartgroup.rename(columns = {'OA019':'Age','OC052':'heartsurgery','OC051':'Heart Treatment','OC037':'Heart Medication','OC240':'Shingles','OC257':'Heart Attack','OC263':'EVER HAD HEART FAILURE','OC269':'ABNORMAL HEART RHYTHM',},inplace=True)
heartgroup.drop('OC036',axis=1,inplace=True)



controlgroup.rename(columns = {'OB000':'Life Statisfaction','OC005':'HIGH BLOOD PRESSURE',
'OC010':"DIABETES",'OC030':"LUNG DISEASE",'OC053':'STROKE','OC070':'ARTHRITIS','OC107':"OTHER MEDICAL CONDITIONS","OC128":"EVER DRINK ALCOHOL","OX060_MC":"Sex","OC280":"HAS OSTEOPOROSIS","OC139":"WEIGHT IN POUNDS","OC134":"HAD 12+ DRINKS OF ALCOHOL OVER ENTIRE LIFE"},inplace=True)
controlgroup.rename(columns = {'OA019':'Age','OC052':'heartsurgery','OC051':'Heart Treatment','OC037':'Heart Medication','OC240':'Shingles','OC257':'Heart Attack','OC263':'EVER HAD HEART FAILURE','OC269':'ABNORMAL HEART RHYTHM',},inplace=True)
controlgroup.to_csv("finaldatasetcontrol1.csv")
heartgroup.to_csv("finaldatasetheart1.csv")
cont.drop('HeartAge',axis=1,inplace=True)
heartgroup.drop('HeartGender',axis=1,inplace=True)


controlgroup.to_csv("finaldatasetcontrol1.csv")
heartgroup=pd.read_csv("finaldatasetheart1.csv")
#removing null in variables by mean replacement

#Starting with data exploration
#start with heart dataset
heartgroup.hist()
controlgroup.hist()


#check for correlation

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





                  
