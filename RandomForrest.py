# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 10:45:14 2016

@author: cfleming
"""
#Import libraries
import pyodbc
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc

#Pull DigitalMart data in
print('Setup SQL')
cnxn = pyodbc.connect(r'Driver={SQL Server};Server=hqetgsql09;Database=digitalmart;Trusted_Connection=yes;') 
cursor = cnxn.cursor()
sql = """
SELECT 
 joined.placementid                
 ,Date
,lower(DcmMetaPnc.[LOB]) as [LOB]
,lower(DcmMetaPNC.[Product]) as [Product]
,lower(DcmMetaPnc.[SubTactic]) as [SubTactic]
,lower(DcmMetaPnc.[FunnelStage]) as [FunnelStage]
,lower(DcmMetaPnc.[Platform]) as [Platform]
,lower(DcmMetaPnc.[InventorySource]) as [InventorySource]
,lower(replace(dcmmeta.[CreativeField2],'Sub-Concept: ','')) as SubConcept
,lower(replace(dcmmeta.[CreativeField4],'Primary Message: ','')) as PrimaryMessage
,sum(joined.[Impressions]           ) as [Impressions]
,sum(joined.[Clicks]                    ) as [Clicks]
,sum(joined.[VideoCompletions]) as [VideoCompletions]
,sum(joined.[PageViews]                        ) as [PageViews]
,sum(joined.[Visits]                     ) as [Visits]
,sum(joined.[Start]                                  ) as [Start]
,sum(joined.[TotalApprovals]  ) as [TotalApprovals]
,sum(joined.[TotalCompletes]  ) as [TotalCompletes]
FROM [DigitalMart].[dev].[vw_14dayDcmJoined] joined
join digitalmart.dbo.DcmMeta DcmMeta
on joined.[PlacementID] = DcmMeta.[PlacementID] and joined.[AdID] = DcmMeta.[AdID] and joined.[CreativeID] = DcmMeta.[CreativeID]
join digitalmart.dbo.DcmMetaPnc DcmMetaPnc on joined.[PlacementID] = DcmMetaPnc.[PlacementID]
where date >= '2016-02-01' and date < '2016-06-01'
and [CreativeField1] != ''
and ([SubTactic] != '' or [SubTactic] is not null)
group by 
 joined.placementid                
 ,date
,lower(replace(dcmmeta.[CreativeField2],'Sub-Concept: ','')) 
 ,lower(replace(dcmmeta.[CreativeField4],'Primary Message: ','')) 
 ,lower(DcmMetaPnc.[LOB]) 
 ,lower(DcmMetaPNC.[Product]) 
 ,lower(DcmMetaPnc.[SubTactic]) 
 ,lower(DcmMetaPnc.[FunnelStage]) 
 ,lower(DcmMetaPnc.[Platform]) 
 ,lower(DcmMetaPnc.[InventorySource]) 
 having sum(impressions) > 0;"""

print('Run SQL')
df1 = pd.read_sql(sql,cnxn)
cnxn.close()

#Calculate Conversion Ratios
df1['cr_p'] = df1['PageViews']/df1['Impressions']
df1['cr_s'] = df1['Start']/df1['Impressions']
df1['cr_v'] = df1['Visits']/df1['Impressions']
df1['cr_sv'] = df1['Start']/df1['Visits']
df1['cr_cs'] = df1['TotalCompletes']/df1['Start']
df1['cr_ac'] = df1['TotalApprovals']/df1['TotalCompletes']

#Convert datatype to numeric where possible and set Na/NaN to zeroes
df1= df1.apply(lambda x: pd.to_numeric(x, errors='ignore'))
df1 = df1.fillna(0)

#Filter out records with fewer than 500 impressions and a 0 cr_p
df1 = df1[(df1['Impressions']>500) & (df1['cr_p']>0)]

#Filter out records more than one standard deviation from the mean
df1 = df1[df1['cr_p'] <= (df1['cr_p'].mean()+df1['cr_p'].std())]

#Decision Tree Method Start
number = preprocessing.LabelEncoder()
train=df1

# Convert categorical variable to arrays and set NaN values to zero.
def convert(data):
    number = preprocessing.LabelEncoder()
    data['Date']=number.fit_transform(data.Date)
    data['LOB'] = number.fit_transform(data.LOB)
    data['Product'] = number.fit_transform(data.Product)
    data['FunnelStage']=number.fit_transform(data.FunnelStage)
    data['Platform'] = number.fit_transform(data.Platform)
    data['InventorySource']=number.fit_transform(data.InventorySource)
    data['SubTactic'] = number.fit_transform(data.SubTactic)
    data['SubConcept'] = number.fit_transform(data.SubConcept)
    data['PrimaryMessage'] = number.fit_transform(data.PrimaryMessage)
    data=data.fillna(0)
    return data
train=convert(train)

#Split the data set to train and validate
train['is_train'] = np.random.uniform(0, 1, len(train)) <= .75
train, validate = train[train['is_train']==True], train[train['is_train']==False]

#Define training variables
x_train=train[list(train.columns[2:9])]
print(x_train)
y_train=train['cr_p']
print(y_train)

x_validate=validate[list(train.columns[2:9])]
y_validate=validate['cr_p']

#Random Forest
rf=RandomForestRegressor(n_estimators=20, max_depth=3)
rf.fit(x_train,y_train)
rf.predict(x_validate)
