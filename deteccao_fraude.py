import pandas as pd

df = pd.read_csv('fraud_dataset_example.csv')

df.info()
#%%
df.describe()
#%%
# Frauds number
df.groupby('isFraud').type.count()
 
#%%
# FlaggedFrauds number is zero!
df.groupby('isFlaggedFraud').type.count()
#%%
#dropping clients ids and isFlaggedFraud
 
df.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1,inplace=True)

#%%

# Getting dummyes from types of transfereces
df=pd.get_dummies(data=df,columns=['type'])

#%%

#  Logistic regresssion

y = df['isFraud']

x = df.drop(['isFraud'],axis=1)