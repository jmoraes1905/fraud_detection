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

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=300, random_state=42)

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

#%%

#Evaluating the logist regression model

from sklearn import metrics

print("LR Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("LR Precision: ",metrics.precision_score(y_test, y_pred))
print("LR Recall: ",metrics.recall_score(y_test, y_pred))
print("LR F1-score: ",metrics.f1_score(y_test, y_pred))

# General accuracy is pretty good, but the oother metrics are not fine

# Lets evaluate the confusion matrix

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm= confusion_matrix(y_test,y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()

# We see that the true positive score is terrible, probably due to unbalanced data