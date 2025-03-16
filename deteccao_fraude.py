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

# We see that the true positive score is terrible, probably due to unbalanced data

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm= confusion_matrix(y_test,y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
#%%
# Plotting de ROC curve and gettting its AUC
# The AUC metrics is alright despite the rest of the other metrics
y_pred_prob = lr.predict_proba(x_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_prob)
auc = metrics.roc_auc_score(y_test,y_pred_prob)

plt.rcParams['figure.figsize']=(12.,8.)
plt.plot(fpr,tpr,label="LR,auc="+str(auc))
plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--')
plt.legend(loc=4)
#%%
# Since we are dealing with unbalanced sinthetic data we shall use an oversampling technique to balance our database

from imblearn.over_sampling import  SMOTE
import numpy as np

smote = SMOTE(random_state=42)

x_resampled, y_resampled = smote.fit_resample(x, y)

df_balanced = pd.concat([x_resampled,y_resampled],axis=1)

# Checking if the database is actually balanced

df_balanced.groupby('isFraud').step.count()

#%%

#  Logistic regresssion of the balanced model

y = df_balanced['isFraud']

x = df_balanced.drop(['isFraud'],axis=1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=42)

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=300, random_state=42)

lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

#%%

#Evaluating the logist regression model with balanced data

from sklearn import metrics

print("LR Accuracy: ",metrics.accuracy_score(y_test, y_pred))
print("LR Precision: ",metrics.precision_score(y_test, y_pred))
print("LR Recall: ",metrics.recall_score(y_test, y_pred))
print("LR F1-score: ",metrics.f1_score(y_test, y_pred))

# General accuracy is pretty good, but the oother metrics are not fine

# Lets evaluate the confusion matrix

# We see that the true positive score is terrible, probably due to unbalanced data

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm= confusion_matrix(y_test,y_pred)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
#%%
# Plotting de ROC curve and gettting its AUC for the balanced model

y_pred_prob = lr.predict_proba(x_test)[::,1]
fpr,tpr,_ = metrics.roc_curve(y_test,y_pred_prob)
auc = metrics.roc_auc_score(y_test,y_pred_prob)

plt.rcParams['figure.figsize']=(12.,8.)
plt.plot(fpr,tpr,label="LR,auc="+str(auc))
plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--')
plt.legend(loc=4)

#%%
# Fast balanced database analysis using pandas_profiling
#  OBS: pandas_profiling has been renamed to ydata_profiling you may need to update your code too!
from ydata_profiling import ProfileReport

# If you want to generate the HTML file in a IDE you have to first generate the report and then create the html
# You won't have this issue if you're already using an online IDE
profile=ProfileReport(df_balanced,title="Balanced Profile Report")

#Create HTML file
profile.to_file("balanced_profile_report.html")

# We could have done the same for our unbalanced data in the begginning of the project... sorry

# type_TRANSFER and step are highly correlated with isFraud