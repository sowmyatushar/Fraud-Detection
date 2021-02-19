# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import cohen_kappa_score

data1=pd.read_csv("C:\\Users\\tussh\\Documents\\Project\\Credit Card Fraud\\creditcard.csv")

data1.shape
data1.columns
data1.describe()

##Checking null value..
data1.isnull().sum()

# lets find the distribution of class variable...##
data1["Class"].value_counts()

######.. taking the sample of data because if we take the entire value computational time will be very high...######################
data=data1.sample(frac=0.3,random_state=1)
data.shape
data["Class"].value_counts()
### findout the fraud and regular transaction...###
fraud=data[data['Class']==1]
regular=data[data['Class']==0]

print(fraud.shape,regular.shape)

#create count plot for Class
#%matplotlib inline
sns.set(style="darkgrid")
plt.title("Transaction Class Distribution")
ax = sns.countplot(x='Class',data=data,color="green" )
plt.show()

## fraction of outlier
frac_out=len(fraud)/float(len(regular))
print(frac_out)

## correlation matrix....##
cormat=data.corr()

fig= plt.figure(figsize=(10,7))
sns.heatmap(cormat,annot=True,cmap="YlGnBu")
plt.show()


X=data.iloc[:,0:30]
X.shape

y=data.iloc[:,-1]
y.head()
y.shape

## Isolation Forest Algorithm....#####
from sklearn.ensemble import IsolationForest

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

isof=IsolationForest(n_estimators=100,max_samples=len(X_train),contamination=frac_out)

model=isof.fit(X_train,y_train)
y_pred=model.predict(X_test)

## aS the output of isolation forest is in -1 & 1 we need convert it out target variable..##
y_pred[y_pred==1]=0
y_pred[y_pred==-1]=1

np.mean(y_test==y_pred)*100
print(classification_report(y_test,y_pred))

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=64..##

cohen_kappa_score(y_test,y_pred)
# kappa score 0.3457

## Logistic regression Algorithm....####

from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

model=logreg.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=31..##

np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
# kappa score 0.6259

#### Extreme Gradient Boosting Algo.....######

import xgboost as xgb 

XGBC=xgb.XGBClassifier() 

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

model=XGBC.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=12..##
np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
# kappa score is .8418

###### CAT boost algorithm.....####

from catboost import CatBoostClassifier

cb=CatBoostClassifier()

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.30, random_state=42)

model=cb.fit(X_train,y_train)

y_pred=model.predict(X_test)

n_errors=(y_pred!=y_test).sum()
print(n_errors)
## Number of errors=7..##

np.mean(y_test==y_pred)*100

print(classification_report(y_test,y_pred))

cohen_kappa_score(y_test,y_pred)
## kappa score=0.9039

#kappa score higher the better......

## ....So from the kappa Score and number of predicting errors, my finalised model is CatBoostClassifier...###
