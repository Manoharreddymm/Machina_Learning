
#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the csv file
df=pd.read_csv('framingham.csv')


#view the first 5 rows
df.head()


#view the columns
df.columns


#view the unique values present in the column
df['TenYearCHD'].unique()


#view the unique values present in the column
df['diabetes'].unique()


#view the information about the data
df.info()


#view the description of the data
df.describe()


#view the null values present in the data
df.isnull().sum()


df['education'].unique()


#filling the null values with 0
df['education'].fillna(0)


#view mean of column
df['education'].mean()


#view mode of column
df['education'].mode()


df.columns


#ploting the categorical target data
plt.figure(figsize=(5,3))
sns.countplot(x=df['TenYearCHD'])


import warnings
warnings.filterwarnings('ignore')


#SUBPLOT
plt.figure(figsize=(10,10))
plot = 1
for i in df:
    if plot<=9:
        ax = plt.subplot(3,3,plot)
        sns.distplot(x=df[i])
        plt.xlabel(i)
    plot+=1
plt.tight_layout()


#SUBPLOT
plt.figure(figsize=(10,10))
plot = 1
for i in df:
    if plot<=9:
        ax = plt.subplot(3,3,plot)
        sns.histplot(x=df[i],hue=df['TenYearCHD'])
        plt.xlabel(i)
    plot+=1
plt.tight_layout()


#SUBPLOT
plt.figure(figsize=(10,10))
plot = 1
for i in df:
    if plot<=9:
        ax = plt.subplot(3,3,plot)
        sns.boxplot(x=df[i])
        plt.xlabel(i)
    plot+=1
plt.tight_layout()


#view the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(),annot=True)


#fillina null values with mean
df['glucose'].fillna(df['glucose'].mean(),inplace=True)


df.isnull().sum()


df['education'].fillna(0,inplace=True)


x=df.drop('TenYearCHD',axis=1)


y=df['TenYearCHD']


#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


#predicting the y_pred
y_pred=logreg.predict(X_test)


y_pred


y_test


#score of test data
logreg.score(X_test, y_test)


#score of train data
logreg.score(X_train, y_train)


from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))


#view confussion matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
