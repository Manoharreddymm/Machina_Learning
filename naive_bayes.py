#importing the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#reading the dataset
df=pd.read_csv('iris.csv')
df.head()

#view the unique values present in the column
df['Species'].unique()

#splotting the data
from sklearn.model_selection import train_test_split
x=df.drop(['Species','Id'],axis=1)
y=df['Species']
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=14)

#Naivebayes
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)

#predicting the output with x_test
y_pred=nb.predict(X_test)

#score for train data
nb.score(X_train,y_train)

#score for test data
nb.score(X_test,y_test)

#predicting the output for new data
nb.predict([[5.1,3.2,1.3,0.2]])

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)