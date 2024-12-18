#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#reading the csv file
df=pd.read_csv('iris.csv')


#view the first 5 rows
df.head()


#gives the information of the data
df.info()


#checks for null values in the data
df.isnull().sum()


#describes the data
df.describe()


#plotting the boxplot for checking the outliers
plt.figure(figsize=(5,5))
sns.boxplot(df)


df=df.drop('Id',axis=1)


#view the columns in the dataset
df.columns


#drops the un-required columns
x=df.drop('Species',axis=1)


y=df['Species']


df.head()


#converting object into numeric data
df= pd.get_dummies(df, columns = ['Species'])
df.head()


df['setosa'] = df['Species_Iris-setosa'].astype(int)
df['versicolor'] = df['Species_Iris-versicolor'].astype(int)
df['virginica'] = df['Species_Iris-virginica'].astype(int)


df.tail()
df=df.drop(['Species_Iris-setosa','Species_Iris-versicolor','Species_Iris-virginica'],axis=1)
df.head()


x=df.drop(['setosa','versicolor','virginica'],axis=1)


x.head()


y=df[['setosa','versicolor','virginica']]

y.head()


#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


#Decision tree
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)


#predicting the output for x_test
y_pred=dt.predict(X_test)


print(y_pred)


y_test.head()


dt.score(X_train,y_train)


dt.score(X_test,y_test)


#this is for the entropy
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(X_train,y_train)


y_pred=dt.predict(X_test)


print(y_pred[:14])


y_test.head()


dt.score(X_train,y_train)


dt.score(X_test,y_test)


#displaying the accuracy 
from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion  index: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


from sklearn.metrics import accuracy_score
print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#plotting the tree
from sklearn import tree
plt.figure(figsize=(12,8))  # Set the size of the plot
tree.plot_tree(dt, filled=True)
plt.show()


#Hyperparameters
import warnings
warnings.filterwarnings('ignore')
grid=GridSearchCV(dt,param_grid=parm_dict,cv=10,verbose=1,n_jobs=-1)
grid.fit(X_train,y_train)


#displays the which parameters for getting high accuracy
grid.best_params_


grid.best_estimator_


grid.best_score_
