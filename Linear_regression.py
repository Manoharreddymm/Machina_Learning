#importing the librarys 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#importing the dataset 
df=pd.read_csv('50_Startups.csv')
df.head()

#reading the rows and columns
df.shape

df.columns

#gives the information about the data types and null values 
df.info()

#describes what kind of data it is
df.describe()

#checking for null values
df.isnull().sum()

df.duplicated().sum()

#retriving the unique values in the colums
df['State'].unique()

#plotting the relation between two columns
sns.lineplot(x=df['R&D Spend'],y=df['Administration'],hue=df['State'])

#plotting the relation between two columns
sns.scatterplot(x=df['R&D Spend'],y=df['Administration'],hue=df['State'])

#univarient analysis
import warnings
warnings.filterwarnings('ignore')
sns.distplot(x=df['R&D Spend'])

#univarient analysis
sns.distplot(x=df['Administration'])

#univarient analysis
sns.distplot(x=df['Marketing Spend'])

#bi-varient analysis
#this is plot between input and target value
plt.figure(figsize=(5,5))
sns.scatterplot(data=df,x='Marketing Spend',y='Profit')

#scatter plotting
plt.figure(figsize=(5,5))
sns.scatterplot(data=df,x='Administration',y='Profit')

plt.figure(figsize=(5,5))
sns.scatterplot(data=df,x='R&D Spend',y='Profit')

sns.pairplot(df)

#checking for outliers
plt.figure(figsize=(5,3))
sns.boxplot(x=df['R&D Spend'])

#checking for outliers
plt.figure(figsize=(5,3))
sns.boxplot(x=df['Administration'])

#checking for outliers
plt.figure(figsize=(5,3))
sns.boxplot(x=df['Marketing Spend'])

#to check the correlation between individual columns
sns.heatmap(df.drop(['State','Profit'],axis=1).corr(),annot=True)

#converting the object into numeric value
from sklearn.preprocessing import LabelEncoder
a=LabelEncoder()
df['state']=a.fit_transform(df['State'])

df=df.drop('State',axis=1)

df.info()


y=df['Profit']
x=df[['R&D Spend','Administration','Marketing Spend','state']]

#splitting the data into train and test 
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=15)

#linearregression process
from sklearn.linear_model import LinearRegression
model=LinearRegression()

x_train.shape

model.fit(x_train,y_train)

model.predict([[165349.20,136897.80,471784.10,2]])

y_pred=model.predict(x_test)

y_test

y_pred

#checking for error in for the model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test,y_pred)
mse

mae = mean_absolute_error(y_test,y_pred)
mae

rmse = np.sqrt(mse)
rmse

r2_score(y_test,y_pred)

#checking the score for train
model.score(x_train,y_train)

#checking the score for test
model.score(x_test,y_test)

#lasso for overfitting
from sklearn import linear_model
lasso_reg=linear_model.Lasso(alpha=50,max_iter=100,tol=0.1)
lasso_reg.fit(x_train,y_train)

lasso_reg.score(x_train,y_train)

lasso_reg.score(x_test,y_test)

from sklearn import linear_model
ridge_reg=linear_model.Ridge(alpha=50,max_iter=100,tol=0.1)
ridge_reg.fit(x_train,y_train)

ridge_reg.score(x_train,y_train)

ridge_reg.score(x_test,y_test)

y_pred=ridge_reg.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
mse

r2_score(y_test,y_pred)

# Plot the best fit line graph
plt.figure(figsize=(4, 4))
plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Best Fit Line')
plt.title('Actual vs. Predicted values in Multi-linear Regression')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()