#importing the required libraries
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()

iris.feature_names

#importinh the dataframe
import pandas as pd
df=pd.DataFrame(iris.data,columns=iris.feature_names)

#viewing the first 5 rows
df.head()

df['target']=iris.target

df[df.target==0].head()

df['flower_name']=df.target.apply(lambda x:iris.target_names[x])

from matplotlib import pyplot as plt
%matplotlib inline

df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]

plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],marker='+',color='blue')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='green')

from sklearn.model_selection import train_test_split
y=df.target
x=df.drop(['target','flower_name'],axis='columns')

from sklearn.svm import SVC
model=SVC()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

y_pred

y_test

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


import seaborn as sns
sns.heatmap(df.corr())

