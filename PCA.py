#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer


cancer_dataset=load_breast_cancer()


#view the keys present in the data
cancer_dataset.keys()


#loading the data
df=pd.DataFrame(cancer_dataset['data'],columns=cancer_dataset['feature_names'])


#view the firdt 5 rows
df.head()


#scatter plot
plt.figure(figsize=(10,6))
plt.scatter(df['worst concavity'],df['worst concave points'],c=df['worst symmetry'],edgecolors='k',alpha=0.75,s=150)
plt.grid(True)
plt.show()


#heatmap for data
sns.heatmap(df.corr())


#standardscaler
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
scale.fit(df)


df.head()


#transforming the data
scale_data=scale.transform(df)


scale_data


#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
data=pca.fit_transform(scale_data)
data


pca.explained_variance_


#scatter plot
plt.figure(figsize=(8,5))
plt.scatter(scale_data[:,0],scale_data[:,1],c=cancer_dataset['target'],cmap='plasma')


#Assuming `pca` is your PCA object, not the transformed data
plt.figure(figsize=(10,6))
plt.scatter(x=[i+1 for i in range(len(pca.explained_variance_ratio_))],
            y=pca.explained_variance_ratio_,
            s=200, alpha=0.75, c='orange', edgecolor='k')
plt.grid(True)
plt.title("Explained variance ratio of the \nfitted principal component vector\n", fontsize=25)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.show()


from sklearn.decomposition import PCA
pca=PCA(n_components=None)
data=pca.fit_transform(scale_data)
data


#applying iris dataset for logistic regression
d=pd.read_csv('iris.csv')
d.head()


d= pd.get_dummies(d, columns=['Species'],drop_first=False)
d['Species_Iris-setosa'] = d['Species_Iris-setosa'].astype(int)


d['Species_Iris-versicolor'] = d['Species_Iris-versicolor'].astype(int)


d['Species_Iris-virginica'] = d['Species_Iris-virginica'].astype(int)


d.head()


x=d.drop(['Id','Species_Iris-setosa','Species_Iris-versicolor','Species_Iris-virginica'],axis=1)


x.head()


y=d[['Species_Iris-setosa','Species_Iris-versicolor','Species_Iris-virginica']]


y.head()



y= np.argmax(y, axis=1)


y


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
y_train = np.argmax(y_train, axis=1)
y_test = np.argmax(y_test, axis=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


y_test


X_train


logreg.predict([[5,3.6,1.4,0.2]])


logreg.score(X_train,y_train)


logreg.score(X_test,y_test)


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
data=pca.fit_transform(x)
data


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


logreg.score(X_train,y_train)


logreg.score(X_test,y_test)
