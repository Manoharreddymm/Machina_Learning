#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reeading the data
df=pd.read_csv('penguins.csv')
df.head()

#view the columns present in the dataset
df.columns

#checking for null values
df.isnull().sum()

#view the unique values present in the datset
df['sex'].unique()

#filling null values with the mode
df['sex'].fillna(df['sex'].mode()[0],inplace=True)

df['sex'].unique()

df['sex'].replace('.','MALE',inplace=True)

df['sex'].unique()

df.isnull().sum()

#filling null values with the mean
import warnings
warnings.filterwarnings('ignore')
df['culmen_length_mm'].fillna(df['culmen_length_mm'].mean(),inplace=True)

#filling null values with the mean
df['culmen_depth_mm'].fillna(df['culmen_depth_mm'].mean(),inplace=True)

#filling null values with the mean
df['flipper_length_mm'].fillna(df['flipper_length_mm'].mean(),inplace=True)


#filling null values with the mean
df['body_mass_g'].fillna(df['body_mass_g'].mean(),inplace=True)

#gives the information about the data
df.info()

#describes the data
df.describe()

#checking the model by labelencoder
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Sex'] = le.fit_transform(df['sex'])

df=df.drop('sex',axis=1)
df.head()

#k-means
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=2)
kmeans.fit(x)

kmeans.inertia_

kmeans.cluster_centers_

labels=kmeans.labels_
labels

correct=sum(y==labels)
print("%d out of %d are correct"%(correct,y.size))

correct/float(y.size)

#using elbow method for finding clusters
cs=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=200,n_init=10)
    kmeans.fit(x)
    cs.append(kmeans.inertia_)
plt.plot(range(1,11),cs)

#k-means with 5 clusters
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=5)
kmeans.fit(x)
correct=sum(y==labels)
print("%d out of %d are correct"%(correct,y.size))
print(correct/float(y.size))

#k-means with 10 clusters
from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=10)
kmeans.fit(x)
correct=sum(y==labels)
print("%d out of %d are correct"%(correct,y.size))
print(correct/float(y.size))

