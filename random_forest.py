import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits 


digits=load_digits()
digits


df=pd.DataFrame(digits.data)



df['target']=digits.target


df.head()


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df.drop(['target'],axis='columns'),digits.target,test_size=0.2)


len(x_train)


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)


y_pred=model.predict(x_test)


y_pred


y_test


from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm


%matplotlib inline
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
