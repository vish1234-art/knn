import pandas as pd
import numpy as np

dataset=pd.read_csv("data.csv")

X=dataset.iloc[:,:-1].values
print("X:")
print(X)



y=dataset.iloc[:,2].values
print("y:")
print(y)
from sklearn.neighbors import KNeighborsClassifier


classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X,y)
X_test=np.array([6,6])
y_pred=classifier.predict([X_test])
print("General KNN:")
print(y_pred)

classifier=KNeighborsClassifier(n_neighbors=3,weights='distance')
classifier.fit(X,y)
X_test=np.array([6,6])
y_pred1=classifier.predict([X_test])
print("Distance weighted KNN:")
print(y_pred1)
