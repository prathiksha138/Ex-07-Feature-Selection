# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
from sklearn.datasets import load_boston

boston_data=load_boston()

import pandas as pd

boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

boston['MEDV'] = boston_data.target

dummies = pd.get_dummies(boston.RAD)

boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)

X = boston.drop(columns='MEDV')

y = boston.MEDV

boston.head(10)

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import KFold

from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LinearRegression

from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False) classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))

print("R_squared: " + str(round(r2_score(y,y_pred),2)))

boston.var()

X = X.drop(columns = ['NOX','CHAS'])

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))

print("R_squared: " + str(round(r2_score(y,y_pred),2)))

Filter Features by Correlation
import seaborn as sn import matplotlib.pyplot as plt

fig_dims = (12, 8)

fig, ax = plt.subplots(figsize=fig_dims)

sn.heatmap(boston.corr(), ax=ax)

plt.show()

abs(boston.corr()["MEDV"])

abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()

vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

for val in vals:

features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()

X = boston.drop(columns='MEDV')

X=X[features]

print(features)

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))

print("R_squared: " + str(round(r2_score(y,y_pred),2)))
Feature Selection Using a Wrapper
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)

boston['MEDV'] = boston_data.target

boston['RAD'] = boston['RAD'].astype('category')

dummies = pd.get_dummies(boston.RAD)

boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)

X = boston.drop(columns='MEDV')

y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline,

       k_features=1, 
       
       forward=False, 
       
       scoring='neg_mean_squared_error',
       
       cv=cv)
X = boston.drop(columns='MEDV')

sfs1.fit(X,y)

sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]

y = boston['MEDV']

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))

print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]

y = boston['MEDV']

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))

print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']] y = boston['MEDV']

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))

print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]

y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)

print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))

print("R_squared: " + str(round(r2_score(y,y_pred),3)))

# OUPUT
![pics1](https://user-images.githubusercontent.com/94169318/170408106-fc70065f-c83f-49e2-9b19-a1121f3a0080.jpeg)
![pics2](https://user-images.githubusercontent.com/94169318/170408132-911b0b71-462b-40a0-b5aa-3f3e880508fa.jpeg)
![pics3](https://user-images.githubusercontent.com/94169318/170408358-47134aa9-c0ea-4cbb-8140-b9c6acf3ab08.jpeg)
![pics4](https://user-images.githubusercontent.com/94169318/170408365-631cee50-4996-423f-bab3-f0cbf83fdc6f.jpeg)
![pics5](https://user-images.githubusercontent.com/94169318/170408373-e3f756a8-be51-4b0e-abb5-fd92832deedf.jpeg)
![pics6](https://user-images.githubusercontent.com/94169318/170408382-ff66246d-1f07-4686-9ddc-3524d819a0f2.jpeg)
![pics7](https://user-images.githubusercontent.com/94169318/170408390-1c624742-cc32-4197-8966-744738a387e9.jpeg)
## FILTER METHODS:
![pics8](https://user-images.githubusercontent.com/94169318/170408400-e198a097-61e0-4985-8553-fa659ab618db.jpeg)![pics10](https://user-images.githubusercontent.com/94169318/170408528-d4eeb3d1-0a69-4c3f-99a3-010323fbbb56.jpeg)

## 2.Information gain/Mutual Information:
![pics9](https://user-images.githubusercontent.com/94169318/170408411-957c8662-afdc-408f-9337-28ac9588c3e7.jpeg)

## 3.SelectKBest Model:
![pics10](https://user-images.githubusercontent.com/94169318/170408535-c930faed-3096-41f8-af68-840d70e48023.jpeg)

## 5.Mean Absolute Difference:
![pics11](https://user-images.githubusercontent.com/94169318/170408545-a3ea44e1-1681-43dd-9e1f-3042ef84f957.jpeg)

![pics12](https://user-images.githubusercontent.com/94169318/170408555-3add27fe-4b07-49f9-b83b-73b560b3a422.jpeg)

## 6.Chi Square Test:
![pics13](https://user-images.githubusercontent.com/94169318/170408562-3d71867f-7339-4139-9ad7-b02dcad6f94b.jpeg)

## 7.SelectPercentile method
![pics14](https://user-images.githubusercontent.com/94169318/170408569-5346ba02-85a6-4c82-bd5c-7b707b9d05bf.jpeg)

# WRAPPER METHOD

## 1.Forward feature selection:
![pics15](https://user-images.githubusercontent.com/94169318/170408580-1ab01083-c7d4-4118-bd8f-4f4b3d4c7b4f.jpeg)

## 2.Backward feature elimination:
![pics16](https://user-images.githubusercontent.com/94169318/170408592-b94d8ce4-fe9c-4f84-9527-dcae68239a63.jpeg)

## 3.Bi-directional elimination
![pics17](https://user-images.githubusercontent.com/94169318/170408601-ba67fafe-e3c7-4519-86b8-7b26ab16845f.jpeg)

## 4.Recursive Feature Selection
![pics18](https://user-images.githubusercontent.com/94169318/170408624-0e9c5e31-6271-4637-a4d1-aafb6561a1b6.jpeg)

## EMBEDDED METHOD

## 1.Random Forest Importance:
![pics19](https://user-images.githubusercontent.com/94169318/170408653-23600260-3caa-4c84-9371-66075fcea96f.jpeg)


## RESULT:
Hence various feature selection techniques are applied to the given data set successfully and saved the data into a file.
