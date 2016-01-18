# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 11:00:52 2016

@author: arcosdid
"""

from sklearn import datasets, cluster
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%%

diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test  = diabetes.data[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test  = diabetes.target[-20:]

#%%
from sklearn import linear_model
regr = linear_model.Lasso(alpha=.3)
regr.fit(diabetes_X_train, diabetes_y_train) 

regr.coef_ # very sparse coefficients
regr.score(diabetes_X_test, diabetes_y_test) 

lin = linear_model.LinearRegression()
lin.fit(diabetes_X_train, diabetes_y_train) 
lin.score(diabetes_X_test, diabetes_y_test) 

#%% CLUSTERING
#sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
k_means = cluster.KMeans(n_clusters=2)
k_means.fit(diabetes.data)
df = pd.DataFrame(diabetes.data)

#2-D plot
df.plot(x=[0], y=[2], kind='scatter', c=k_means.labels_, cmap='summer')
#df.plot(x='sepal length (cm)', y='sepal width (cm)', kind='scatter', c=k_means.labels_, cmap='summer')

#%% Dimension reduction with 2 components (Principal Component Analysis)
pca = decomposition.PCA(n_components=2)
pca.fit(diabetes.data)
X = pca.transform(diabetes.data)
df2 = pd.DataFrame(X)
df2.plot(x=[0], y=[1], kind='scatter', c=k_means.labels_, cmap='summer', colorbar=False)

# Dimension reduction with 3 components (PCA)
pca = decomposition.PCA(n_components=3)
pca.fit(diabetes.data)
X = pca.transform(diabetes.data)
df2 = pd.DataFrame(X)

#3-D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', axisbg='white')
ax.scatter(xs=df2[0], ys=df2[1], zs=df2[2], zdir=u'z', c=k_means.labels_, cmap='summer', depthshade=True)

#add labels to axes
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
#remove ticks from axes
ax.set_xticks([])                               
ax.set_yticks([])                               
ax.set_zticks([])