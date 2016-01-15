# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:40:11 2016

@author: arcosdid
"""

from sklearn import datasets, svm, neighbors, cluster, decomposition
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
style.use('ggplot')

#%%
iris = datasets.load_iris()
iris.data.shape
iris.target.shape
np.unique(iris.target)


#%% CLUSTERING
#sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
k_means = cluster.KMeans(n_clusters=3)
k_means.fit(iris.data)
print(k_means.labels_[::10])
print(iris.target[::10])
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#2-D plot
df.plot(x='petal length (cm)', y='petal width (cm)', kind='scatter', c=k_means.labels_, cmap='summer')
df.plot(x='sepal length (cm)', y='sepal width (cm)', kind='scatter', c=k_means.labels_, cmap='summer')

#3-D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', axisbg='white')
ax.scatter(xs=df['petal length (cm)'], ys=df['petal width (cm)'], zs=df['sepal length (cm)'], zdir=u'z', c=k_means.labels_, cmap='summer', depthshade=True)
#add labels to axes
ax.set_xlabel('petal length (cm)')
ax.set_ylabel('petal width (cm)')
ax.set_zlabel('sepal length (cm)')
#remove ticks from axes
ax.set_xticks([])                               
ax.set_yticks([])                               
ax.set_zticks([])

#%%Dimension reduction with 2 components (Principal Component Analysis)
pca = decomposition.PCA(n_components=2)
pca.fit(iris.data)
X = pca.transform(iris.data)
df2 = pd.DataFrame(X)
df2.plot(x=[0], y=[1], kind='scatter', c=k_means.labels_, cmap='summer', colorbar=False)

#%%Dimension reduction with 3 components (PCA)
pca = decomposition.PCA(n_components=3)
pca.fit(iris.data)
X = pca.transform(iris.data)
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

#%%
clf = svm.LinearSVC()
clf.fit(iris.data, iris.target) # learn from the data 
clf.predict([[ 1.0,  0.1,  1.3,  0.25]])
clf.coef_

#%%
# Create and fit a nearest-neighbor classifier
knn = neighbors.KNeighborsClassifier()
knn.fit(iris.data, iris.target) 
knn.predict([[0.1, 0.2, 0.3, 0.4]])
