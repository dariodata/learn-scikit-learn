# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:40:11 2016

@author: arcosdid
"""

from sklearn import datasets, svm, neighbors, cluster, decomposition, grid_search
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D
#style.use('ggplot')

#%%

digits = datasets.load_digits()
digits.images.shape

pl.imshow(digits.images[5], cmap=pl.cm.gray_r)

#%% Support Vector Classifier
svc = svm.SVC(kernel = 'linear')
#svc.fit(digits.data, digits.target)

perm = np.random.permutation(digits.target.size)
digits.data = digits.data[perm]
digits.target = digits.target[perm]

svc.fit(digits.data[:1600], digits.target[:1600]) 

svc.score(digits.data[1600:], digits.target[1600:])

#%%
data = digits.images.reshape((digits.images.shape[0], -1))

gammas = np.logspace(-6, -1, 10)
svc = svm.SVC()
clf = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),
                   n_jobs=-1)
clf.fit(digits.data[:1000], digits.target[:1000]) 


#%%
from scipy import misc
lena = misc.lena().astype(np.float32)
X = lena.reshape((-1, 1)) # We need an (n_sample, n_feature) array
k_means = cluster.KMeans(n_clusters=5)
k_means.fit(X) 

values = k_means.cluster_centers_.squeeze()
labels = k_means.labels_
lena_compressed = np.choose(labels, values)
lena_compressed.shape = lena.shape
