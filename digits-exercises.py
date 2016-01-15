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
style.use('ggplot')

#%%
digits = datasets.load_digits()
digits.images.shape
pl.imshow(digits.images[0], cmap=pl.cm.gray_r) 

data = digits.images.reshape((digits.images.shape[0], -1))

gammas = np.logspace(-6, -1, 10)
svc = svm.SVC()
clf = grid_search.GridSearchCV(estimator=svc, param_grid=dict(gamma=gammas),
                   n_jobs=-1)
clf.fit(digits.data[:1000], digits.target[:1000]) 
