# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:55:19 2021

@author: asdg
"""

from sklearn import datasets, svm
from sklearn.kernel_approximation import Nystroem
X, y = datasets.load_digits(n_class=9, return_X_y=True)
data = X / 16.
clf = svm.LinearSVC()
feature_map_nystroem = Nystroem(gamma=.2,
                                random_state=1,
                                n_components=300)
data_transformed = feature_map_nystroem.fit_transform(data)
clf.fit(data_transformed, y)

clf.score(data_transformed, y)