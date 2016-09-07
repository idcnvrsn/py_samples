# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:10:40 2016
"""
import numpy as np
from sklearn.cross_validation import KFold

y = np.array([0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2])

print("kfold")
kf = KFold(9, n_folds=3)
print(len(kf))

for train, test in kf:
    print(train,test)

print()
for train, test in kf:
    print(y[train],y[test])
    
print()

from sklearn.cross_validation import StratifiedKFold
print("StratifiedKFold")
kf = StratifiedKFold(y,n_folds=3)
print(len(kf))
for train, test in kf:
    print(train,test)

print()
for train, test in kf:
    print(y[train],y[test])