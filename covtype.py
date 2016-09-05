# -*- coding: utf-8 -*-
"""
Created on Sun Sep  4 21:18:30 2016
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier#ExtraTreesClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from datetime import datetime
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.cross_validation import StratifiedKFold
#from xgboost import XGBClassifier
import math
from sklearn.metrics import mean_absolute_error

import os
import time


cov = pd.read_csv("covtype.data",header=None)

#fMakeTrain = 0
fDoMode = 0

X = cov.ix[:,:53]
y = cov.ix[:,54]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.8, random_state=1234)
print('学習データの数:', len(X_train))
print('検証データの数:', len(X_test))

if __name__ == '__main__':

    #train

    print('Start train.\n')
    start = time.time()          
  
#    k = ( 1 + int(math.log(len(X_train))/math.log(2)) ) #* 4    
    if fDoMode == 0:
        param_grid = {#'bootstrap':[False],
                     #'criterion': ['entropy'],
                     #'max_depth': [19],
                     #'max_features': [100],
                     #'min_samples_split': [3],
                     'n_estimators': [100],
                     'n_jobs': [-1],
                     "verbose":[1]}
                      
        
        estimator = GridSearchCV(RandomForestClassifier(10),param_grid=param_grid,cv=StratifiedKFold(y_train, n_folds=10),n_jobs=1,verbose=1)
        #cv = StratifiedKFold(y_train, n_folds=3)
        '''                                                   

        param_grid = {"n_estimators":[100],
                      'objective':['multi:softprob'],
    #                  "max_depth": [3, None],
#                      "max_features": ['sqrt', 'None'],
    #                  "min_samples_split": [1, 3, 10],
    #                  "min_samples_leaf": [1, 3, 10],
    #                  "bootstrap": [True, False],
#                      "n_jobs": [-1],
#                      "verbose":[1]
                      "nthread":[-1]
                      }
        estimator = GridSearchCV(XGBClassifier(10),param_grid=param_grid,cv = 10,n_jobs=1,verbose=1)
        '''


        estimator.fit(X_train, y_train)
    
        elapsed_time = time.time() - start
        print( ("elapsed_time:{0}".format(elapsed_time)) + "[sec]")
    
        print('gridsearchcv best score:',estimator.best_score_)
        train_score = estimator.best_score_

#        joblib.dump(estimator,'estimator.pkl',compress=1)

    pred = estimator.best_estimator_.predict(X_test)
    print('train score',train_score)
    print('test score',accuracy_score(y_test,pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test,pred))
