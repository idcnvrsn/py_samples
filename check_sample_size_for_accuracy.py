

from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from numpy import random

# Loading some example data
digits = datasets.load_digits()
Xr = digits.data
yr =digits.target

#np.random.seed(100)
tempX,dum1,tempy,dum2 = train_test_split(Xr,yr, test_size=0.0,random_state=1234)
X_A,X_B,y_A,y_B = train_test_split(tempX,tempy, test_size=0.5,random_state=1234)

for ratio in [0.1,0.2,0.4,0.6,0.8,1.0]:
    X = X_A[:int(len(X_A)*1.0)]
    y = y_A[:int(len(y_A)*1.0)]
    
    X_test = X_B[:int(len(X_B)*ratio)]
    y_test = y_B[:int(len(y_B)*ratio)]
        
    # Training classifiers
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=7)
    clf3 = SVC(kernel='rbf', probability=True)
    
    clf1.fit(X, y)
    pred = clf1.predict(X_test)
    print(accuracy_score(pred,y_test))
    clf2.fit(X, y)
    pred = clf2.predict(X_test)
    print(accuracy_score(pred,y_test))
    clf3.fit(X, y)
    pred = clf3.predict(X_test)
    print(accuracy_score(pred,y_test))
    print("")


            
            

