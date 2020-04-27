# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:58:27 2017

"""
import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.9,0.8,0.55,0.45,0.3])
n = np.array([0.7,0.4,0.34,0.2,0.1])


tpr=[]
fpr=[]
for th in [0.1*x for x in range(10,1,-1)]:
#    print(th)
    tpr.append(np.count_nonzero(p >= th) / p.shape[0])
    fpr.append(np.count_nonzero(n > th)  / n.shape[0])

print(tpr)
print(fpr)

plt.plot(fpr, tpr, color='darkorange')
#         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.xlim([0.0, 1.0])
plt.ylim([-0.05, 1.05])
plt.show()