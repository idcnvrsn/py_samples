# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:21:14 2016
"""
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import math

from sklearn import datasets
import matplotlib.pyplot as plt
#from skimage import data

from scipy.misc import imresize

#chelsea = data.chelsea()
#plt.imshow(chelsea)

image = datasets.load_sample_image("flower.jpg")
image = imresize(image,(400,300))
image = image.astype("float32") / 255.0

plt.imshow(image)


#image = image[:,:,:,np.newaxis]
image = np.transpose(image,(2,0,1))
image = image[np.newaxis,:,:,:]
print(image.shape)

#conv1=L.Convolution2D(3,  96, 11)#, stride=)
w = math.sqrt(2)  # MSRA scaling
conv1=L.MLPConvolution2D(3, (96, 96, 96), 11, stride=4, wscale=w)

conv1(image)


