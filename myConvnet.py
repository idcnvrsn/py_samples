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
image = imresize(image,(300,400))
image = image.astype("float32") / 255.0

plt.imshow(image)

print( int(400/4 ))
print( int(300/4 ) )

print(98*4)
print(73*4)

#image = image[:,:,:,np.newaxis]
image = np.transpose(image,(2,0,1))
image = image[np.newaxis,:,:,:]
print("input image:",image.shape)

#conv1=L.Convolution2D(3,  96, 11)#, stride=)
w = math.sqrt(2)  # MSRA scaling
conv1=L.MLPConvolution2D(3, (96, 96, 96), 11, stride=4, wscale=w)

mlpconv1 = conv1(image)
print("conv result1:",mlpconv1.data.shape)

#rev_image = np.transpose(cv1.data,(1,2,3,0))

#plt.imshow(rev_image[0])
type(mlpconv1)

print(mlpconv1.data[0].shape)
print("mlpconv1 is")
plt.imshow(mlpconv1.data[0][0])

#n1, n2, h, w = model.conv1.W.shape
fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(96):
    ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
#    ax.imshow(model.conv1.W[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.imshow(mlpconv1.data[0][i])#, cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

relu1 = F.relu(mlpconv1)
print("relu result1:",relu1.data.shape)

mp1 = F.max_pooling_2d(relu1, 3, stride=2)
print("max_pooling result1:",mp1.data.shape)

dr1 = F.dropout(mp1)
print("dropout result1:",dr1.data.shape)

mlpconv4=L.MLPConvolution2D(96, (1024, 1024, 1000), 3, pad=1, wscale=w)
h = mlpconv4(dr1)
print("mlpconv4 result:",h.data.shape)

ap1 = F.average_pooling_2d(h, 6)
print("ap1 result:",ap1.data.shape)
h = F.reshape(ap1, (1, 1000))

#self.mlpconv4(h, train=self.train))

