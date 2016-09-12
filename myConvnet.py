# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 10:21:14 2016
"""
import chainer
import chainer.functions as F
import chainer.links as L

from sklearn import datasets
import matplotlib.pyplot as plt

image = datasets.load_sample_image("flower.jpg")

conv1=L.Convolution2D(3,  96, 11, stride=4)

conv1(image)

plt.imshow(image)

