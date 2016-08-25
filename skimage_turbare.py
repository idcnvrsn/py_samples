# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 16:40:02 2016

@author: d2713
"""

from skimage import data
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform

tform = AffineTransform(scale=(1.3, 1.1), rotation=1, shear=0.7,
                        translation=(210, 50))

img = warp(data.checkerboard(), tform.inverse, output_shape=(350, 350))

coords = corner_peaks(corner_harris(img), min_distance=5)
coords_subpix = corner_subpix(img, coords, window_size=13)

import matplotlib.pyplot as plt

plt.imshow(img,cmap="gray")
plt.scatter(coords_subpix[:,1],coords_subpix[:,0])
plt.show()
