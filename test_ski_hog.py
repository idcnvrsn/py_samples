# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 15:12:14 2016

@author: d2713
"""
from skimage import io
from skimage.feature import hog
import matplotlib.pyplot as plt

image = io.imread("crop.png",0)

orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (3, 3)

import skimage.color
gimg = skimage.color.rgb2gray(image)
gimg = skimage.img_as_ubyte(gimg)

h = hog(gimg, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

plt.imshow(gimg,cmap="gray")
plt.show()
