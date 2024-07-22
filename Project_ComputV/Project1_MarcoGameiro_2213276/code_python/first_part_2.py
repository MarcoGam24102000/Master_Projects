# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:20:06 2022

@author: marco
"""

#%%

import cv2 
import os
import numpy as np
import scipy.ndimage as ndi
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.morphology import watershed
# from skimage.morphology import morphology 

#%%

this_dir = os.getcwd()
foldername_string = "CVis_2223_Assign1_MarcoGameiro"
dir_parts = this_dir.split(foldername_string)
base_dir = dir_parts[0] + foldername_string + "\\"

#%%
 
bocia_ball_image = cv2.imread(base_dir + "\\data\\Boccia_balls.jpg")
bocia_grey = cv2.cvtColor(bocia_ball_image, cv2.COLOR_BGR2GRAY)

#%%
 
lim1 = 100
lim2 = 150

elevation_map = sobel(bocia_grey)
markers = np.zeros_like(bocia_grey)

markers[bocia_grey < lim1] = 1
markers[bocia_grey > lim2] = 2

## 
segmentation = watershed(elevation_map, markers)
segmentation = ndi.binary_fill_holes(segmentation-1)
labeled_regions, _ = ndi.label(segmentation)
image_label_overlay = label2rgb(labeled_regions, image = bocia_grey)

image_overlay = image_label_overlay[:,:,0]

mask_image = np.zeros((len(image_overlay), len(image_overlay[0])))

decis = 0.5

x_coords = []
y_coords = []

for j in range(len(image_overlay[0])):
    for i in range(len(image_overlay)):
        if image_overlay[i,j] < decis:
            image_overlay[i,j] = 255
            x_coords.append(i)
            y_coords.append(j)
            
        else:
            image_overlay[i,j] = 0

x_min = np.min(np.array([x_coords]))
x_max = np.max(np.array([x_coords]))

y_min = np.min(np.array([y_coords]))
y_max = np.max(np.array([y_coords]))

roi = bocia_grey[x_min:x_max, y_min:y_max]

cv2.imwrite("first_method.jpg", roi)



            



  

















