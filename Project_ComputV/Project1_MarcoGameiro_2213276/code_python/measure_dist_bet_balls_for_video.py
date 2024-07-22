# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:52:40 2022

@author: marco
"""

import cv2
import imageio
import os
from measure_distance_bet_balls_image import find_dist
import numpy as np

src_dir = "C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\data\\Boccia_video.avi"

vidcap = cv2.VideoCapture(src_dir) 

fps_out = 50    
index_in = -1
index_out = -1    
reader = imageio.get_reader(src_dir)
fps_in = reader.get_meta_data()['fps'] 

count = 0  

list_img = []   
list_gray = []

successCounting = 0

## dist_redBall_camera = 40  ## cm        
## real_diam_red_ball = 8.2  ## cm

final_img = ""

dim = (500, 493)

list_listParams = []

dists = []

while(True): 
    
    success, image = vidcap.read()
    if success:     
        successCounting += 1
        if os.path.isfile(src_dir):            
             
            list_img.append(image)             
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            
            list_gray.append(gray)            
            dist = find_dist(gray, image)            
            dists.append(dist)            
            
            if successCounting == 260:
                break

dists_twoBallsPresent = []


for d in dists:
    if d>0:
        dists_twoBallsPresent.append(d)
        
######################################################
## Get info about the number of boccia balls in the image

info_number_balls = []

for d in dists:
    if d == 0:
        info = "Only 1 ball found"
    elif d>0:
        info = "2 boccia balls found"
    info_number_balls.append(info) 


######################################################
        
distsArrayUnique = np.array([np.unique(np.array([dists_twoBallsPresent]))])

dists_unique = []

for d_coord in range(len(distsArrayUnique[0])): 
    val = distsArrayUnique[0, d_coord]
    dists_unique.append(val)
    

## Get distances info in cm 
    

    
        
        






