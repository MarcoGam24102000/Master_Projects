# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:20:48 2022

@author: marco
"""

import numpy as np
import cv2
import math
import sys
import os

import time
startTime = time.time() 


def third_goal(dir_imx, dist_redBall_camera, real_diam_red_ball):
    
    imx = cv2.imread(dir_imx)
    
    print(imx)


    def get_relation_px_mm(real_diam_red_ball, height_red_ball):
        diam_real_mm = real_diam_red_ball*10     ## convert cm to mm   
        px_per_mm = height_red_ball/diam_real_mm 
        
        print("\n" + str(px_per_mm) + " px/mm")   
    
        return px_per_mm 
    
    
    def get_angle(real_d, real_diam_red_ball):
        angle_rad = math.atan2(int(real_diam_red_ball/2),real_d)   
        
        angle_deg = angle_rad*(180/math.pi)
        
        return angle_deg
    
    
    def find_focal_length(image_real_height, angle, pix_per_cm, real_d, real_diam_red_ball):
        angle_rad = angle*(math.pi/180)
        
        focal_length = round(((153*real_d)/real_diam_red_ball)/pix_per_cm,3)       
        focal_length *= 10
        
        return focal_length        
    
    
    def rgb2gray(rgb):
    
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
        return gray
    
    
    # Find contours, obtain bounding box, extract and save ROI
     
    def extract_dims_red_ball(mask):  
        
        print("A")
    
        cnts = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     ## canny
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        dimensions_rois = []   
        
        print("B")
        
        print(cnts)
        
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
             
            
            print("C")
            
            dim = [x,y,w,h]
            print(dim) 
           
            
            if True:
                 
                    ## code to not create bounding boxes inside another already in that place
                    if abs(w-h) <= 15 or abs(w-h) >= 25:
                        
                        if h > 50:
                            w = h                         
                            
                        dimensions_rois.append([w,h])
        height_balls_list = []
                        
        for ind_d, d in enumerate(dimensions_rois):
            print("Width for image " + str(ind_d) + ": " + str(d[0]))
            print("Height for image" + str(ind_d) + ": " + str(d[1])) 
            height_balls_list.append(d[1])     
            print("D")
        
        height_red_ball = np.max(np.array([height_balls_list]))
        
        print("Height for red ball: " + str(height_red_ball))
            
        return height_red_ball
   
    print(rgb2gray(imx).astype('uint8'))
    
    print(rgb2gray(imx).astype('uint8').shape)
    
    cv2.imwrite("C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python/another_img.png", rgb2gray(imx).astype('uint8'))
                            
    height_red_ball = extract_dims_red_ball(rgb2gray(imx).astype('uint8'))    ## .astype('uint8')
    
    
    height_image = len(imx[0])
    
##    dist_redBall_camera = 40    ## 40 cm
    
##    real_diam_red_ball = 8.2    ## 8.2 cm
     
    px_per_mm = get_relation_px_mm(real_diam_red_ball, height_red_ball) 
    
    angle = get_angle(dist_redBall_camera, real_diam_red_ball)
    
    angle_shown = round(angle,2)
    
    print("Angle: " + str(angle_shown) + "º")
    
    image_real_height = (height_image/height_red_ball)*(1/px_per_mm)
    
    focal_length = find_focal_length(image_real_height, angle, 10*px_per_mm, dist_redBall_camera, real_diam_red_ball)
    
    print("Focal length: " + str(focal_length) + " mm")
    
    
    return (angle, focal_length)


def gui_third_goal(python_file_dir, angle, focal_length, fast_third):

    import PySimpleGUI as sg
    from PIL import Image
    import io
    import os
    
    image = Image.open('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/images_gui/scheme_focal_dist.png')
    image.thumbnail((400, 400))    ## (100,5000)
    bio_scheme = io.BytesIO()    
    image.save(bio_scheme, format="PNG")         
    
    
    if fast_third == False:
        
        uni_sized = (300,300)
        size_text = (50,2)
        size_window = (800, 800)
    
        layout = [
          [sg.Image(data= bio_scheme.getvalue(), key = "-FIRST_IMG-", size=uni_sized)],
          [sg.Text("Angle (º): ", justification="center", expand_x=True, expand_y=True)],
          [sg.Text("", size=(5, 5), key='ANGLE', justification="center", expand_x=True, expand_y=True)],
          [sg.Text("Focal Length (mm): ", justification="center", expand_x=True, expand_y=True)],
          [sg.Text("", size=(5, 5), key='FOCAL_LENGTH', justification="center", expand_x=True, expand_y=True)],
          [sg.Button("Get results ...")],
          [sg.Button("Back")]
          
        ]
        
        window = sg.Window("Third Goal Panel", layout, resizable=True, finalize=True, margins=(0,0))
        
        firstLoaded = False
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           if event == "Get results ...":
               window['ANGLE'].update(value=angle)
               window['FOCAL_LENGTH'].update(value=focal_length)    
               
           if event == "Back":
                           
               dir_parts = python_file_dir.split('/')

               direc_new = ''

               for ind_d, d in enumerate(dir_parts):
                   if ind_d < len(dir_parts)-1:
                       direc_new += d + '/'
                        
               base_path = direc_new         
               
               os.system('python ' + base_path + 'menu.py')
           
        window.close()
    else:
        
        uni_sized = (300,300)
        size_text = (50,2)
        size_window = (800, 800)
        
        layout = [
          [sg.Image(data= bio_scheme.getvalue(), key = "-FIRST_IMG-", size=uni_sized)],
          [sg.Text("Angle (º): ", justification="center", expand_x=True, expand_y=True)],
          [sg.Text("", size=(5, 5), key='ANGLE', justification="center", expand_x=True, expand_y=True)],
          [sg.Text("Focal Length (mm): ", justification="center", expand_x=True, expand_y=True)],
          [sg.Text("", size=(5, 5), key='FOCAL_LENGTH', justification="center", expand_x=True, expand_y=True)]          
        ]
        
        windowx = sg.Window("Third Goal Panel", layout, resizable=True, finalize=True, margins=(0,0))        
        
        windowx['ANGLE'].update(value=angle)
        windowx['FOCAL_LENGTH'].update(value=focal_length)    


numberArguments = 3

if len(sys.argv) == numberArguments + 1:
    print("Right number of parameters")
    print(sys.argv[1:])
    
    if os.path.isfile(sys.argv[1]):
        if '.jpg' in sys.argv[1]:    ## png
            print("First parameter valid")
    
            if sys.argv[2].isdigit():
                print("Second parameter valid") 
               
                if sys.argv[3].replace(".", "", 1).isdigit():
                    print("Third parameter valid")                    
                    
                    angle, focal_length = third_goal(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))  
                    
                    angle_shown = round(angle,2)
                    
                    fast_third = False
                    
                    executionTime = (time.time() - startTime)
                    print('Execution time in seconds: ' + str(executionTime))
                  
                    gui_third_goal(sys.argv[0], angle_shown, focal_length, fast_third)
                    
                    
    
else:
    print("Wrong number of arguments") 




 










