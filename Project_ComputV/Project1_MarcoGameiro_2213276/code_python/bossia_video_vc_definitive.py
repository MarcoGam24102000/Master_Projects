

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 22:22:14 2022

@author: marco
"""

print("Going to perform computacional vision activity over bossia related video")

import time
startTime = time.time() 

import cv2
import imageio
import csv
import os
import sys
import re
import numpy as np

sys.path.insert(0, 'C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python')

from first_part import first_goal, gui_first_goal
from second_part import second_goal, gui_second_goal
from third_part import third_goal, gui_third_goal 


def has_numbers(inputString):
    return bool(re.search(r'\d', inputString))


def write_data_to_csv_file(list_listParams, name_excel_file):
    
    print("Going to write to csv file ...") 
    
    import csv  
    import pandas as pd 

    header = ['Dice Coeff (RGB Image)', 'Dice Coeff (HSV Image)', 'Angle', 'Focal length']
    
    with open(name_excel_file + '.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)   
       
        writer.writerow(header)   
        
        for data in list_listParams: 
            writer.writerow(data)               
  
    rows = []
    with open(name_excel_file + '.csv', 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        rows.append(row)
            
    print("Going to write to excel file ...")
            
    df = pd.DataFrame(rows)
    
    df.to_excel(name_excel_file + ".xlsx")   
    

def allFunctionsAtOnce(base_dir, base_img, final_img, successCounting, dist_redBall_camera, real_diam_red_ball):  
    
    flag_first = False
    flag_second = False
    flag_third = False    
    
    print("base_dir: " + str(base_dir))
    print("base_img: " + str(base_img))
    print("final_img: " + str(final_img))   
    
    
    paramsToSave = []
    
    ## For first      
    
    direc_first_image_bef = base_dir + "data\\" + base_img + '.tiff'
    imx = cv2.imread(direc_first_image_bef)
    
    cv2.imwrite(base_dir + "data\\" + base_img + '.jpg', imx)    
    imx_new = cv2.imread(base_dir + "data\\" + base_img + '.jpg')    
    imx_gray = cv2.cvtColor(imx_new, cv2.COLOR_BGR2GRAY)       
    cv2.imwrite(base_dir + "data\\" + "grey_" + str(successCounting) + '.jpg', imx_gray)
    
    imx_gray_clean = np.zeros_like(imx_gray)
    
    area_to_clean_counter = 0    
    area_to_clean_counter_axis = 0     
    
    for a in range(len(imx_gray)):
        for b in range(len(imx_gray[0])):
            if np.all(imx_gray[b] == 0):
                area_to_clean_counter = a
            else:
                break
        
    for b in range(len(imx_gray[0])):    
        for a in range(len(imx_gray)):
            if np.all(imx_gray[a] == 0):
                area_to_clean_counter_axis = b                
            else:
                break 
            
    if area_to_clean_counter != 0 and area_to_clean_counter_axis == 0:
        imx_gray_clean = imx_gray[area_to_clean_counter:,:]
    elif area_to_clean_counter == 0 and area_to_clean_counter_axis != 0:
        imx_gray_clean = imx_gray[:,area_to_clean_counter_axis:]
        
    cv2.imwrite(base_dir + "data\\" + "grey_after_" + str(successCounting) + '.jpg', imx_gray_clean)         
    
    direc_first_image = base_dir + "data\\" + base_img + '.jpg' 
    direc_final_image =  base_dir + "data\\"   + 'image_1.jpg'    
    
    print("Direc Final Image" + direc_final_image) 
    
    print("First Dir: " + direc_first_image)    
    print("Final Dir: " + direc_final_image)  
     
    if os.path.isfile(direc_first_image):    
        
        dice_coe_perc = first_goal(base_dir, base_img, final_img)
        
        flag_first = True       
        fast = True  
    
        gui_first_goal(direc_first_image, direc_final_image, dice_coe_perc, fast)
            
    time.sleep(1)
    
    ## For second  

    im_ref = cv2.imread(direc_first_image)    
    im_ref = cv2.cvtColor(im_ref , cv2.COLOR_BGR2HSV)    
    
    dir_first_image = base_dir + "data\\" + "hsv_image_" + str("1") + '.jpg'
    
    cv2.imwrite(dir_first_image, im_ref) 
    
    if os.path.isfile(direc_first_image): 
        print("First gone right")
    
    
    if successCounting == 1:  
        
        print("Initial HSV image")        
        
        dir_first_image = base_dir + "data\\" + "hsv_image_" + str(successCounting) + '.jpg'
        
        cv2.imwrite(dir_first_image, im_ref) 
        
        if os.path.isfile(direc_first_image): 
            print("First gone right")
        
    else:
        
        im_in = cv2.imread(direc_final_image)        
        im_in = cv2.cvtColor(im_in , cv2.COLOR_BGR2HSV)
        
        dir_final_image = base_dir + "data\\" + "hsv_image_" + str(successCounting) + '.jpg'
        
        cv2.imwrite(dir_final_image, im_in)
        
        print("HSV Image " + str(successCounting))         
        
        if os.path.isfile(dir_final_image):  
            
            fast_sec = True
            flag_second = True   
             
            dice_coef_perc = second_goal(base_dir, im_in, im_ref, fast_sec, successCounting)           
            gui_second_goal(dir_first_image, dir_final_image, dice_coef_perc, fast_sec)       
    
    time.sleep(1)   
    
    ## For third    

    fast_third = True
    
    dir_imx = base_dir + "data\\" + base_img + '.jpg'    
    
    print("Going to achieve third step for: " + dir_imx)
    
    angle, focal_length = third_goal(dir_imx, dist_redBall_camera, real_diam_red_ball)   
    
    flag_third = True
    
    gui_third_goal(angle, focal_length, fast_third)  
    
    if flag_first == True and flag_second == True and flag_third == True:  
        
        print("Valid output data. Going to next step ...")
        
        paramsToSave.append(dice_coe_perc)
        paramsToSave.append(dice_coef_perc)
        paramsToSave.append(round(angle, 2)) 
        paramsToSave.append(focal_length) 
        
        flag_first = False
        flag_second= False
        flag_third = False
        
    else:
        print(" --- Could not save data. Check for output missing parameters !!!")   
        
        
        if flag_first == False: print("Missing first one ...")
        if flag_second == False: print("Missing second one ...")
        if flag_third == False: print("Missing third one ...")        
    
    return paramsToSave 


###################### Params:  ###########################################
##  - video_name
##  - foldername_string
##  - name_image
##  - dist_redBall_camera
##  - real_diam_red_ball
##  - name_excel_file
###########################################################################

numberArguments = 6

if len(sys.argv) == numberArguments + 1:
    print("Right number of parameters")
    print(sys.argv[1:]) 
    
    python_file = sys.argv[0]
    python_file_parts= python_file.split('/')
    
    print(python_file_parts)
    
    python_file_parts = python_file_parts[:-1]
    
    this_dir = ""
    
    for part in python_file_parts:
        this_dir += part + '/'
    
    foldername_string = sys.argv[2] 
 ##   foldername_string = "CVis_2223_Assign1_MarcoGameiro" 
    dir_parts = this_dir.split(foldername_string)
    base_dir = dir_parts[0] + foldername_string + "\\"   
    
    src_dir_base = base_dir + "data\\" 

##    name_video = "Boccia_video.avi"     
    video_name = sys.argv[1] 
    
    src_dir = src_dir_base + video_name 
    
    print("Src dir: " + src_dir)
    
    ## CVis_2223_Assign1_MarcoGameiro image_ 40 8.2 data_project_VC
    
    if os.path.isfile(src_dir):     
        if ".avi" in sys.argv[1]:
            print("Valid video name")         
            
            name_image = sys.argv[3] 
            
            if name_image[-1] != '_':
                name_image = name_image + '_'
             
            dist_redBall_camera = int(sys.argv[4])        
            real_diam_red_ball = float(sys.argv[5])
            
            name_excel_file = sys.argv[6]
            
            if ".xlsx" in name_excel_file:
                
                output_string = ""
                str_list = name_excel_file.split(".xlsx")
                for element in str_list:
                    output_string += element 
                
                name_excel_file = output_string                
                
            if ".xls" in name_excel_file:
                
                output_string = ""
                str_list = name_excel_file.split(".xls")
                for element in str_list:
                    output_string += element 
                
                name_excel_file = output_string
                 
            if ".csv" in name_excel_file:
                
                output_string = ""
                str_list = name_excel_file.split(".csv")
                for element in str_list:
                    output_string += element 
                
                name_excel_file = output_string             
            
        else:
            print("Video filename must have one of the following extensions: \n { .avi }")
    else:
        print("Video file not found !!!")        
else:
    print("Wrong number of arguments")        
vidcap = cv2.VideoCapture(src_dir) 

fps_out = 50    
index_in = -1
index_out = -1    
reader = imageio.get_reader(src_dir)
fps_in = reader.get_meta_data()['fps']

count = 0  

list_img = []   

successCounting = 0

## dist_redBall_camera = 40  ## cm        
## real_diam_red_ball = 8.2  ## cm

final_img = ""

dim = (500, 493)

list_listParams = []

while(True): 
    
    success, image = vidcap.read()
    if success:     
        successCounting += 1
        if os.path.isfile(src_dir):            
            
            list_img.append(image)  
            
            if successCounting >= 2:            
                cv2.imwrite(src_dir_base + name_image + str(successCounting) + ".tiff" , image) 
                
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)               
                cv2.imwrite(src_dir_base + name_image + str(successCounting) + ".tiff" , resized)     
                
                image_fromVideo = name_image + str(successCounting)    ## name_image = "image_"
                
                paramsToSave = allFunctionsAtOnce(base_dir, image_fromVideo, final_img, successCounting, dist_redBall_camera, real_diam_red_ball)
                
                list_listParams.append(paramsToSave)   
                 
            else:    ## successCounting = 1              
            
                cv2.imwrite(src_dir_base + name_image + str(successCounting) + ".tiff" , image)            
                
                resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)                
                cv2.imwrite(src_dir_base + name_image + str(successCounting) + ".jpg" , resized)
            
                final_img = name_image + str(successCounting) + ".jpg"             
        else:
            print("Video file not found")
    else:
        print("Video capture doesn´t succeed")  
        break
    
    
write_data_to_csv_file(list_listParams, name_excel_file)


executionTime = (time.time() - startTime)
print('Total execution time in seconds: ' + str(executionTime))
        
