# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:07:17 2022

@author: marco
"""

import sys
import os

import time
startTime = time.time() 

DictImages = {}

###############################################

def first_goal(base_dir, first_img, imgFinal):
    
    
    DictImages['code_task'] = 1

    import cv2
    import os  
    import numpy as np   
    import matplotlib.pyplot as plt
    import time     
    
    #%% 
 
    if not('.jpg' in first_img):
        first_img += '.jpg'
        
    print(base_dir + "data\\" + first_img)
 
    image = cv2.imread(base_dir + "data\\" + first_img)
    
    print(os.path.isfile(base_dir + "data\\" + first_img ))
    
    print(type(image))
    
    print("Length image: (" + str(len(image)) + " , " + str(len(image[0])) + ")")
 
    imToCompare = cv2.imread(base_dir + "data\\" + imgFinal)
      
    imToCompare_gray = cv2.cvtColor(imToCompare, cv2.COLOR_BGR2GRAY) 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    plt.figure()
    vals = image.mean(axis=2).flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)    

    plt.title("Grayscale Histogram before raising on brightness")
    plt.xlabel("grayscale value")
    plt.ylabel("pixel count")
    plt.xlim([0,255])
    plt.savefig("bef_bright.png")
 
    DictImages['image_bef_bright'] = base_dir + "data\\" + first_img 
    DictImages['hist_image_bef_bright'] = "bef_bright.png"   
    
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  
    
    ################################################################
    this_dir_x = os.getcwd()
    foldername_string_x = "CVis_2223_Assign1_MarcoGameiro"
    dir_parts_x = this_dir.split(foldername_string)
    base_dir_x = dir_parts[0] + foldername_string + "\\"
    base_path_x = base_dir_x + "code_python" + "\\"    
    
    print(base_path_x)    
    
    ################################################################
        
    im_hist_bef = cv2.imread(base_path_x + "bef_bright.png")
    im_hist_after = cv2.imread(base_path_x + "after_bright.png")   
    
    #################################################################   
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2)
    
    cv2.imwrite("Threshold_image.png", thresh) 
    
    DictImages['image_output_adap_thresh'] = "Threshold_image.png" 
    
    masked = cv2.bitwise_and(image, image, mask=thresh)
    cv2.imwrite('Masked.png', masked) 
    
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)    
    
    DictImages['masked_image'] = "Masked.png"  
     
    # Find contours, obtain bounding box, extract and save ROI 
    ROI_number = 0
    thickness = 2
    color = (0, 0, 255) 
    cnts = cv2.findContours(masked_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     ## canny
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    dimensions_rois = []
    
    roi_final = np.zeros_like(image)
    
    not_include = False
    
    result_rois = []
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)        
        ROI = thresh[y:y+h,x:x+w]
        
        if w > 50 and w < 500 :            
            if h > 50 and h < 500:           
                
                if True:
                    
                    if h > 50:    
                        w = h 
                        
                    gray_roi = gray[y:y+h,x:x+w]
                    
                    white_points = 0
                    
                    for b in range(len(gray_roi[0])):
                        for a in range(len(gray_roi)):
                            if gray_roi[a,b] > 200:      ## 200
                                white_points += 1
                    
                    print("Number of white points: " + str(white_points))
                      
                    if white_points < 0.15*(len(gray_roi[0])*(len(gray_roi))):                      
                        
                        if len(dimensions_rois) > 0:
                            
                            for dim in dimensions_rois:
                                if x > dim[0] and x+w < (dim[0] + dim[2]):
                                    if y > dim[1] and y+h < (dim[1] + dim[3]):
                                        not_include = True
                             
                            if not_include == False:                       
                         
                                dimensions_rois.append([x, y, w, h])
                
                                imLines = cv2.line(image, (x,y), (x+w,y), color, thickness)
                                imLines = cv2.line(imLines, (x+w,y), (x+w,y+h), color, thickness)
                                imLines = cv2.line(imLines, (x+w,y+h), (x,y+h), color, thickness)
                                imLines = cv2.line(imLines, (x,y+h), (x,y), color, thickness)
                                
                                cv2.imwrite('BOSSIA_BALL_ROI_{}.png'.format(ROI_number), imLines) 
                                
                                dim_imLines = np.array([w, h]) 
                           
                                radius =  int((np.min(dim_imLines))/2)
                               
                                imCircle = cv2.circle(imLines, (int(x+w/2), int(y+h/2)), radius, color, thickness)                               
                                
                                # draw filled circle in white on black background as mask
                                mask_circle = np.zeros_like(image)
                                mask_circle = cv2.circle(mask_circle, (int(x+w/2), int(y+h/2)), radius, (255,255,255), -1)
                                
                                result = cv2.bitwise_and(image, mask_circle)
                                
                                result_rois.append(result)               
                       
                                ROI_number += 1
                        else:
                            dimensions_rois.append([x, y, w, h])
            
                            imLines = cv2.line(image, (x,y), (x+w,y), color, thickness)
                            imLines = cv2.line(imLines, (x+w,y), (x+w,y+h), color, thickness)
                            imLines = cv2.line(imLines, (x+w,y+h), (x,y+h), color, thickness)
                            imLines = cv2.line(imLines, (x,y+h), (x,y), color, thickness)                      
                            
                            dim_imLines = np.array([w, h]) 
                            
                            radius =  int((np.min(dim_imLines))/2)
                            
                            imCircle = cv2.circle(imLines, (int(x+w/2), int(y+h/2)), radius, color, thickness)                       
                            
                            # draw filled circle in white on black background as mask
                            mask_circle = np.zeros_like(image)
                            mask_circle = cv2.circle(mask_circle, (int(x+w/2), int(y+h/2)), radius, (255,255,255), -1)
                            
                            result = cv2.bitwise_and(image, mask_circle)
                            
                            result_rois.append(result)                           
                       
                            ROI_number += 1
                            
    shadows_image = np.zeros_like(gray)
    
    for res in result_rois:
        
        for b in range(len(roi_final[0])):
            for a in range(len(roi_final)):
                roi_final[a,b] += res[a,b]
        
    cv2.imwrite('FINAL_ROIS.png', roi_final) 
    
    coes = []   
    
    def DICE_COE(img1,img2):
         intersection = np.logical_and(img1, img2)
         union = np.logical_or(img1, img2)
         dice = (2*np.sum(intersection))/(np.sum(union)+np.sum(intersection))
         return dice   
     
        
    black_level = 117   ## 117
    
    if True:
        
        print("For black level for shadow regions of about " + str(black_level) + " ...")
    
        for b in range(0, len(gray[0])):    
            for a in range(0, len(gray)):
                if gray[a,b] < black_level:
                    shadows_image[a,b] = 255   
            
        roi_final_grey = cv2.cvtColor(roi_final, cv2.COLOR_BGR2GRAY)   
        
        for b in range(len(roi_final_grey[0])):
            for a in range(len(roi_final_grey)):
                if roi_final_grey[a,b] != 0:
                    roi_final_grey[a,b] = 255
        
        
        for b in range(len(roi_final_grey[0])):
            for a in range(len(roi_final_grey)):
                roi_final_grey[a,b] += shadows_image[a,b]
                
                    
        cv2.imwrite('FINAL_ROIS_BINARY.jpg', roi_final_grey)   

        DictImages['image_with_rois'] = "FINAL_ROIS_BINARY.jpg"        
        
            
        dice_coe = DICE_COE(roi_final_grey,imToCompare_gray)
        
        diff_img = np.zeros_like(roi_final_grey)
        
        if len(roi_final_grey) == len(imToCompare_gray) and len(roi_final_grey[0]) == len(imToCompare_gray[0]):
            for b in range(len(roi_final_grey[0])):
                for a in range(len(roi_final_grey)):
                    diff_img[a,b] = imToCompare_gray[a,b] - roi_final_grey[a,b]
                    
                    if diff_img[a,b] < 0:
                        diff_img[a,b] = 0
                   
            cv2.imwrite("im_diff.png", diff_img)        
        
        coes.append(dice_coe)     
    
    os.chdir('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python')
    
    black_level_found = coes.index(np.max(np.array([coes])))
    best_dice_coe = np.max(np.array([coes]))
    dice_coe_perc = round(best_dice_coe*100, 2)
    
    print("Dice Coeff: " + str(dice_coe_perc) + " %")  
    
    
    return dice_coe_perc,  DictImages
    
    
def gui_first_goal(direc_first_image, direc_final_image, dice_coe, fast, DictImages):

    import PySimpleGUI as sg
    from PIL import Image
    import io
    import os
    import time   

    
    if fast == False:
        
        print("Individually")
    
        image_viewer_coloumn = [
            [sg.Text("Initial image")],
            [sg.Image(key="-IMAGE_INI-", size=(300,300))],
            [sg.Button("Load Initial Image")]
        ]
        
        
        image_sec_viewer_coloumn = [
            [sg.Text("Final image")],
            [sg.Image(key="-IMAGE-", size=(300,300))],
            [sg.Button("Load Image")]        
        ]
        
        third_layout = [
            
            [sg.Text("Dice Coefficent: ")],
            [sg.Text("", size=(2, 10), key='DICE')],
            [sg.Button("Back"), sg.Button("Intermediate steps")] 
        ]
         
        
        layout = [
            [
                sg.Column(image_viewer_coloumn),
                sg.VSeparator(),
                sg.Column(image_sec_viewer_coloumn),
                sg.VSeparator(),
                sg.Column(third_layout)
            ]
        ]
     
         
        window = sg.Window("First Goal Panel", layout)
        
        firstLoaded = False
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           
           if event == 'Load Initial Image':
               if os.path.exists(direc_first_image):
                   
                   image = Image.open(direc_first_image)           
                   image.thumbnail((400, 400))
                   
                   bio = io.BytesIO()
                   image.save(bio, format="PNG")
                   window["-IMAGE_INI-"].update(data=bio.getvalue())
                   
                   firstLoaded = True
                   
           if firstLoaded == True and event == 'Load Image':
               
               print("Hear") 
               
               if os.path.exists(direc_final_image):
                              
                   print("Hear 2") 
                    
                   
                   image = Image.open(direc_final_image)            
                   image.thumbnail((400, 400))
                   
                   bio = io.BytesIO()
                   image.save(bio, format="PNG")
                   window["-IMAGE-"].update(data=bio.getvalue())
                   
                   window['DICE'].update(value=str(int(dice_coe)))   
                   
                   
                   executionTime = (time.time() - startTime)
                   print('Execution time in seconds: ' + str(executionTime))
                   
           if event == "Back":              
                 
                 this_dir = os.getcwd()
                 foldername_string = "CVis_2223_Assign1_MarcoGameiro"
                 dir_parts = this_dir.split(foldername_string)
                 base_dir = dir_parts[0] + foldername_string + "\\"
                 
                 base_path = base_dir + "code_python\\"
                 
                 os.system('python ' + base_path + 'menu.py')
                 
           if event == "Intermediate steps":
                 
                 sys.path.append('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python')
                 
                 print(os.getcwd())
    
                 import interm_steps
               
                 interm_steps.gui_inter_steps(1, DictImages)      
               
                   
        window.close()
    else:
        
        print("Continuously")
        
        if True:       
        
        
            image_viewer_coloumn = [
                [sg.Text("Initial image")],
                [sg.Image(key="-IMAGE_INI-", size=(300,300))],            
            ]
            
            
            image_sec_viewer_coloumn = [
                [sg.Text("Final image")],
                [sg.Image(key="-IMAGE-", size=(300,300))]              
            ]
            
            third_layout = [
                
                [sg.Text("Dice Coefficent: ")],
                [sg.Text("", size=(2, 10), key='DICE')]
            ]
            
            layout = [
                [
                    sg.Column(image_viewer_coloumn),
                    sg.VSeparator(),
                    sg.Column(image_sec_viewer_coloumn),
                    sg.VSeparator(),
                    sg.Column(third_layout)
                ]
            ]
            
            windowx = sg.Window("First Goal Panel", layout, finalize =True)      
      
 
        if True:        
           
           image = Image.open(direc_first_image)           
           image.thumbnail((400, 400))        
           bio = io.BytesIO()
           image.save(bio, format="PNG")
           windowx["-IMAGE_INI-"].update(data=bio.getvalue())
           
           image = Image.open(direc_final_image)            
           image.thumbnail((400, 400))        
           bio = io.BytesIO()
           image.save(bio, format="PNG")
           windowx["-IMAGE-"].update(data=bio.getvalue()) 
           
  
## Receive params cmd

numberArguments = 3

if len(sys.argv) == numberArguments + 1:
    print("Right number of paramters")
    print(sys.argv[1:])
    
    if(".jpg" in sys.argv[2] or ".jpeg" in sys.argv[2]):
        print("Second parameter valid")
        
        if(".jpg" in sys.argv[3] or ".jpeg" in sys.argv[3]):
            print("Third parameter valid")
            
            os.chdir('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python')
            
            this_dir = os.getcwd()
            foldername_string = sys.argv[1]
            dir_parts = this_dir.split(foldername_string)
            base_dir = dir_parts[0] + foldername_string + "\\"           
            
            dice_coe_perc, DictImages = first_goal(base_dir, sys.argv[2], sys.argv[3])           
            direc_first_image = base_dir + "\\data\\" + sys.argv[2]   
            direc_final_image =  base_dir + "\\code_python\\"   + 'FINAL_ROIS_BINARY.jpg'
             
            fast = False
            
            gui_first_goal(direc_first_image, direc_final_image, dice_coe_perc, fast, DictImages) 
   
else:
    print("Wrong number of parameters")
    
    
    
    
    
    
    
    
    
    
    
    
    


    
           
           
           
           
    
    




    


    
    
    
                
                
     
    