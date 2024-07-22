# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:20:40 2022

@author: marco
"""

import time
startTime = time.time() 

import os
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


###################################################
###################################################
###################################################
###################################################

def second_goal(base_dir, first_img, imgFinal, fast_sec, successCounting):

    dictImages = {}
    
    dictImages['code_task'] = 2
 
    def DICE_COE(img1,img2):
             intersection = np.logical_and(img1, img2)
             union = np.logical_or(img1, img2)
             dice = (2*np.sum(intersection))/(np.sum(union)+np.sum(intersection))
             return dice
         
    if base_dir[-1] != "\\":
        base_dir += "\\"
        
    print(base_dir)
    print(first_img)
    print("----------------------------------------------")
    
    if fast_sec == False:    
    
        if os.path.isfile(base_dir + "data\\" + first_img):
        
            hsv_image = cv2.imread(base_dir + "data\\" + first_img)       
        else:            
            print("Not a file")
    else:
        
        first_img = "hsv_image_" + str(successCounting) + ".jpg"
        
        hsv_image = cv2.imread(base_dir + "data\\" + first_img)

    dictImages['Init_image'] = base_dir + "data\\" + first_img   
    
    grey_img = hsv_image[:,:,1] 
    
    cv2.imwrite("gray_hsv.png", grey_img)
    
    dictImages['OneChannelHSV_Image'] = "gray_hsv.png"
    
    plt.figure()
    vals = grey_img.mean(axis=1).flatten()
    # plot histogram with 255 bins
    b, bins, patches = plt.hist(vals, 255)   

    plt.title("Histogram for Saturation component")
    plt.xlabel("Intensity value")
    plt.ylabel("pixel count")
    plt.xlim([0,255])
    plt.savefig("after_hist.png") 
    
    dictImages['image_after_hist'] = 'after_hist.png'   
    
    image_out = np.zeros_like(grey_img)
    
    for b in range(len(grey_img[0])): 
        for a in range(len(grey_img)):
            if grey_img[a,b] > 40 :   
                image_out[a,b] = 255
            else:
                image_out[a,b] = 0
                
    cv2.imwrite("out_binarized.png", image_out)
                
    dictImages['image_after_binarization'] = "out_binarized.png"
    
    im_floodfill = image_out.copy()  
    
    h, w = image_out.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    im_out = image_out | im_floodfill_inv
    
    cv2.imwrite("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\im_out_hsv.png", im_out)
    
    dictImages['Final_Image'] = "im_out_hsv.png"  
        
        #%%    

    best_dice_coe = 0

    if fast_sec == False:      
     
        if os.path.isfile(base_dir + "data\\" + imgFinal):
            
            imToCompare = cv2.imread(base_dir + "data\\" + imgFinal)              
            imToCompare_gray = cv2.cvtColor(imToCompare, cv2.COLOR_BGR2GRAY)
            
            best_dice_coe = DICE_COE(im_out,imToCompare_gray)
        
    else:
        imgFinal = "hsv_image_1.jpg"
        
        if os.path.isfile(base_dir + "data\\" + imgFinal):
        
            imToCompare = cv2.imread(base_dir + "data\\" + imgFinal)              
            imToCompare_gray = cv2.cvtColor(imToCompare, cv2.COLOR_BGR2GRAY) 
            
            best_dice_coe = DICE_COE(im_out,imToCompare_gray)
            
        else:
            print(" ---- Not a file !!!")     
     
    dice_coe_perc = round(best_dice_coe*100, 2)
        
    print("Dice Coeff: " + str(dice_coe_perc) + " %")  
    
    return dictImages, dice_coe_perc
    

def gui_second_goal(direc_first_image, direc_final_image, dice_coe, fast_sec, dictImages):

    import PySimpleGUI as sg
    from PIL import Image
    import io
    import os
    
    if fast_sec == False:
        
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
            [sg.Text("", size=(5, 10), key='DICE')],
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
         
        window = sg.Window("Second Goal Panel", layout)
        
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
                   
                   executionTime = (time.time() - startTime)
                   print('Execution time in seconds: ' + str(executionTime))
                   
           if firstLoaded == True and event == 'Load Image':
               
               print("Hear") 
               
               if os.path.exists(direc_final_image):
                              
                   print("Hear 2")
                     
                   
                   image = Image.open(direc_final_image)           
                   image.thumbnail((400, 400))
                   
                   bio = io.BytesIO()
                   image.save(bio, format="PNG")
                   window["-IMAGE-"].update(data=bio.getvalue())
                   
                   window['DICE'].update(value=str(int(dice_coe)))   ##  + "." + str(dice_coe-int(dice_coe)) + "%"               
                   
            
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
               
                 interm_steps.gui_inter_steps(1, dictImages)
            
                   
        window.close()
    else:
        print("Continuously")
        
        
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
           
        windowx = sg.Window("Second Goal Panel", layout, finalize =True)   
    
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
    
    if(".png" in sys.argv[2]):   
        print("Second parameter valid")
        
        if(".jpg" in sys.argv[3]):    
            print("Third parameter valid")
            
            os.chdir('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python')
            
            this_dir = os.getcwd()
            foldername_string = sys.argv[1]
            dir_parts = this_dir.split(foldername_string)
            base_dir = dir_parts[0] + foldername_string + "\\"
            
            fast_sec = False
            
            dictImages, dice_coe_perc = second_goal(base_dir, sys.argv[2], sys.argv[3], fast_sec, 0)           
            direc_first_image = base_dir + "\\data\\" + sys.argv[2]   
            direc_final_image =  base_dir + "\\code_python\\"   + 'im_out_hsv.png' 
            
            gui_second_goal(direc_first_image, direc_final_image, dice_coe_perc, fast_sec, dictImages)
      
else:
    print("Wrong number of parameters")




























    
    