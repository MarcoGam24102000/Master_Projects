# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:52:40 2022

@author: marco
"""

import time
startTime = time.time() 

import sys 
import os
import numpy as np



def aux_gui():
    
    import PySimpleGUI as sg
    
    fps_value = 0
    
    layout = [ [sg.Text('FPS:')], 
              [sg.Input(default_text= "50", size=(19, 1), key="FPS")],
              [sg.Button('Next')]
             ]
    
    uni_sized = (300,300)
    size_text = (50,2)
    size_window = (800, 800)
    
    window = sg.Window("Video parameters ", layout, resizable = True, disable_close = True, finalize = True, size=size_window) 

    while True:  
        event, values = window.read() 
        if event == "Exit" or event == sg.WIN_CLOSED:      
            break
        if event == "Next":
            if int(values['FPS']) <= 10 or int(values['FPS']) >= 100:
                print("Wrong value for FPS inserted !!! \n Try again")
            else:
                
                fps = int(values['FPS'])     
                
                fps_value = fps                
                break
        
    return fps_value    
    

def write_data_to_csv_file(list_listParams, name_excel_file):   
    
    import csv  
    import pandas as pd 
    
    list_listParams = np.array([list_listParams]).T.tolist()
    
    distRedToBlueBall_list = list_listParams[0]
    dist_persp_3d_redToBlueBall = list_listParams[1]
    dist_blueBallToCamera = list_listParams[2] 
    
    
    print("distRedToBlueBall_list: ")
    print(distRedToBlueBall_list)
    
    print("dist_persp_3d_redToBlueBall: ")
    print(dist_persp_3d_redToBlueBall)    
    
    print("dist_blueBallToCamera: ")
    print(dist_blueBallToCamera) 
    
    listToWrite = []
    
    print("Going to write data to csv file")
    
    header = ['Distance from red to blue ball', 'Depth distance from red to blue ball', 'Distance from blue ball to camera']
    
    
    if len(distRedToBlueBall_list) == len(dist_persp_3d_redToBlueBall) and len(distRedToBlueBall_list) == len(dist_blueBallToCamera):
        
        for i in range(len(distRedToBlueBall_list)):
            
            if distRedToBlueBall_list[i] != 0:
            
                singleList = []
                
                singleList.append(distRedToBlueBall_list[i])
                singleList.append(dist_persp_3d_redToBlueBall[i])
                singleList.append(dist_blueBallToCamera[i])

        listToWrite.append(singleList)
    
    
        with open(name_excel_file + '.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)   
           
            writer.writerow(header)   
            
            for data in listToWrite:  
                writer.writerow(data)
    else:
        print("Dimensions for CSV file doesn´t fit each other")
  
    rows = []
    with open(name_excel_file + '.csv', 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        rows.append(row)
            
    print("Going to write to excel file ...")
            
    df = pd.DataFrame(rows)
    
    df.to_excel(name_excel_file + ".xlsx")
    
   
    
    print("Generated !!!")
    


def video_activity(python_filename, initVideoPath, finalVideoPath, dist_camera_redBall, real_diam_red_ball, name_excel_file):

    import cv2
    import imageio
    import os    

    print(imageio)
    
    
    # os.chdir('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/code_python')  
    
    # print(os.getcwd())
    
    # this_dir = os.getcwd()
    # foldername_string = "CVis_2223_Assign1_MarcoGameiro"
    # dir_parts = this_dir.split(foldername_string)
    # base_dir = dir_parts[0] + foldername_string + "\\"
    
    python_filename_parts = python_filename.split('/')
    python_new_filename = ''
    
    for ind_folder, folder in enumerate(python_filename_parts):
        if ind_folder < len(python_filename_parts)-1:
            python_new_filename += folder + '/'      
   
##    python_new_filename += 'code_python' + "/"
    
    print(python_new_filename)
    
    import sys
    
 #   sys.path.append(base_path)  
    os.chdir(python_new_filename + '/')
    
    print("CMD")
    print(os.getcwd())

    
    print("Base") 
    
    from importlib.machinery import SourceFileLoader

    dists_image = SourceFileLoader("module.name", python_new_filename + "dists_image.py").load_module() 
    find_dist = dists_image.find_dist
 ##   import code_python.dists_image
 ##   from dists_image import find_dist
##    from .dists_image import find_dist 
     
    import numpy as np 
    import glob   
    from PIL import Image, ImageDraw, ImageFont
    import time
 ##   import interm_steps.py 
 ##   from . import interm_steps
 
    interm_steps = SourceFileLoader("module2.name", python_new_filename + "interm_steps.py").load_module() 
    gui_show_inter_buffer = interm_steps.gui_show_inter_buffer
    display_video = interm_steps.display_video
    
     
##    from interm_steps import gui_show_inter_buffer, display_video 
    
 ##   C:\VisComp\PL\Project\CVis_2223_Assign1_MarcoGameiro\code_python 
    
    dictVideos = {}
    
 ##   src_dir = "C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\data\\Boccia_video.avi"
    
    src_dir = initVideoPath
    
    dictVideos['code_task'] = str(4)
    
    dictVideos['Init_video'] = src_dir
    
     
    
    src_dir = "Boccia_video.avi"
    
    if os.path.isfile(src_dir):
        reader = imageio.get_reader(src_dir)
        fps_in = reader.get_meta_data()['fps'] 
    else:
        print("Not a file")
        
    vidcap = cv2.VideoCapture(src_dir)  
    
    print(src_dir)
    
    src_dir = str(src_dir)
    
    fps_sel = aux_gui()
    
    if fps_sel == 0:
        fps_sel = 50     ## 50 by default
    
    fps_out = fps_sel         ## 0.5 sec = 25 frames
    index_in = -1
    index_out = -1   
    
    count = 0  
     
    list_img = []   
    list_gray = []
    
    successCounting = 0 
    
    ## dist_redBall_camera = 40  ## cm        
##    real_diam_red_ball = 8.2  ## cm
    
    final_img = ""
    
    dim = (500, 493)
    
    list_listParams = []
    
    dists = []
    dims_rois_video  = []
    
    center_camera = (10,10)      ## Initial random value
    
    dist_blueBallToCamera = []
    
    def drawLine(image, filename, center_BlueBall, centerCamera, thickness):
     ##   thickness = 9
        
        # Green color in BGR
        color = (0, 255, 0)
        
        image = cv2.line(image, center_BlueBall, centerCamera, color, thickness)
        
        cv2.imwrite(filename, image) 
         
    
    def find_dist_bet_centers(centerFirst, centerSec):
        first_parcel = (centerFirst[0] - centerSec[0])**2
        sec_parcel = (centerFirst[1] - centerSec[1])**2
        
        result_dist = np.sqrt(first_parcel + sec_parcel)
        
        return result_dist
    
    
    index_dists = []
    info_centers = []
    buf_rois = []
    
    python_newfilename_parts =  python_new_filename.split('/')
    python_new_filename2 = ''
    python_new_filenamex = ''
  
    
    for ind_folder, folder in enumerate(python_newfilename_parts):
        if ind_folder < len(python_newfilename_parts)-2:
            python_new_filename2 += folder + '/'   
    
             
    python_new_filenamex = python_new_filename2.replace('/', "\\")  
    print(python_new_filenamex)
    
    print("---- Filename 2:" + python_new_filename2)
    
    dist_persp = []
    
    while(True): 
        
        success, image = vidcap.read()
        
        print("Reading image " + str(successCounting))
        
        if success:     
            successCounting += 1
            if os.path.isfile(src_dir):            
                 
                list_img.append(image)             
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)            
                list_gray.append(gray)            
                dist, dims, rois_final = find_dist(gray, image)
                print("Segmentation done for image " + str(successCounting)) 
                
             ##   roi_final = np.zeros_like(image)
                
                # for res in result_rois:
                    
                #     for b in range(len(roi_final[0])):
                #         for a in range(len(roi_final)):
                #             roi_final[a,b] += res[a,b]  
                
                if len(dims) == 2:                 
                    
                    
                    firstDims = dims[0]
                    secDims = dims[1]    
                    
                    
                        
                    #####################################################################
                    ## Distances Perspective
                    
                    if firstDims[2] != secDims[2] and firstDims[3] != secDims[3]:
                        dist_p = np.sqrt((firstDims[2]- secDims[2])**2 + (firstDims[3]- secDims[3])**2)
                    elif firstDims[2] == secDims[2] and firstDims[3] != secDims[3]:
                        dist_p = abs(firstDims[3] - secDims[3])
                    elif firstDims[2] != secDims[2] and firstDims[3] == secDims[3]:
                        dist_p = abs(firstDims[2] - secDims[2])
                    else:
                        dist_p = 0
                        
                    dist_persp.append(dist_p)             
                    
                    #####################################################################
                
                    
                    w_min = 5000
                    
                    ind_blue_ball = 0
                    
                    if abs(secDims[0] - firstDims[0]) > 100:     
                        if firstDims[2] < w_min:
                            w_min = firstDims[2]
                            ind_blue_ball = 0
                        if secDims[2] < w_min:
                            w_min = secDims[2]
                            ind_blue_ball = 1                        
                        
                        dims_blue_ball = dims[ind_blue_ball] 
                        
                        x_blue_ball = dims_blue_ball[0]
                        y_blue_ball= dims_blue_ball[1]
                        w_blue_ball = dims_blue_ball[2]
                        h_blue_ball = dims_blue_ball[3] 
                        
                        center_blue_ball = (int(x_blue_ball + w_blue_ball/2), int(y_blue_ball+ h_blue_ball/2))
                        
                        dist_blue_ball = find_dist_bet_centers(center_blue_ball, center_camera)   
                        
                        info_centers.append((center_blue_ball, center_camera))
                    
                        dist_blueBallToCamera.append(dist_blue_ball)                      
                        index_dists.append(successCounting-1) 
                        
                
                        
                cv2.imwrite(python_new_filenamex + 'data\\segmented_' + str(successCounting) + '.png', rois_final)  
                    
                rois_final_rgb = cv2.imread(python_new_filenamex + 'data\\segmented_' + str(successCounting) + '.png')
                
                roi_finalrgb = cv2.bitwise_not(rois_final_rgb)
                
                buf_rois.append(roi_finalrgb)
                
                if successCounting == 1:
                    segmented_filename = "segmented_1"          ## replace above with this variable
                    
                    print("File path: " + (python_new_filenamex + "data\\" + segmented_filename + ".png"))
    
                    image_ref = cv2.imread(python_new_filenamex + "data\\" + segmented_filename + ".png")
    
    
                    center_camera = (int(len(image_ref)/2), int(len(image_ref)))      ## for the situation in  which camera is placed in the middle of the image, at the bottom     
    
                    print(center_camera)
                 
                dims_rois_video.append(dims)
                dists.append(dist)            
                
                if successCounting == 260: 
                    break
                
    print("Distances perspective")
    print(dist_persp)
    
    size = (1000,1500)
    
    # out = cv2.VideoWriter('C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\data\\Boccia_output_video_inter_example_anx.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    # print("Draining images to intermediate video ...")
    
    # for i in range(len(buf_rois)):
    #     print("Writing image " + str(i))
    #     out.write(buf_rois[i])
    #     time.sleep(1)
    # out.release()
    
    ######################################################
    ## Analyse size of blue ball along time
    
    listDimsBlueBall = [] 
    listDimsBlueBall_coord = []
    
    for dimArrayValues in dims_rois_video:
        
        if len(dimArrayValues) == 1:
            print("Blue ball not showing up")
        elif len(dimArrayValues) == 2:
            firstDims = dimArrayValues[0]
            secDims = dimArrayValues[1]
            
            if abs(secDims[0] - firstDims[0]) > 100:
                listDimsBlueBall.append((secDims[2], secDims[3]))
                listDimsBlueBall_coord.append((secDims[0], secDims[1]))
        else:
            print("Discarding this one") 
        
    
    
    ######################################################
    
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
    
    
    ######################################################
    
            
    distsArrayUnique = np.array([np.unique(np.array([dists_twoBallsPresent]))])
    
    dists_unique = []
    
    for d_coord in range(len(distsArrayUnique[0])): 
        val = distsArrayUnique[0, d_coord]
        dists_unique.append(val)    
    
    ## Get distances info in cm     
    
    size = (1000,1500)
    
    
    ##
    ## Delete left side 
    
    # for filename in glob.glob('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/data/*.png'):
        
    #     if 'segmented_' in filename:
    #         img = cv2.imread(filename)
    #         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
    #         for b in range(len(img_gray[0])):
    #             for a in range(len(img_gray)):
    #                 if b <= (1/4)*len(img_gray[0]):
    #                     if img_gray[a,b] > 30:
    #                         img_gray[a,b] = 0
            
    #   ##      cv2.imwrite(filename, img_gray)
    #         image_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    #         cv2.imwrite(filename, image_rgb)
            
            
            
    
    ##
    
    ##########################################################################################################
    
    ## For each one of the distances, in pixels, between blue ball and the camera, convert to centimeters
    ## After conversion to cm, plot output graph
    
    ####
    
##    dist_camera_redBall = 40  ## cm
    
    dim_this = dims[0]
    
    dist_camera_redBall_straightLine = abs(len(image_ref[0]) - dim_this[1])
    
    pix_per_cm = dist_camera_redBall_straightLine/dist_camera_redBall
    
    #####
    
    
    list_realDists = []
    
    ref = 153    ## 135
    
    focal_dist_list = []
    
    for ind_d, d in enumerate(dist_blueBallToCamera): 
        
        print("------------" + str(ind_d) + " ind")
        
      
        
        w_0 = 0
        w_1 = 0
        
        if len(dims) == 2:    
            
            dim_one = dims[0]
            dim_two = dims[1]
            
            w_0 = dim_one[2]
            w_1 = dim_two[2]
            
            print("First width: " + str(w_0))
            print("Second width: " + str(w_1))
        else:
            dim_part = dims[0]
            w_0 = dim_part[2]
            
            print(" --- Width: " + str(w_0))

        real_diam = 0
        
        if abs(w_0-ref) < 5:
            real_diam = w_0
        elif abs(w_1-ref) < 5:
            real_diam = w_1
            
        print("Widths: " + "(" + str(w_0) + " , " + str(w_1)  + ")")
        
     
        real_dist = d/pix_per_cm
        
        print("Real distance for indice " + str(ind_d) + " :" + str(real_dist))
        
        list_realDists.append(real_dist)
        
        real_diam = ref      ## For assurance
        
        focal_dist = ((real_diam*dist_camera_redBall)/real_diam_red_ball)/pix_per_cm      ## 135 - pixels ## width
        
        focal_dist_list.append(focal_dist)
        
        
        print("Real focal distance : " + str(focal_dist) + " cm")           
         
 ##       dist_persp 
            
    #### Insert text on image
    
    # my_image.save("ImageDraw.png")
    # font = ImageFont.load_default()
    # image_editable.text((150,150), title_text, 140, font=font)
    # my_image.save("ImageDraw.png")
     
    
    
    #########################################################################################################
    
    img_array = []
    
    inc_new = 0
    ind_seg = 0
    
    
    BlueBallToCamera = "Blue Ball -> Camera: "
    RedToBlue = "Red Ball -> Blue Ball: "
    
    interImagesToVideo = []
    
    
##    for filename in glob.glob('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/data/*.png'):
    for filename in glob.glob(python_new_filename2 + 'data/*.png'):
     ##   segmented_
     
        if 'segmented_' in filename:
            
            print("Analysing image " + str(ind_seg) + " ...")
            img = cv2.imread(filename)
            
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            for b in range(len(img_gray[0])):
                for a in range(len(img_gray)):
                    if b <= (1/4)*len(img_gray[0]):
                        if img_gray[a,b] > 30:
                            img_gray[a,b] = 0
            
      ##      cv2.imwrite(filename, img_gray)
            image_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(filename, image_rgb)
             
            img = image_rgb
            
            height, width, layers = img.shape 
            
            size = (width,height)
            
            print(size)
            
            averageBlur = cv2.blur(img, (5, 5))
            
            cv2.imwrite(filename, averageBlur)
            
            interImagesToVideo.append(averageBlur)
            
            ind_seg += 1
             
##    out = cv2.VideoWriter('C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\data\\Boccia_output_video_inter2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     

    finalVideoPath_2 = finalVideoPath.split('.')
    new_finalVideoPath = finalVideoPath_2[0]    
    interVideoPath = new_finalVideoPath + '_inter.' + finalVideoPath_2[1]    
    
    out = cv2.VideoWriter(interVideoPath,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    print("Draining images to intermediate video ...")
    
    for i in range(len(interImagesToVideo)):
        out.write(interImagesToVideo[i])
    out.release() 
    
    
    if os.path.isfile(python_new_filename2 + 'data\\Boccia_output_video_inter2.avi'):
        dictVideos['InterVideo'] = python_new_filename2 + 'data\\Boccia_output_video_inter2.avi'
    else:
        print("Intermediate video not generated !")
            
    
    ind_seg = 0
    
    distRedToBlueBall_list = []
             
            
    for filename in glob.glob(python_new_filename2 + 'data/*.png'): 
        
        
            if ind_seg < len(interImagesToVideo):
             
       
                if ind_seg in index_dists:
                    my_image = Image.open(filename)
                    
                    title_text = BlueBallToCamera + str(round(list_realDists[inc_new],2)) + " cm"         
                    
                ##    font = ImageFont.load_default()
                    font = ImageFont.truetype("arial.ttf", 20)
                    image_editable = ImageDraw.Draw(my_image)
                    center_BlueBall, centerCamera = info_centers[inc_new]
                    imx = cv2.imread(filename)
                     
                    drawLine(imx, filename, center_BlueBall, centerCamera, 20)
                    
                    image_editable.text((50,50), title_text, 140, font=font)
                    my_image.save(filename)    ## "ImageDraw.png"
                    
                    ######################################################################
                    
                    distRedToBlueBall = dists[ind_seg]/pix_per_cm
                    
                    title_anText = RedToBlue +  str(round(distRedToBlueBall,2)) + " cm"  
                    
                    font = ImageFont.truetype("arial.ttf", 20)
                    image_editable = ImageDraw.Draw(my_image)
                    image_editable.text((50,100), title_anText, 140, font=font)
                    my_image.save(filename) 
                    
                    distRedToBlueBall_list.append(distRedToBlueBall)
                    
                    ######################################################################
                    
                    
                    
                    im = cv2.imread(filename)
                     
                    cv2.imwrite("image.png", im)
                   
                    inc_new += 1
                else:
                    my_image = Image.open(filename)
                    
             ###       title_text = str(0) + " cm"   
                    
                    title_text = BlueBallToCamera + str(round(list_realDists[inc_new-1],2)) + " cm"         
                    
                ##    font = ImageFont.load_default()
                    font = ImageFont.truetype("arial.ttf", 20)
                    image_editable = ImageDraw.Draw(my_image)
                   
                    imx = cv2.imread(filename)           
                    
                    image_editable.text((50,50), title_text, 140, font=font)
                    my_image.save(filename)    ## "ImageDraw.png"
                    
                    ######################################################################
                    
                    distRedToBlueBall = dists[ind_seg-1]/pix_per_cm
                    
                    title_anText = RedToBlue +  str(round(distRedToBlueBall,2)) + " cm"   
                    
                    font = ImageFont.truetype("arial.ttf", 20)
                    image_editable = ImageDraw.Draw(my_image)
                    image_editable.text((50,100), title_anText, 140, font=font)
                    my_image.save(filename)            
                    
                    ######################################################################
                    
                    im = cv2.imread(filename)
                    
                    cv2.imwrite("image.png", im)
                    
                    distRedToBlueBall_list.append(distRedToBlueBall)
                    
                ind_seg += 1
                
                img_array.append(im)
                
                # if meas == False:
                #     img_array.append(img)
                # else:
                #     img_array.append(im)
                
    
    data_to_csv = [distRedToBlueBall_list, dist_persp, list_realDists]
    
    print("Data to CSV file: ")
    print(data_to_csv) 
    
    write_data_to_csv_file(data_to_csv, name_excel_file) 
      
            
            
    print("Cutting buffer ...")
    
    ## img_array = img_array[int((len(img_array)/3)*2):]       ## 5 - 1/3     ## 1/2
    
    new_img_array = []    
    
    lag = 25  ## 10
    
    for ind_iminarray, iminarray in enumerate(img_array):
        if ind_iminarray > lag-1:
            image_a = iminarray
            image_b = img_array[ind_iminarray-lag]  
             
            image_a = np.array([image_a])[0]
            image_b = np.array([image_b])[0]
            
            cv2.imwrite("image_a.png", image_a)
            cv2.imwrite("image_b.png", image_b) 
            
            image_a = cv2.imread("image_a.png")
            image_b = cv2.imread("image_b.png")
                                 
            image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2GRAY)
            image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2GRAY)            
            
            print("Images")
            
            print(image_a.shape)
            
            print(str(len(image_a)))
            print(str(len(image_a[0])))
            
            if len(image_a) == len(image_b) and len(image_a[0]) == len(image_b[0]):

       ##     print(image_a.shape)             
            
                for b in range(len(image_a[0])):
                    for a in range(len(image_a)):
                        if image_b[a,b] - image_a[a,b] > 100:
                             print("Unexpectable image detected ...")
                             iminarray = img_array[ind_iminarray-lag]  
                             break          
                
            
        new_img_array.append(iminarray)           
                    
                    
                    
                    # elif image_a[a,b] - image_b[a,b] > 100:
                    #      iminarray = image_a[a,b]
                        
    
    print("Defining Video Writer ...")
    
 ##   out = cv2.VideoWriter('C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\data\\Boccia_output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
     
    out = cv2.VideoWriter(finalVideoPath,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
    
    print("Draining images to video ...")
    
    # for i in range(len(img_array)):      ## Change afterwards
    #     out.write(img_array[i]) 
    # out.release() 
    
     
    for i in range(len(new_img_array)):      ## Change afterwards
        out.write(new_img_array[i]) 
    out.release()      
    
    
    dictVideos['FinalVideo'] = python_new_filename2 + 'data\\Boccia_output_video.avi'
    
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    print('Execution time, in seconds, per image: ' + str(round(executionTime/260,2)))
    
    gui_show_inter_buffer(2, dictVideos)   
    


numberArguments = 5      ## added two

if len(sys.argv) == numberArguments + 1: 
    
    dirok = 0
    
    
    if(".avi" in sys.argv[1] or ".mp4" in sys.argv[1]):
        print("First parameter valid")
        
        if(".avi" in sys.argv[2] or ".mp4" in sys.argv[2]):
            print("Second parameter valid")
            
            if os.path.isfile(sys.argv[1]):
                dirok = 1
            if dirok == 1 and os.path.isfile(sys.argv[2]):
                dirok = 2    
            if dirok == 2:                
               if sys.argv[3].isdigit() and sys.argv[4].replace('.','',1).isdigit():                   
                    real_dist = int(sys.argv[3])
                    real_diam = float(sys.argv[4])
                    video_activity(sys.argv[0], sys.argv[1], sys.argv[2], real_dist, real_diam, sys.argv[5])                   
                  
               else:
                    print("Input distance data are not numeric. PLease insert only numbers")
            else:
                print("Provided paths don´t end with existing files")
        else:
            print("Initial video filename should have one of the following extensions: \n {.avi, \t .mp4}")
    else:
        print("Initial video filename should have one of the following extensions: \n {.avi, \t .mp4}")
else:
    print("Wrong number of parameters")
 
