# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:29:33 2022

@author: marco
"""

import PySimpleGUI as sg

def display_video(video_filename):
    
    import cv2
    import numpy as np
    
    if '.mp4' in video_filename or '.avi' in video_filename:
     
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
        cap = cv2.VideoCapture(video_filename)
         
        # Check if camera opened successfully
        if (cap.isOpened()== False): 
          print("Error opening video stream or file")
         
        # Read until video is completed
        while(cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = cap.read()
          if ret == True:
         
            # Display the resulting frame
            cv2.imshow('Frame',frame)
         
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
              break
         
          # Break the loop
          else: 
            break
         
        # When everything done, release the video capture object
        cap.release()
         
        # Closes all the frames
        cv2.destroyAllWindows()
    else:
        print("Video filename is not valid !")
        

def gui_show_inter_buffer(code_data, dictImages): 
    
    import os    
    from PIL import Image
    import io
    
    if code_data == 1:
        add_str = "Images"
    elif code_data == 2:
        add_str = "Videos"
    
    
    print("Intermdiate " + add_str)
    
    tasks_number = int(dictImages['code_task'])
    
    if tasks_number == 1:        
        print("First task")
        
        ind_checked = 0
        
        # image_bef_bright = dictImages['Compute histogram'] ##
        
        # if os.path.isfile(image_bef_bright):     ## to others also
        #     print("Checked")           
            
        #     ind_checked = 1
            
        #     image = Image.open(image_bef_bright)
        #     image.thumbnail((400, 25000))    ## (100,5000)
        #     bio_first = io.BytesIO()    
        #     image.save(bio_first, format="PNG")  
        
        
        # print("First task")
        
        # ind_checked = 0
        
        image_bef_bright = dictImages['image_bef_bright']
        
        if os.path.isfile(image_bef_bright):     ## to others also
            print("Checked")           
            
            ind_checked = 1
            
            image = Image.open(image_bef_bright)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_first = io.BytesIO()    
            image.save(bio_first, format="PNG")  
            
            
            ## bio.getvalue()    
            
        
            
            ## bio.getvalue()            
        
        # image_after_binarization = dictImages['image_after_binarization']
        
        # if os.path.isdir(image_after_binarization):
        #     print("Checked")
            
        #     ind_checked = 3

        #     image = Image.open(image_after_binarization)
        #     image.thumbnail((400, 25000))    ## (100,5000)
        #     bio_third = io.BytesIO()    
        #     image.save(bio_third, format="PNG")    
            
            ## bio.getvalue()            
            
        image_output_adap_thresh = dictImages['image_output_adap_thresh']
        
        if os.path.isfile(image_output_adap_thresh):
            print("Checked")
            
            ind_checked = 4
            
            image = Image.open(image_output_adap_thresh)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_fourth = io.BytesIO()    
            image.save(bio_fourth, format="PNG")    
            
            ## bio.getvalue() 
            
        masked_image = dictImages['masked_image']
        
        if os.path.isfile(masked_image):
            print("Checked")  
             
            ind_checked = 5
            
            image = Image.open(masked_image)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_fifth = io.BytesIO()    
            image.save(bio_fifth, format="PNG")     
            
            ## bio.getvalue()
            
        image_with_rois = dictImages['image_with_rois']
        
        if os.path.isfile(image_with_rois):
            print("Checked") 

            ind_checked = 6            
            
            image = Image.open(image_with_rois)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_sixth = io.BytesIO()    
            image.save(bio_sixth, format="PNG")     
            
            ## bio.getvalue()     
        
                
        if ind_checked == 6:
            
            uni_sized = (300,300)
            size_text = (50,2)
            size_window = (800, 800)
            
            layout = [[sg.Image(data= bio_first.getvalue(), key = "-FIRST_IMG-", size=uni_sized), sg.Image(data= bio_fourth.getvalue(), key = "-FOURTH_IMG-", size=uni_sized)],    ##  sg.Image(data= bio_sec.getvalue(), key = "-SEC_IMG-", size=uni_sized)]
                      [sg.Text("Image without brightness increase", size=size_text), sg.Text("   Image output of adaptable threshold", size=size_text)],    ## , sg.Text("Histogram of image without brightness increase", size=size_text)
                  ##    [sg.Image(data= bio_fourth.getvalue(), key = "-FOURTH_IMG-", size=uni_sized)],    ## sg.Image(data= bio_third.getvalue(), key = "-THIRD_IMG-", size=uni_sized), 
                  ##    [sg.Text("Image output of adaptable threshold", size=size_text)],    ## sg.Text("Image after binarizaton", size=uni_sized), 
                      [sg.Image(data= bio_fifth.getvalue(), key = "-FIFTH_IMG-", size=uni_sized), sg.Image(data= bio_sixth.getvalue(), key = "-SIXTH_IMG-", size=uni_sized)],
                      [sg.Text("Masked image", size=size_text), sg.Text(" \t Image with ROI's", size=size_text)],
                      [sg.Button("Back")]]
            
            window = sg.Window("Intermdiate " + add_str, layout, resizable = True, disable_close = True, finalize = True, size=size_window) 
           
            while True:  
                event, values = window.read() 
                if event == "Exit" or event == sg.WIN_CLOSED:      
                    break
                if event == "Back":
                    break
            
            window.close()
            
        else:
            print("At least one path to image is wrong !")
            
        
    elif tasks_number == 2:              ## Do for this
        print("Second task")
        
        init_image = dictImages['Init_image'] 
        
        if os.path.isfile(init_image):
            ind_checked = 1
            
            image = Image.open(init_image)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_first = io.BytesIO()    
            image.save(bio_first, format="PNG")    
        
        one_channelImage = dictImages['OneChannelHSV_Image']
        
        if os.path.isfile(one_channelImage):
            ind_checked = 2
            
            image = Image.open(one_channelImage)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_sec = io.BytesIO()    
            image.save(bio_sec, format="PNG")    
      
        image_after_binarization = dictImages['image_after_binarization']
        
        if os.path.isfile(image_after_binarization): 
            ind_checked = 3
             
            image = Image.open(image_after_binarization)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_third = io.BytesIO()    
            image.save(bio_third, format="PNG")    
        
        final_image = dictImages['Final_Image']
    
        if os.path.isfile(final_image):
            ind_checked = 4
            
            image = Image.open(final_image)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_fourth = io.BytesIO()     
            image.save(bio_fourth, format="PNG")    
            
            
        if ind_checked == 4:
             
            uni_sized = (300,300)
            size_text = (50,2)
            size_window = (800, 800)
            
            
            layout = [[sg.Image(data= bio_first.getvalue(), key = "-FIRST_IMG-", size=uni_sized), sg.Image(data= bio_sec.getvalue(), key = "-SEC_IMG-", size=uni_sized)],
                      [sg.Text("Initial HSV Image", size=size_text), sg.Text("Saturation component of HSV Image", size=size_text)],
                      [sg.Image(data= bio_third.getvalue(), key = "-FIRST_IMG-", size=uni_sized), sg.Image(data= bio_fourth.getvalue(), key = "-SEC_IMG-", size=uni_sized)],
                      [sg.Text("Image after binarization step", size=size_text), sg.Text("Final Image (after filling the holes)", size=size_text)],
                      [sg.Button("Back")]] 
            
            window = sg.Window("Intermdiate Graph", layout, resizable = True, disable_close = True, finalize = True, size=size_window) 
             ##   + add_str,
            while True:  
                event, values = window.read() 
                if event == "Exit" or event == sg.WIN_CLOSED:      
                    break
                if event == "Back":
                    break
            
            window.close()
            
        else:
            print("At least one path to image is wrong !")
            
            
            
        
        
    elif tasks_number == 3:
        print("Third task")
    
    elif tasks_number == 4:
        print("Fourth task")
        
        ## Check
   ##     dictImages
   
        if os.path.isfile(dictImages['Init_video']):
          ind_checked = 1
        else:
            print("Failed on loading directory for first video")
        if os.path.isfile(dictImages['InterVideo']):
          ind_checked = 2
        else:
            print("Failed on loading directory for second video")
        if os.path.isfile(dictImages['FinalVideo']):
          ind_checked = 3
        else:
          print("Failed on loading directory for third video")
          
        if ind_checked == 3:
          
            image = Image.open('C:/Users/marco/Downloads/block_diagram_4.png')
            image.thumbnail((2000, 400))    ## (100,5000)
            bio_block_gen = io.BytesIO()    
            image.save(bio_block_gen, format="PNG")    
          
            uni_sized = (300,300)
            size_text = (50,2)
            size_window = (800, 800)
        
       
            init_video_dir = dictImages['Init_video']
            inter_video_dir = dictImages['InterVideo']
            final_video_dir =  dictImages['FinalVideo']      
   
            layout = [[sg.Image(data= bio_block_gen.getvalue(), key = "-BLOCK_GEN_IMG-", size=uni_sized)],
                      [sg.Button("Show initial video")],
                      [sg.Button("Show intermediate video")],
                      [sg.Button("Show final video")],
                      [sg.Button("Back")]
                     ]
            
            uni_sized = (300,300)
            size_text = (50,2)
            size_window = (800, 800)
            
            window = sg.Window("Intermediate " + add_str, layout, resizable = True, disable_close = True, finalize = True, size=size_window) 
            
            while True:  
                event, values = window.read() 
                if event == "Exit" or event == sg.WIN_CLOSED:      
                    break
                if event == "Back":
                    break
                if event == "Show initial video":
                    display_video(init_video_dir)
                if event == "Show intermediate video":
                    display_video(inter_video_dir)
                if event == "Show final video":
                    display_video(final_video_dir)
                    
            
            window.close()     
 ##   C:/Users/marco/Downloads/block_diagram_4.png
    
def gui_show_block_diag(numberTask):
    
    import os    
    from PIL import Image
    import io
    import cv2
    
    if numberTask == 1:        
        name_block_diag = "block_diagram_1.png"        
    elif numberTask == 2:
        name_block_diag = "block_diagram_2.png"
    elif numberTask == 3:
        name_block_diag = "block_diagram_3.png"
        
    this_dir_x = os.getcwd()
    foldername_string_x = "CVis_2223_Assign1_MarcoGameiro"
    dir_parts_x = this_dir_x.split(foldername_string_x)
    base_dir_x = dir_parts_x[0] + foldername_string_x + "\\"
    base_path_x = base_dir_x + "code_python" + "\\"
    
    # img = cv2.imread(base_path_x + name_block_diag)
    
    uni_sized = (1000,200)   
    size_window = (1000, 250)
    
    # dim_block_diag = (int(len(img[0])*2), int(len(img)*2))
    
    # resized = cv2.resize(img, dim_block_diag, interpolation = cv2.INTER_AREA) 
    # cv2.imwrite(base_path_x + name_block_diag, resized)
    
    image = Image.open(base_path_x + name_block_diag)
    new_image = image.resize(uni_sized)
    new_image.save(base_path_x + name_block_diag) 
    
    image = Image.open(base_path_x + name_block_diag)
    image.thumbnail((1000, 50000))    ## (100,5000)
    bio_block = io.BytesIO()    
    image.save(bio_block, format="PNG")     
    
    layout = [[sg.Image(data= bio_block.getvalue(), key = "-BLOCK_IMG-", size=uni_sized)],
              [sg.Button("Back")]
             ] 
    
    window = sg.Window("Block Diagram", layout, resizable = True, disable_close = True, finalize = True, size=size_window)
    
    while True:  
        event, values = window.read() 
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == "Back":
            break
    
    window.close()  
    
    
    
def gui_show_inter_hists(code_data, dictImages):
    
    import os    
    from PIL import Image
    import io
    
    print("Intermdiate histograms")    
    
    if code_data == 1:
        add_str = "Images"
    elif code_data == 2:
        add_str = "Videos"
        
    tasks_number = int(dictImages['code_task']) 
    
    if tasks_number == 1:
        
        print("First task")
        
        hist_image_bef_bright = dictImages['hist_image_bef_bright']
        
        if os.path.isfile(hist_image_bef_bright):
            print("Checked")
            
            ind_checked = 2
            
            image = Image.open(hist_image_bef_bright)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_sec = io.BytesIO()    
            image.save(bio_sec, format="PNG")    
            
        if ind_checked == 2:
            
            uni_sized = (300,300)
            size_text = (50,2)
            size_window = (800, 800)
            
            layout = [[sg.Image(data= bio_sec.getvalue(), key = "-SEC_IMG-", size=uni_sized)],
                      [sg.Text("Histogram of image without brightness increase", size=size_text)],
                      [sg.Button("Back")]
                     ]
            
            window = sg.Window("Intermdiate Graphs", layout, resizable = True, disable_close = True, finalize = True, size=size_window)
            
            while True:  
                event, values = window.read() 
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
                if event == "Back":
                    break
            
            window.close()
            
        else:
            print("At least one path to image is wrong !")
            
    elif tasks_number == 2:
        
        print("Second task")
        
        hist_image = dictImages['image_after_hist']
        
        if os.path.isfile(hist_image):
            print("Checked")
            
            ind_checked = 1
            
            image = Image.open(hist_image)
            image.thumbnail((400, 25000))    ## (100,5000)
            bio_sec = io.BytesIO()    
            image.save(bio_sec, format="PNG")     
            
        if ind_checked == 1:
            
            uni_sized = (300,300)
            size_text = (50,2)
            size_window = (800, 800)
            
            layout = [[sg.Image(data= bio_sec.getvalue(), key = "-SEC_IMG-", size=uni_sized)],
                      [sg.Text("Histogram for S component of HSV image", size=size_text)],
                      [sg.Button("Back")]
                     ]
            
            window = sg.Window("Intermdiate Graphs", layout, resizable = True, disable_close = True, finalize = True, size=size_window)
            
            while True:  
                event, values = window.read() 
                if event == "Exit" or event == sg.WIN_CLOSED:
                    break
                if event == "Back":
                    break
            
            window.close() 
            
        else:
            print("At least one path to image is wrong !")
        
        
        
    elif tasks_number == 3:
        print("Third task")
        
    
    
##############################################
# image_bef_bright (grey)
# image_after_bright (grey)
# hist_image_bef_bright
# hist_image_after_bright 
# image_after_binarization
# image output of adaptable threshold
# masked image
# image with ROIS
##############################################
    

def gui_inter_steps(code_data, dictImages):
    
    if code_data == 1:
        add_str = "Images"
    elif code_data == 2:
        add_str = "Videos"        
   

    layout = [
          [sg.Button("Intermediate " + add_str)],      
          [sg.Button("Intermediate Graphs")]
    ]
    
    window = sg.Window("Intermdiate Steps", layout, finalize = True, resizable = True) 
    
    while True:
        
        event, values = window.read()
        print(event, values)   
        
        if event == "Exit" or event == sg.WIN_CLOSED: 
            break  
        if event == "Intermediate " + add_str:
            
            gui_show_block_diag(int(dictImages['code_task']))  
            
            gui_show_inter_buffer(code_data, dictImages)             
            
        if event == "Intermediate Graphs": 
            
            if code_data == 2:
                print("Option only available for images, not videos")
            else:
                gui_show_inter_hists(code_data, dictImages)  ## only for code_data = 1
 
       
    window.close()
     
## gui_inter_steps(2)    ## code_data = 1 
    
## code_data = 1 -> Image
## code_data = 2 -> Video
    
    