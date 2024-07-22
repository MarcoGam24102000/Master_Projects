# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:43:27 2022

@author: marco
"""

from __future__ import print_function


def extra_task_by_menu_calib():    
    
    import numpy as np
    import cv2 as cv
    from common import splitfn
    import os
    
    
    def gui_reference_image(code_where):
        import PySimpleGUI as sg
        
        msg = ''
        
        if code_where == 1:
            msg = 'l'
        elif code_where == 2:
            msg = 'r'
        
        again = True
        
        while again == True:
        
            windowx = sg.Window('Choose path to png reference file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
            (keyword, dict_dir) = windowx                
        
            dir_png_path = dict_dir['Browse'] 
            
            if '/' in dir_png_path:
                dir_png_parts = dir_png_path.split('/')
            elif "\\" in dir_png_path:
                dir_png_parts = dir_png_path.split("\\")
            
            png_filename = dir_png_parts[-1]    ## Check if it is a bag file
            
            if ('.png' in png_filename) and (msg in png_filename):
                again = False
            
        base_dir_png = ''
        
        for ind_d, d in enumerate(dir_png_parts):
             
            if ind_d < len(dir_png_parts) - 1: 
                base_dir_png += d + '/'
         
        png_info = [base_dir_png, png_filename] 
        
        return dir_png_path
    
    
    
    def getNumberContoursForImage(img_mask, bin_image):
        cnts = cv.findContours(bin_image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)     ## canny
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        imp_cnts = 0
        
        x_cnt = 0
        y_cnt = 0    

        
        for c in cnts:
            x,y,w,h = cv.boundingRect(c)
            
            if w> 20 and h> 20 and w< 200 and h< 200:      ## 50 50
                imp_cnts += 1
                
                print("Width: " + str(w))
                print("Height: " + str(h))
            
                # Blue color in BGR
                color = (255, 255, 0)
                  
                # Line thickness of 2 px
                thickness = 2
                  
                # Using cv2.rectangle() method
                # Draw a rectangle with blue line borders of thickness of 2 px
                bin_image = cv.rectangle(bin_image, (x,y), (x+w,y+h), color, thickness)
                
                roi_chess = bin_image[y:y+h,x:x+w]
                
                # roi_chess_after = np.zeros_like(roi_chess)          
                 
                
                for b in range(len(roi_chess[0])):
                    for a in range(len(roi_chess)):
                        if b < int(len(roi_chess[0])/16):
                            roi_chess[a,b] = 255
                            
                roi_chess = roi_chess[int(len(roi_chess)/4):,int(len(roi_chess[0])/4):]    
                            
                roi_chess_path = "C:\\VisComp\\PL\\Project_2\\Data\\roi_chess.png" 
                       
                
                cv.imwrite(roi_chess_path, roi_chess)
                        
                
          ##      roi_chess = cv.bitwise_not(roi_chess)
                
                cv.imwrite(img_mask, bin_image)
                
                x_cnt = x
                y_cnt = y
         
     ##   return imp_cnts,  (x_cnt, y_cnt) 
     
        return roi_chess_path, roi_chess


    def fillhole(input_image):
        '''
        input gray binary image  get the filled image by floodfill method
        Note: only holes surrounded in the connected regions will be filled.
        :param input_image:
        :return:
        '''
        im_flood_fill = input_image.copy()
        h, w = input_image.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        im_flood_fill = im_flood_fill.astype("uint8")
        cv.floodFill(im_flood_fill, mask, (0, 0), 255)
        im_flood_fill_inv = cv.bitwise_not(im_flood_fill)
        img_out = input_image | im_flood_fill_inv
        return img_out 


    def rotateImage(image, angle):
        row,col = image.shape
        center=tuple(np.array([row,col])/2)
        rot_mat = cv.getRotationMatrix2D(center,angle,1.0)
        new_image = cv.warpAffine(image, rot_mat, (col,row))
        return new_image

    def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized 


    def background_subtraction_chess_seg(img_mask, bin_image):       ## , dir_ref_img  


        if 'left' in img_mask:    
       ##     dir_ref_img = "C:\\VisComp\\PL\\Project_2\\Data\\CV_D435_IR_left_01.png"
           code_where = 1
            
        elif 'right' in img_mask:    
       ##     dir_ref_img = "C:\\VisComp\\PL\\Project_2\\Data\\CV_D435_IR_right_01.png"
           code_where = 2
        
        dir_ref_img = gui_reference_image(code_where)
        
        
        if len(bin_image.shape) == 3:         
            bin_image = cv.cvtColor(bin_image, cv.COLOR_BGR2GRAY)
        
        img_ref = cv.imread(dir_ref_img)
        img_ref = cv.cvtColor(img_ref, cv.COLOR_BGR2GRAY)
        
        img_parts = img_mask.split('.png')

        for im_part in img_parts:
            if len(im_part) != 0: 
                img_mask = im_part
        
        img_mask += "_corr"
        img_mask += '.png' 
        
        
        diff_img = np.zeros_like(img_ref)
        
        if len(bin_image) == len(img_ref) and len(bin_image[0]) == len(img_ref[0]):
            
            for b in range(len(img_ref[0])):
                for a in range(len(img_ref)):
                    diff_img[a,b] = abs(bin_image[a,b]-img_ref[a,b]) + 100
                    
                    if diff_img[a,b] >= 255:
                        diff_img[a,b] = 255
                    if diff_img[a,b] <= 0:
                        diff_img[a,b] = 0
                                           
                    if diff_img[a,b] < 200:
                        diff_img[a,b] = 0
                        
                        
                    ## Invert image 
                    
                    # if diff_img[a,b] > 250:
                    #     diff_img[a,b] = 0
                    # if diff_img[a,b] < 250:
                    #     diff_img[a,b] = 255 
                     
                    
            
            diff_img = cv.medianBlur(diff_img, 3)   
            
     ##       diff_img = cv.bitwise_not(diff_img)

            # # Creating kernel
            # kernel = np.ones((3, 3), np.uint8)
              
            # # Using cv2.erode() method 
            # diff_img = cv.erode(diff_img, kernel)   
            
            # kernel = np.ones((5, 5), 'uint8')

            # diff_img = cv.dilate(diff_img, kernel, iterations=1) 
            
                    
         
        
        cv.imwrite(img_mask, diff_img)  
            
        got_img = np.zeros_like(diff_img)     
        
        
        for b in range(len(got_img[0])):
            for a in range(len(got_img)):
                got_img[a,b] = 125
                
        chess_w_bg = got_img.copy()
        
    ##    c, (x_cnt, y_cnt) = getNumberContoursForImage(img_mask, diff_img)
        
        roi_chess_path, roi_chess = getNumberContoursForImage(img_mask, diff_img)
        
        roi_plot = False
        
        print("Width: " + str(len(roi_chess[0])))
        print("Height: " + str(len(roi_chess)))
         
        i = 0
        j = 0    
        
        
    ##    roi_chess =  cv.resize(roi_chess, (2*len(roi_chess[0]), 2*len(roi_chess)), interpolation = cv.INTER_AREA)  
        
      
        
        for b in range(len(got_img[0])):
            for a in range(len(got_img)):
                
                lim_down_b = int(len(got_img[0])/2 - len(roi_chess[0])*(1/2))
                lim_up_b =  int(len(got_img[0])/2 + len(roi_chess[0])*(1/2))
                
                lim_down_a = int(len(got_img)/2 - len(roi_chess)*(1/2)) 
                lim_up_a = int(len(got_img)/2 + len(roi_chess)*(1/2))
                
                if (b > lim_down_b and b < lim_up_b and a > lim_down_a and a < lim_up_a):
                      
                    chess_w_bg[a,b] = roi_chess[i,j]
                    roi_plot = True 
                
                if roi_plot == True and a < int(len(got_img)/2 + len(roi_chess)/2 - 1):
                    i += 1
                    roi_plot = False
            
            if roi_plot == True:
                
                j += 1 
                i = 0 
                roi_plot = False
        
        chess_w_bg_new = chess_w_bg.copy() 
        
        for b in range(len(chess_w_bg[0])):
            for a in range(len(chess_w_bg)):    
                if chess_w_bg[a,b] != 0:
                    chess_w_bg_new[a,b] = 255
        
        cnts_after_seg = 0
                    
        cnts = cv.findContours(chess_w_bg_new, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)     ## canny
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        
        x_c = 0
        y_c = 0
        w_c = 0
        h_c = 0
        
        
        for c in cnts:
            x,y,w,h = cv.boundingRect(c)
        
            if w > 100 and h> 50 and w < 200 and h< 150:      ## 50 50
                cnts_after_seg += 1
                
                break
        
        print("Number of contours after segmentation: " + str(cnts_after_seg))
        
        
        # chess_w_bg_new = cv.bitwise_not(chess_w_bg_new)
        # chess_w_bg_new = fillhole(chess_w_bg_new) 
        # chess_w_bg_new = cv.bitwise_not(chess_w_bg_new)
        
                    
        
        thickness = 2
        color = (0,0,0)   
        
        
    ##    chess_w_bg_new = cv.rectangle(chess_w_bg_new, (int(len(got_img[0])/2 - len(roi_chess[0])/2), int(len(got_img)/2 - len(roi_chess)/2)), 
    ##                                  ((int(len(got_img[0])/2 + len(roi_chess[0])/2), int(len(got_img)/2 + len(roi_chess)/2))), color, thickness)
                 
         
        chess_w_bg_new = cv.rectangle(chess_w_bg_new, (x_c, y_c), (x_c+w_c, y_c+h_c), color, thickness)
                                      
                                      
        roi_chess_path_parts = roi_chess_path.split('.png')
        
        for p in roi_chess_path_parts:
            if len(p) != 0:
                roi_chess_path_rem = p
        
        new_path_roi_chess = roi_chess_path_rem + '_bg' + '.png'
        
        
        chess_w_bg_new = chess_w_bg_new[int((1/4+1/8+1/16)*len(chess_w_bg_new)):int((1-(1/4+1/8+1/16))*len(chess_w_bg_new)), int((1/4+1/8+1/16)*len(chess_w_bg_new[0])):int((1-(1/4+1/8+1/16))*len(chess_w_bg_new[0]))]
        
        chess_w_bg_new = cv.cvtColor(chess_w_bg_new, cv.COLOR_GRAY2BGR)
        
        chess_w_bg_new = image_resize(chess_w_bg_new, width = 2*len(roi_chess[0]), height =  2*len(roi_chess))
        
        chess_w_bg_new = cv.resize(chess_w_bg_new, (6*len(chess_w_bg_new[0]), 6*len(chess_w_bg_new)), interpolation= cv.INTER_AREA)

             
        cv.imwrite(new_path_roi_chess, chess_w_bg_new)  
        
        ###########################################################################################################
        ###########################################################################################################
        ## sum chess board to image to a background grey (125 grey value) rectangular image.
         
        bg_basis = np.zeros((576,1024,3))    
        
        for c in range(0,3):
            for b in range(len(bg_basis[0])):
                for a in range(len(bg_basis)):
                    
                    bg_basis[a,b,c] = 210
                    
                    # if c == 0 or c == 2:
                    #     bg_basis[a,b,c] = 0
                    # else:
                    #     bg_basis[a,b,c] = 255
                        
        imx = bg_basis.copy()    
        
        print("Dims chess board: " + str(chess_w_bg_new.shape))
         
        
        for c in range(0,3):
            for b in range(len(bg_basis[0])):
                for a in range(len(bg_basis)):
                    if (a > 50 and b > 50) and (a < len(chess_w_bg_new) + 50 and b < len(chess_w_bg_new[0]) + 50):
                        imx[a,b,c] = chess_w_bg_new[a-51,b-51,c]   
        
        
        imx_dir = "C:\\VisComp\\PL\\Project_2\\Data\\test_image_example.png"
        cv.imwrite(imx_dir, imx)
        
        print("Dims after: " + str(imx.shape))
        
        ## Rotate    
         
        # imx_dir = "C:\\VisComp\\PL\\Project_2\\Data\\test_image_3.png"

        # imx = cv.imread(imx_dir)
        
        imx = cv.imread(imx_dir)

        imx = cv.cvtColor(imx, cv.COLOR_BGR2GRAY)
        
        angle = 0
        full_lap = 360
        
        print("Img Mask: " + img_mask)
        
        if 'left' in img_mask:
            angle = full_lap - 2
        elif 'right' in img_mask:
            angle = full_lap + 2
        
    #    imx = cv.rotate(imx, cv.ROTATE_180)
        new_imx = rotateImage(imx, angle)
         
        # Filter boards around image

        for b in range(len(new_imx[0])):
            for a in range(len(new_imx)):
                if a<(int((1/8)*len(new_imx))) or a>((int((7/8)*len(new_imx)))) or b < (int((1/8)*len(new_imx[0]))) or b > ((int((7/8)*len(new_imx[0])))):
                    new_imx[a,b] = 210               
        
        ## Move to left ?
        
        dir_new_imx = "C:\\VisComp\\PL\\Project_2\\Data\\test_image_3_rot_good.png"

        cv.imwrite("C:\\VisComp\\PL\\Project_2\\Data\\test_image_3_rot_good.png", new_imx)
        
        ## Try chessboard detection
            
            
        print("Number of contours: " + str(c))  
        # print("X Pos: " + str(x_cnt))
        # print("Y Pos: " + str(y_cnt))
         
    #    return img_mask  

    #    return new_path_roi_chess  



        for b in range(len(new_imx[0])):
            for a in range(len(new_imx)):
                
                if new_imx[a,b] > 80:   ## imx[a,b] < 220 and 
                
                    new_imx[a,b] -= 70
                    
                    if new_imx[a,b] >= 255:
                        new_imx[a,b] = 255
        
        print("Shape 1: " + str(new_imx.shape))
       
        new_imx = new_imx[20:,100:]
        
        print("Shape 2: " + str(new_imx.shape))
        
        cv.imwrite("C:\\VisComp\\PL\\Project_2\\Data\\test_image_3_rot_good.png", new_imx)


        return dir_new_imx 
    
    
    
    
    def gui_metadata_file_example(img_new_dir):
        
        import PySimpleGUI as sg
        
        again = True
        
        print("A")
        print(img_new_dir)
        
        csv_filename = ""
        
        if '/' in img_new_dir:
            img_2 = img_new_dir.split('/')
        elif "\\" in img_new_dir:
            img_2 = img_new_dir.split("\\")
            
        img_2 = img_2[:-2]
        
        print("B")
        print(img_2)
        
        new_dir = ''
        
        for d in img_2:
            new_dir += d + '/'
        
        
        layout = [
            [sg.Text('Metadata CSV filename: '), sg.Input(default_text= "", size=(19, 1), key="METADATA_FILE_EXAMPLE")],
            [sg.Button("Next")]
        ]
        
        window = sg.Window('Basis metadata CSV file', layout)
        
        while again == True:
            event, values = window.read()
            if event == "Exit" or event == sg.WIN_CLOSED:
                break
            if event == "Next":
                  csv_filename = values["METADATA_FILE_EXAMPLE"]  
                  
                  print(csv_filename)
                   
                  if not('.csv' in csv_filename):
                      csv_filename += '.csv'
                  
                  print(csv_filename) 
                      
                  # os.chdir("C:\\VisComp\\PL\\Project_2\\")              
                
                  # base_path = os.getcwd()
                  
                  base_path = new_dir
                  
                  if '/' in base_path:              
                      base_csv = base_path + csv_filename  
                  elif "\\" in base_path:              
                      base_csv = base_path + csv_filename 
                  
                  print(base_csv)
                  
                  if os.path.exists(base_csv):
                      again = False
                      print("CSV file found")
                  else:
                      again = True   
                      print("CSV file not found")
                
        window.close()
        
        
        return csv_filename     
    
    
    def gui_png_file(): 
        
        import PySimpleGUI as sg
        
        again = True
        
        while again == True:
        
            windowx = sg.Window('Choose path to png file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
            (keyword, dict_dir) = windowx                
        
            dir_png_path = dict_dir['Browse'] 
            
            if '/' in dir_png_path:
                dir_png_parts = dir_png_path.split('/')
            elif "\\" in dir_png_path:
                dir_png_parts = dir_png_path.split("\\")
            
            png_filename = dir_png_parts[-1]    ## Check if it is a bag file
            
            if '.png' in png_filename:
                again = False
            
        base_dir_png = ''
        
        for ind_d, d in enumerate(dir_png_parts):
             
            if ind_d < len(dir_png_parts) - 1: 
                base_dir_png += d + '/' 
         
        png_info = [base_dir_png, png_filename]
         
        return dir_png_path    
    
    
    def gui_params_metadata(params_start, params, diff_params):
         
        import PySimpleGUI as sg
        
        fx = params_start[0]
        fy = params_start[1]
        ppx = params_start[2]
        ppy = params_start[3]
        
        fx_calib = params[0]
        fy_calib = params[1]
        ppx_calib = params[2]
        ppy_calib = params[3] 
        
        diff_fx = diff_params[0]
        diff_fy = diff_params[1]
        diff_ppx = diff_params[2]
        diff_ppy = diff_params[3]
        
        
        layout_one = [
            [sg.Text("Parameters before calibration:")],
            [sg.Text('Fx:'), sg.Input(default_text= str(fx), key="Fx")],
            [sg.Text('Fy:'), sg.Input(default_text= str(fy), key="Fy")],
            [sg.Text('PPx:'), sg.Input(default_text= str(ppx), key="PPx")],
            [sg.Text('PPy:'), sg.Input(default_text= str(ppy), key="PPy")]
        ]
        
        layout_two = [
            [sg.Text("Parameters after calibration:")],
            [sg.Text('Fx Calib:'), sg.Input(default_text= str(fx_calib), key="Fx_Calib")],
            [sg.Text('Fy Calib:'), sg.Input(default_text= str(fy_calib), key="Fy_Calib")],
            [sg.Text('PPx Calib:'), sg.Input(default_text= str(ppx_calib), key="PPx_Calib")],
            [sg.Text('PPy Calib:'), sg.Input(default_text= str(ppy_calib), key="PPy_Calib")]
        ]
        
        layout_three = [
            [sg.Text("Results:")],
            [sg.Text('Fx Progression:'), sg.Input(default_text= str(diff_fx), key="Fx_Diff")],
            [sg.Text('Fy Progression:'), sg.Input(default_text= str(diff_fy), key="Fy_Diff")],
            [sg.Text('PPx Progression:'), sg.Input(default_text= str(diff_ppx), key="PPx_Diff")],
            [sg.Text('PPy Progression:'), sg.Input(default_text= str(diff_ppy), key="PPy_Diff")]
        ]
        
        layout_opt = [
            [sg.Button("Back")]
        ]
        
        
        layout = [ 
            [
                sg.Column(layout_one),
                sg.VSeparator(),
                sg.Column(layout_two),
                sg.VSeparator(),
                sg.Column(layout_three),
                sg.VSeparator(),            
                sg.Column(layout_opt)
            ]
        ]
        
        
        if fx_calib > fx:
            print("Fx parameter increased")
        elif fx_calib == fx:
            print("Same Fx")
        else:
            print("Fx parameter decreased")
            
        if fy_calib > fy:
             print("Fy parameter increased")
        elif fy_calib == fy:
             print("Same Fy")
        else:
             print("Fy parameter decreased")
        
        if ppx_calib > ppx:
            print("PPx parameter increased")
        elif ppx_calib == ppx:
            print("Same PPx")
        else:
            print("PPx parameter decreased")
        
        if ppy_calib > ppy:
            print("PPy parameter increased")
        elif ppy_calib == ppy:
            print("Same PPy")
        else:
            print("PPy parameter decreased")       
        
        
        window = sg.Window('Camera calibration results', layout)
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           if event == "Back":
               break
           
        window.close()
         
    
    def getDateTimeStrMarker(): 
    
        import datetime
        
        e = datetime.datetime.now()
        
        print ("Current date and time = %s" % e)
        
        print ("Today's date:  = %s/%s/%s" % (e.day, e.month, e.year))
        
        print ("The time is now: = %s:%s:%s" % (e.hour, e.minute, e.second))
        
        if e.day < 10:
            daystr = '0' + str(e.day)
        else:
            daystr = str(e.day)
        
        if e.month < 10:
            monthstr = '0' + str(e.month)
        else:
            monthstr = str(e.month)
            
        year_list = list(str(e.year))
        yearList = year_list[2:]
        yearstr = str(yearList[0]) + str(yearList[1])    
            
        
        if e.hour < 10:
            hourstr = '0' + str(e.hour)
        else:
            hourstr = str(e.hour)
        
        if e.minute < 10:
            minutestr = '0' + str(e.minute)
        else:
            minutestr = str(e.minute)
            
        if e.second < 10:
            secondstr = '0' + str(e.second)
        else:
            secondstr = str(e.second)    
            
            
        adderToPath = '_' + daystr + monthstr + yearstr + '_' + hourstr + minutestr + secondstr + '_'
        
        return adderToPath
    
    
    def write_data_to_csv_file(data_lab, name_csv_file):
         with open(name_csv_file + '.csv', 'a', encoding='utf-8') as f:
             
             for data in data_lab:
             
                 line = ', '.join(data)
                 f.write(line + '\n')  
                 
                 print("One more line written to csv file ...")  
    
    
    def write_params_to_metadata_file(metadata_csv_filename, params):
        
        title_one = "Frame Info: "
        
        frame_number = 1690
        timestamp = 1604506748310.08 
        resolution_x = 848
        resolution_y = 480
        bytes_per_pix = 1
        
        if 'left' in metadata_csv_filename:
            code_cam = 1
        elif 'right' in metadata_csv_filename:
            code_cam = 2
        
        type_line = ["Type", "Infrared " + str(code_cam)]
        format_line = ["Format", "Y8"]
        frame_number_line = ["Frame Number", str(frame_number)]
        timestamp_line = ["Timestamp (ms)", str(timestamp)]    
        resolution_x_line = ["Resolution x", str(resolution_x)]
        resoltion_y_line = ["Resolution y", str(resolution_y)]
        bytes_per_pix_line = ["Bytes per pixel", str(bytes_per_pix)]
        
        empty_line = ""
        
        title_two = "Intrinsic:"
        title_two = [title_two, ""]
        
        #########################################3
        
        fx = params[0]
        fy = params[1]
        ppx = params[2]
        ppy = params[3]
        
        fx_line = ["Fx", str(fx)]
        fy_line = ["Fy", str(fy)]
        ppx_line = ["PPx", str(ppx)]
        ppy_line = ["PPy", str(ppy)]
        
        distorsion_description = "Brown Conrady"
        
        distorsion_line = ["Distorsion", distorsion_description]
        
        data_lab = [type_line, format_line, frame_number_line, timestamp_line, 
                    resolution_x_line, resoltion_y_line, bytes_per_pix_line, 
                    empty_line, title_two, fx_line, fy_line, ppx_line, ppy_line,
                    distorsion_line]    
        
        
        write_data_to_csv_file(data_lab, metadata_csv_filename)    
     
    
    def extract_params_infrared_metadata(metadata):
        
        import csv
        
        os.chdir("C:\\VisComp\\PL\\Project_2\\")
        
        base_folder = os.getcwd()
        base_folder = base_folder.replace('\\', '/')
        base_folder += '/'
        
        frameInfo = False
        intrinsic = False
        
        type_acq = "" 
        format_d = ""
        frame_number = 0
        timestamp = 0
        resx = 0
        resy = 0    
        bytes_per_pix = 0
        
        fx = 0
        fy = 0
        ppx = 0
        ppy = 0
        distorsion = ""
        skip = False
        
        counter_params = 0
        
        base_folder += 'Data/'
    
        with open(base_folder + metadata, 'r') as file:
            reader = csv.reader(file) 
            
            for rowNumber, row in enumerate(reader):  
        ##        if rowNumber > 0:  
            
    ##          print("Reading line number " + str(rowNumber))         
               
            
                # print(row[0])
                
                # if len(row) == 2:
                #     print("Two params in this row")
                #     print(row[1])
                
    #           print(len(row))
                
                if len(row) > 0:
                   
                    if 'Frame Info:' in row[0]:
                        frameInfo = True
                        counter_params += 1
      ##                 print("Frame Info")
                    if frameInfo == True:
                        if row[0] == 'Type':
                            type_acq = row[1]
                            counter_params += 1
      ##                     print("Type")
                        elif row[0] == 'Format':
                            format_d = row[1]
                            counter_params += 1
    ##                      print("Format")
                        elif row[0] == 'Frame Number':
                            frame_number = int(row[1])
                            counter_params += 1
      ##                     print("Frame Number")
                        elif row[0] == 'Timestamp (ms)':
                            timestamp = float(row[1])
                            counter_params += 1
    ##                      print("Timestamp (ms)")
                        elif row[0] == 'Resolution x':
                            resx = int(row[1])
                            counter_params += 1
    ##                      print("Resolution x")
                        elif row[0] == 'Resolution y':
                            resy = int(row[1]) 
                            counter_params += 1
    ##                      print("Resolution y")
                        elif row[0] == 'Bytes per pixel':
                            bytes_per_pix = int(row[1])
                            counter_params += 1
                            frameInfo = False
      ##                     print("Bytes per pixel") 
                    elif frameInfo == False:
                        if row[0] == 'Intrinsic:':
                            intrinsic = True
                            counter_params += 1
    ##                      print("Intrinsic")
                    if intrinsic == True:
                        if row[0] == 'Fx':
                            fx = float(row[1])
                            counter_params += 1
      ##                     print("Fx")
                        elif row[0] == 'Fy':
                            fy = float(row[1])
                            counter_params += 1
    ##                      print("Fy")
                        elif row[0] == 'PPx':
                            ppx = float(row[1])
                            counter_params += 1
    ##                      print("PPx")
                        elif row[0] == 'PPy':
                            ppy = float(row[1])
                            counter_params += 1
    ##                      print("PPy")
                        elif row[0] == 'Distorsion':
                            distorsion = row[1] 
                            counter_params += 1
      ##                     print("Distorsion")
                else:
    ##              print("Skipping this line")
                    skip = True
                            
        k = []
                        
        if counter_params == 14:
                        
            k = [[fx, 0, ppx],
                  [0, fy, ppy],
                  [0, 0, 1]
            ] 
            
        else:
            print("Counting params: " + str(counter_params)) 
        
         
        return (fx, fy, ppx, ppy)    
    
    
     
    def main():
        import sys
        import getopt
        from glob import glob
        
        dir_png_path = gui_png_file()  

        args, img_mask = getopt.getopt(sys.argv[1:], '', ['debug=', 'square_size=', 'threads='])
        args = dict(args)
        args.setdefault('--debug', './output/')
        args.setdefault('--square_size', 1.0)
        args.setdefault('--threads', 4)
        if not img_mask:
            img_mask = dir_png_path
            
     ##       img_mask =  "C:\\VisComp\\PL\\Project_2\\Data\\CV_D435_IR_left_calib_Infrared.png"  # '../data/left??.jpg' # default        
            
        else:
            img_mask = img_mask[0] 
            
        if '/' in img_mask:
            img_parts = img_mask.split('/')
        elif "\\" in img_mask:
            img_parts = img_mask.split("\\")        
            
        img_new_dir = ''
        
        for d in img_parts:
            img_new_dir += d + '/'
            
            
        
        # bin_image = cv.cvtColor(bin_image, cv.COLOR_BGR2GRAY)
        
        # for b in range(len(bin_image[0])):
        #     for a in range(len(bin_image)):
        #         if bin_image[a,b] != 255:
        #             bin_image[a,b] = 0
                
        #         if bin_image[a,b] == 255:
        #             bin_image[a,b] = 0
        #         elif bin_image[a,b] == 0:
        #             bin_image[a,b] = 255
                    
                    
        # img_parts = img_mask.split('.png')

        # for im_part in img_parts:
        #     if len(im_part) != 0:
        #         img_mask = im_part 
        
        # img_mask += "_corr"
        # img_mask += '.png'             
        
        
        # cv.imwrite(img_mask, bin_image)
        
        
        if 'D435' in img_mask: 
            
                resp = input('Run with segmentation algorithm ?')
                
                if ('s' in resp) or ('S' in resp) or ('y' in resp) or ('Y' in resp):
                 
                    bin_image = cv.imread(img_mask)   
                
                    img_mask = background_subtraction_chess_seg(img_mask, bin_image)    
            
                    bin_image = cv.imread(img_mask)
            
                    print(bin_image.shape)
        
       
              
        # if 'right' in img_mask:
        #     img_mask = cut_to_center(img_mask) 
     
        img_names = glob(img_mask)  
        debug_dir = args.get('--debug')
        if debug_dir and not os.path.isdir(debug_dir): 
            os.mkdir(debug_dir)
            
        square_size = float(args.get('--square_size'))  
     
      
        pattern_size = (7, 11)    ## (9,6)
        

        obj_points = []
        img_points = []
        h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results

        def processImage(fn):
            print('processing %s... ' % fn)
            
            img = cv.imread(fn, 0)   ## 0
            
     #       img = cv.imread("C:\\VisComp\\PL\\Project_2\\Data\\Im_L_1.png")
            
            print("Shape of image going to be processed: " + str(img.shape))
            
       #     gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            
            if img is None: 
                print("Failed to load", fn)
                return None

            assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
            pattern_size = (9, 6)  
            pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
            pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) 
            pattern_points *= square_size
            
            for j in range(3,12):
                for i in range(3,12):
                    if i != j:
                        pattern_size = (i,j)
                        found, corners = cv.findChessboardCorners(img, pattern_size)
                        if found:
                            i_worked = i
                            j_worked = j
                            
                            print("Worked")
                            
                            if i_worked != 0 and j_worked != 0:
                                pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
                                pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2) 
                                pattern_points *= square_size 
                                break
                        else: 
                            print("Keeps not working")   
            
            
      #      found, corners = cv.findChessboardCorners(img, pattern_size)
            
      ##      pattern_size = (7, 11)       
            
            
            
            
            print("Found: " + str(found))
            if found:
                term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
                cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

            if debug_dir:
                if len(img.shape) == 1:
                    vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
                else:
                    vis = img
                cv.drawChessboardCorners(vis, pattern_size, corners, found)
                _path, name, _ext = splitfn(fn)
                outfile = os.path.join(debug_dir, name + '_chess.png')
                cv.imwrite(outfile, vis)

            if not found:
                print('chessboard not found')
                return None 

            print('           %s... OK' % fn)
            return (corners.reshape(-1, 2), pattern_points)  

        threads_num = int(args.get('--threads'))
         
        print("Number of threads: " + str(threads_num))
        
    ##    if threads_num <= 1:
        if True:
            chessboards = [processImage(fn) for fn in img_names]
        # else:
        #     print("Run with %d threads..." % threads_num)
        #     from multiprocessing.dummy import Pool as ThreadPool
        #     pool = ThreadPool(threads_num)
        #     chessboards = pool.map(processImage, img_names)

        chessboards = [x for x in chessboards if x is not None]
        for (corners, pattern_points) in chessboards:
            img_points.append(corners)
            obj_points.append(pattern_points)

        print("Length of img_points: " + str(len(img_points)))
        print("Length of obj_points: " + str(len(obj_points)))
        
        
        if len(obj_points) != 0 and len(img_points) != 0:

            # calculate camera distortion 
            rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
        
            print("\nRMS:", rms)
            print("camera matrix:\n", camera_matrix)
            print("distortion coefficients: ", dist_coefs.ravel())
        
            # undistort the image with the calibration
            print('')
            for fn in img_names if debug_dir else []:
                _path, name, _ext = splitfn(fn)
                img_found = os.path.join(debug_dir, name + '_chess.png')
                outfile = os.path.join(debug_dir, name + '_undistorted.png')
        
                img = cv.imread(img_found) 
                if img is None:
                    continue
        
                h, w = img.shape[:2]
                newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
        
                dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
        
                # crop and save the image 
                x, y, w, h = roi 
                dst = dst[y:y+h, x:x+w]
        
                print('Undistorted image written to: %s' % outfile)  
                cv.imwrite(outfile, dst)
                
                
        
            print('Done')
            
            return newcameramtx, img_new_dir
        
        
    newcameramtx, img_new_dir = main()
    
    fx_calib = newcameramtx[0,0]
    fy_calib = newcameramtx[1,1]
    
    ppx_calib = newcameramtx[0,2] 
    ppy_calib = newcameramtx[1,2]
    
    
    
    ## metadata_example = "CV_D435_IR_left_calib_Infrared_metadata.csv"
    
    metadata_example = gui_metadata_file_example(img_new_dir)
    
    fx, fy, ppx, ppy = extract_params_infrared_metadata(metadata_example) 
    
    print("Fx from Metadata file: " + str(fx)) 
    print("Fx after calibration: " + str(fx_calib))
    
    print("Fy from Metadata file: " + str(fy))
    print("Fy after calibration: " + str(fy_calib))
    
    print("PPx from Metadata file: " + str(ppx)) 
    print("PPx after calibration: " + str(ppx_calib))
    
    print("PPy from Metadata file: " + str(ppy))
    print("PPy after calibration: " + str(ppy_calib)) 
    
    #########################################################################################
    #########################################################################################
    
    params_start = [fx, fy, ppx, ppy]
    
    params = [fx_calib, fy_calib, ppx_calib, ppy_calib]
    
    delta_fx = fx_calib - fx
    delta__fy = fy_calib - fy
    delta_ppx = ppx_calib - ppx
    delta_ppy = ppy_calib - ppy
    
    diff_params = [delta_fx, delta__fy, delta_ppx, delta_ppy]
    
    gui_params_metadata(params_start, params, diff_params)
    
     
    ## Right again parameters to metadata file
    
    metadata_parts = metadata_example.split(".csv") 
    
    for meta in metadata_parts:
        if len(meta) != 0:
            meta_title_info = meta
            
    metadata_csv_filename = meta_title_info + getDateTimeStrMarker()
    
    ## metadata_csv_filename = "CV_D435_IR_left_calib_Infrared_metadata_2712"
    
    write_params_to_metadata_file(metadata_csv_filename, params)
    
    print("End of calibration  \n-----------------------") 
    
    return 0
    