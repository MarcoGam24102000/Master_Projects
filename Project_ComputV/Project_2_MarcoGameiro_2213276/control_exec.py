# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 07:47:09 2022

@author: marco
"""


def camera_calib_opt():
    
    import PySimpleGUI as sg
    
    opt_camera = -1
    
    again = True
    
    layout = [
        [sg.Text("Which camera(s) do you want to calibrate ? ")],      ## Use a pre-recorded video    ## Perform a new image acquisition process
        [sg.T("         "), sg.Checkbox('From left', default=False, key="-IN1-")],
        [sg.T("         "), sg.Checkbox('From right', default=True, key="-IN2-")],        
        [sg.Button("Next")]
    ]
    
    
    window = sg.Window('Camera calibration menu', layout)
    
    while again == True:
       event, values = window.read()
       if event == "Exit" or event == sg.WIN_CLOSED:
           break
       if event == "Next":
           if values["-IN1-"] == True and values["-IN2-"] == False:
               opt_camera = 0
               again = False
           elif values["-IN1-"] == False and values["-IN2-"] == True:
               opt_camera = 1
               again = False
           elif values["-IN1-"] == True and values["-IN2-"] == True:                
               opt_camera = 2
               again = True
               
           break
    
    window.close()
    
    return opt_camera   

def gui_bag_file(): 
    
    import PySimpleGUI as sg
    
    again = True
    
    while again == True:
    
        windowx = sg.Window('Choose path to bag file', [[sg.Text('File name')], [sg.Input(), sg.FileBrowse()], [sg.OK(), sg.Cancel()] ]).read(close=True)
        (keyword, dict_dir) = windowx                
    
        dir_bag_path = dict_dir['Browse'] 
        
        if '/' in dir_bag_path:
            dir_bag_parts = dir_bag_path.split('/')
        elif "\\" in dir_bag_path:
            dir_bag_parts = dir_bag_path.split("\\")
        
        bag_filename = dir_bag_parts[-1]    ## Check if it is a bag file
        
        if '.bag' in bag_filename:
            again = False
        
    base_dir_bag = ''
    
    for ind_d, d in enumerate(dir_bag_parts):
         
        if ind_d < len(dir_bag_parts) - 1: 
            base_dir_bag += d + '/'
     
    bag_info = [base_dir_bag, bag_filename]
    
    return bag_info

def gui_metadata_files():
    
    import PySimpleGUI as sg  
    
    again = True
    
    while again == True:
    
        layout = [       
            [sg.Text('Metadata IR Left CSV filename:'), sg.Input(default_text= "", size=(19, 1), key="METADATA_IR_LEFT")],
            [sg.Text('Metadata IR Right CSV filename:'), sg.Input(default_text= "", size=(19, 1), key="METADATA_IR_RIGHT")],
            [sg.Button("Next")]
        ]
        
        window = sg.Window('Metadata Files', layout)
        
        while True:
             event, values = window.read()
             if event == "Exit" or event == sg.WIN_CLOSED:
                 break
             if event == "Next":
                 meta_left = values['METADATA_IR_LEFT']
                 meta_right = values['METADATA_IR_RIGHT']
                  
                 if '.csv' in meta_left:
                     meta_left_new = meta_left.split('.csv') 
                     
                     for str_csv in meta_left_new:
                         if len(str_csv) != 0:
                             meta_left = str_csv 
                 
                 if not('left' in meta_left):
                      again = True
                 else:
                      again = False
                     
                        
                 if '.csv' in meta_right:
                     meta_right_new = meta_right.split('.csv') 
                     
                     for str_csv in meta_right_new:
                         if len(str_csv) != 0:
                             meta_right = str_csv 
                             
                 if not('right' in meta_right):
                     again = True  
                 else:
                     again = False 
                                  
                 break
    
    window.close()
    
    meta_files = [meta_left, meta_right]
    
    return meta_files 
   

def aux_guis_first_menu():
    
    import PySimpleGUI as sg
    
    upper_lev_dist = 50
    lower_lev_dist = 1 
    
    again = True
    
    while again == True:
    
        layout = [
            [sg.Text('CSV Filename:'), sg.Input(default_text= "", size=(19, 1), key="CSV_FILENAME")],
            [sg.Text('Distance between infrared cameras:'), sg.Input(default_text= "", size=(19, 1), key="DIST_BET_IR_CAMERAS")],
            [sg.Button("Next")]
        ]
        
        window = sg.Window('Guidelines for the first goal', layout)
    
        while True:
             event, values = window.read()
             if event == "Exit" or event == sg.WIN_CLOSED:
                 break
             if event == "Next":
                 csv_filename = values['CSV_FILENAME']             
                 dist_ir_cameras = float(values['DIST_BET_IR_CAMERAS'])
                 
                 if dist_ir_cameras < lower_lev_dist or dist_ir_cameras > upper_lev_dist:
                     again = True
                 else:
                     again = False
        
                     if '.csv' in csv_filename:
                         csv_filename_new = csv_filename.split('.csv')
                         
                         for str_csv in csv_filename_new:
                             if len(str_csv) != 0:
                                 csv_filename = str_csv 
                 break
             
    window.close()
    
    adit_info = [csv_filename, dist_ir_cameras] 
    
    return adit_info


def aux_guis_sec_menu():
    
    import PySimpleGUI as sg
    
    default_init_n_people_lab = 10
    
    again = True    
    
    layout = [
        [sg.Text('CSV Filename:'), sg.Input(default_text= "", size=(19, 1), key="CSV_FILENAME")],
        [sg.Text('Initial number of people inside lab:'), sg.Input(default_text= str(default_init_n_people_lab), size=(19, 1), key="INIT_PEOPLE_LAB")],
        [sg.Button("Next")]
    ] 
    
    window = sg.Window('Guidelines for the second goal', layout)

    while again == True:
         event, values = window.read()
         if event == "Exit" or event == sg.WIN_CLOSED:
             break
         if event == "Next":
             csv_filename = values['CSV_FILENAME']
             init_people_lab = int(values['INIT_PEOPLE_LAB'])
             
             if init_people_lab < 10:
                 again = True
             else:
                 again = False
             
                 if '.csv' in csv_filename:
                     csv_filename_new = csv_filename.split('.csv')
                     
                     for str_csv in csv_filename_new:
                         if len(str_csv) != 0:
                             csv_filename = str_csv 
             break

    window.close()
    
    adit_info = [csv_filename, init_people_lab]
    
    return adit_info  
    
    

def comp_numberImages(str_stream_time, fps_value):
    stream_time = int(str_stream_time)
    
    n_images = stream_time*fps_value
    
    return n_images


def video_streaming_time():
    import PySimpleGUI as sg
    
    tup_vst = (True, 30)
    code_next = True
    
    
    layout = [
        [sg.Text('Streaming time (s):'), sg.Input(default_text= "30", size=(19, 1), key="STREAM_TIME")],
        [sg.Button("Back"),  sg.Button("Next")]
    ] 
    
    window = sg.Window('Streaming time', layout)
    
    while True:
        event, values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        if event == 'Back':
            code_next = False
            tup_vst = (code_next, 0)
            break
        if event == 'Next':
            code_next = True
            stream_time = int(values['STREAM_TIME'])
            tup_vst = (code_next, stream_time)
            
            break
            
    
    window.close()
    
    return tup_vst  
    

def limited_streaming_time():
    import PySimpleGUI as sg
    
    again = True
    
    while again == True:
    
        limited_stream_time = False
        repInit = True
        
        stream_time = 0
        tup_stream_time = ('Yes', '10')
        resp = ''
        
        layout = [
            [sg.Text("Limited streaming time: ")],      ## Use a pre-recorded video    ## Perform a new image acquisition process
            [sg.T("         "), sg.Checkbox('Yes', default=False, key="-IN1-")],
            [sg.T("         "), sg.Checkbox('No', default=True, key="-IN2-")],
            [sg.Button("Next")]
        ]
         
        window = sg.Window('Limited streaming time', layout)
        
        while repInit == True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           if event == "Next":
               if values["-IN1-"] == True and values["-IN2-"] == False:
                   limited_stream_time = True
                   repInit = False
                   break
               elif values["-IN2-"] == True and values["-IN1-"] == False:
                   limited_stream_time = False
                   repInit = False
                   break
               else:
                   if values["-IN1-"] == True and values["-IN2-"] == True:
                       print("Only pick one option !!!")                   
                       repInit = True           
                       continue
        window.close() 
    
        if limited_stream_time == True:
            resp = 'Yes'
            tup_vst = video_streaming_time()
            (next_resp, stream_time) = tup_vst 
            
            if next_resp == True:
                again = False
            else:
                again = True
        else:
            resp = 'No'
            stream_time = 0  
            again = False
         
    tup_stream_time = (resp, str(stream_time))
    
    return tup_stream_time


def control_stream(fps_value):     
     
    n_images = 0
    lim = 'No'
    
    print("FPS value: " + str(fps_value))
        
    (limited, str_stream_time) = limited_streaming_time()    
    
    if limited == 'Yes':
        print("Limited streaming time")
        n_images = comp_numberImages(str_stream_time, fps_value) 
        print("For " + str(n_images) + " images ...")
        lim = 'Yes'
    else:
        print("Non-limited streaming time")          
        
    tup_n_images = (lim, n_images)
        
    return tup_n_images 
 

## tup_n_images = control_stream(15)
           
    