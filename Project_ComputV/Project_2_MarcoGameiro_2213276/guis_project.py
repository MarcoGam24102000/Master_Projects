# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:48:09 2022

@author: marco
"""

from second_task_by_menu import sec_task_by_menu
from first_task_by_menu import first_task_by_menu_all
from extra_task_by_menu import extra_task_by_menu_calib


def set_guis():
    
    width = 848
    height = 480
    fps = 15
    
    fx = 10
    fy = 10
    ppx = 10
    ppy = 10
    
    
    def video_params(width, height, fps):
        print("Video parameters")
        
        import PySimpleGUI as sg
        
        # default_width = 848
        # default_height = 480
        # default_fps = 15
        
        layout = [
            [sg.Text('Width:'), sg.Input(default_text= str(width), key="WIDTH")],
            [sg.Text('Height:'), sg.Input(default_text= str(height), key="HEIGHT")],
            [sg.Text('Frame-rate:'), sg.Input(default_text= str(fps), key="FRAME_RATE")],
            [sg.Button("Back"), sg.Button("Save")]
        ]
        
        window = sg.Window('Video parameters', layout, modal = True)
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           if event == "Back":
               break
           if event == "Save":
               if int(values['WIDTH']) != width:
                   width = int(values['WIDTH'])
               if int(values['HEIGHT']) != height:
                   height = int(values['HEIGHT'])
               if int(values['FRAME_RATE']) != fps:
                   fps = int(values['FRAME_RATE'])
                   
               break    
           
        window.close()
        
        video_params = [width, height, fps]
        
        return video_params       
        
    
    # def first_task_sec():
    #     print("Compute depth map using stereo vision")
        
    #     import PySimpleGUI as sg
        
    #     show_data = True
        
    #     back = False
        
    #     while show_data == True:
            
    #         ## get depth map as well as its statistics            
    #         ## path to require bag file
    #         ## another function for processing step            
        
    #         basis = 10
    #         focal_lenght = 10
    #         mean_disparity = 10           
         
     
    #         params_layout = [
    #             [sg.Text('Basis:'), sg.Text(str(basis), size = (10,2), key='-BASIS-')],
    #             [sg.Text('Focal lenght:'), sg.Text(str(focal_lenght), size = (10,2), key='-FOCAL_LENGHT-')],
    #             [sg.Text('Disparity:'), sg.Text(str(mean_disparity), size = (10,2), key='-MEAN_DISP-')], 
                
    #         ]
            
         
    #         print("Here")
            
    #         layout_buttons = [
    #             [sg.Button("Back")],
    #             [sg.Button("Stop")]
    #         ]
            
    #         layout = [    
    #             [
    #                 sg.Column(params_layout),
    #                 sg.VSeparator(),                 
    #                 sg.Column(layout_buttons)
                    
    #             ]            
    #         ]
            
    #         print("Here A")
            
    #         window = sg.Window('Camera parameters', layout, resizable = True, finalize = True, margins=(0,0))         ##  modal = True   
            
    #         print("Here B")
            
    #         while True:
    #            event, values = window.read()
    #            if event == "Exit" or event == sg.WIN_CLOSED:
    #                break
    #            if event == "Back":
    #                show_data = False
    #       ##         break
    #                back = True
                    
    #                break
                  
    #            if event == "Stop":
    #                show_data = False 
    #                break  
               
    #         if back == True:
    #             print("Back from first")
    #             window.close()
    #             return back        
                                  
             
        
        
    def third_task_sec(fx, fy, ppx, ppy):
        print("Stereo system calibration") 
        
        import PySimpleGUI as sg
        
        fx = 0
        fy = 0
        ppx = 0
        ppy = 0
        
        layout = [
            [sg.Text('Fx: '), sg.Text(str(fx), key='-FX-')],
            [sg.Text('Fy: '), sg.Text(str(fy), key='-FY-')],
            [sg.Text('PPx: '), sg.Text(str(ppx), key='-PPX-')],
            [sg.Text('PPy: '), sg.Text(str(ppy), key='-PPY-')],
            [sg.Button("Back"), sg.Button("Save")]
        ]
        
        window = sg.Window('Menu', layout)
        
        while True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               break
           if event == "Back":
               break
           if event == "Save":
               
               if int(values['-FX-']) != width:
                   fx = int(values['-FX-'])
               if int(values['-FY-']) != height:
                   fy = int(values['-FY-'])
               if int(values['-PPX-']) != fps:
                   ppx = int(values['-PPX-'])
               if int(values['-PPY-']) != fps:
                   ppy = int(values['-PPY-'])
                   
               break    
           
            
        window.close()      
        
        calib_params = [fx, fy, ppx, ppy]
        
        return calib_params
        
        
    def menu_sec(width, height, fps, fx, fy, ppx, ppy):
        print("Menu")        
        
        import PySimpleGUI as sg
        
        repeat = True       
        
        got_move = True
        
        
        layout = [
            [sg.Button("Compute depth map using stereo vision")],
            [sg.Button("Track number of people inside lab")],
            [sg.Button("Stereo sistem calibration")],
            [sg.Button("Exit"), sg.Button("Video-related parameters")]
        ]
        
        window = sg.Window('Menu', layout)
        
        while repeat == True:
           event, values = window.read()
           if event == "Exit" or event == sg.WIN_CLOSED:
               return True, [width, height, fps, fx, fy, ppx, ppy]
               
       ##        break           
           
           if event == "Compute depth map using stereo vision":
               
               path_to_bag_file = "C:/VisComp/PL/Project_2/CV_D435_20201104_162148.bag"
               
               first_task_by_menu_all(path_to_bag_file, width, height, fps)             
             
                              
        ##       return [False]
           if event == "Track number of people inside lab":
               
               path_to_bag_file = "C:/VisComp/PL/Project_2/CV_D435_20201104_162148.bag"
               
               sec_task_by_menu(path_to_bag_file, width, height, fps) 
               
               repeat = True
         
               
        ##       return [False]
           if event == "Stereo sistem calibration":
             
               resp = extra_task_by_menu_calib()
               
               if resp == 0:
                   print("Calibration process completed with success")
                   repeat = True                   
             
               
        ##       return [False]
           if event == "Video-related parameters":
               video_info = video_params(width, height, fps) 
               width = video_info[0]
               height = video_info[1]
               fps = video_info[2]
           
        window.close()
        
    info = menu_sec(width, height, fps, fx, fy, ppx, ppy)
    
    if info is not None:
    
        if len(info) == 2:
            bool_resp, params = info 
        
            if len(params) == 7: 
                width = params[0]
                height = params[1]
                fps = params[2]  
                fx = params[3]
                fy = params[4]
                ppx = params[5]
                ppy = params[6] 

set_guis()