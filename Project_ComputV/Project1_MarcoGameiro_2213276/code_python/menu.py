# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:02:00 2022

@author: marco
"""

import os
import sys
## import PySimpleGUIWeb as sg
import PySimpleGUI as sg


print(os.getcwd())

## sys.path.append('C:/VisComp/PL/Project/Modules')

# import first_part
# import second_part
# import third_part 
# import fourth_part
# # from subprocess import Popen

this_dir = os.getcwd()
foldername_string = "CVis_2223_Assign1_MarcoGameiro"
dir_parts = this_dir.split(foldername_string)
base_dir = dir_parts[0] + foldername_string + "\\" 


col = sg.Column([     
    [sg.Frame(layout=[    ##  key = '-text-'    ##  [sg.Text('Tasks:', font='bold')],
                      [sg.Button('Segmentation of the Boccia balls applying grayscale conversion', font='italic')],  ## key = '-topic-'
                      [sg.Button('Segmentation of the Boccia balls applying color space conversion', font='italic')],
                      [sg.Button('Distance measurement estimation regarding the camera', font='italic')]],    ## font='bold'                     
              title='Tasks Panel')],
    
    [sg.Frame(layout=[[sg.Button('The real challenge: Run the algorithm in a video', font='italic')]]
                      , title='Bossia Video Panel:')]])
 
layout = [ [col]]

## window = sg.Window('Menu', layout, web_port=2222, disable_close=True, finalize = True)  
window = sg.Window('Menu', layout, disable_close=True, finalize = True)   ## modal=True


# title = window['-text-']
# topic = window['-topic-']
 
# title.update(font='bold')
# topic.update(font='italic')First_Task

base_path = base_dir + "code_python\\"


while True:
    
    event, values = window.read()
    print(event, values)   
    
    if event == "Exit" or event == sg.WIN_CLOSED:
        break    
    else:        
        
        # title.update(font='bold')
        # topic.update(font='italic')
    
        if event == 'Segmentation of the Boccia balls applying grayscale conversion':
            print("Segmentation of the Boccia balls applying grayscale conversion")
            
            name_folder = input('\nMain folder name: ')
            
       ##     input_image_filename = input('\nInput image filename: ')
       
            input_image_filename = ""
            
            while True:
                
                input_image_filename = input('\nInput image filename: ')
            
                if ".jpg" in input_image_filename or ".jpeg" in input_image_filename:
                    break
                else:
                     
                    if not input_image_filename:
                        print("\nEmpty field\n Try again")
                    else:
                    
                        if "." in input_image_filename:
                            print("\nThe extension of the filename has to be one of the following: \n {.jpg, \t .jpeg}")
                            
                        else:                
                            print("\nFilename should contain the extension")
                            
            
            output_image_filename = ""
            
            while True:
                
                output_image_filename = input('\nOutput image filename: ')
            
                if ".jpg" in output_image_filename or ".jpeg" in output_image_filename:
                    break
                else:
                     
                    if not output_image_filename:
                        print("\nEmpty field\n Try again")
                    else:
                    
                        if "." in output_image_filename:
                            print("\nThe extension of the filename has to be one of the following: \n {.jpg, \t .jpeg}")
                            
                        else:                
                            print("\nFilename should contain the extension")        
            
                            
            x = os.system('python ' + base_path + 'first_part.py ' + name_folder + " " + input_image_filename + " " + output_image_filename)
            
                        
            if x == 0:       ## means success
                window.close()
            else:
                print("First event fails to start")
                
                
        if event == 'Segmentation of the Boccia balls applying color space conversion':
            print("Segmentation of the Boccia balls applying color space conversion")            
            
            name_folder = input('\nMain folder name: ')
            
       ##     input_image_filename = input('\nInput image filename: ')
       
            input_image_filename = ""
            
            while True:
                
                input_image_filename = input('\nInput image filename: ')
            
                if ".jpg" in input_image_filename or ".jpeg" in input_image_filename:
                    break
                else:
                     
                    if not input_image_filename:
                        print("\nEmpty field\n Try again")
                    else:
                    
                        if "." in input_image_filename:
                            print("\nThe extension of the filename has to be one of the following: \n {.jpg, \t .jpeg}")
                            
                        else:                
                            print("\nFilename should contain the extension")
                            
            
            output_image_filename = ""
            
            while True:
                
                output_image_filename = input('\nOutput image filename: ')
            
                if ".jpg" in output_image_filename or ".jpeg" in output_image_filename:
                    break
                else:
                     
                    if not output_image_filename:
                        print("\nEmpty field\n Try again")
                    else:
                    
                        if "." in output_image_filename:
                            print("\nThe extension of the filename has to be one of the following: \n {.jpg, \t .jpeg}")
                            
                        else:                
                            print("\nFilename should contain the extension")        
            
            
            x = os.system('python ' + base_path + 'second_part.py ' + name_folder + " " + input_image_filename + " " + output_image_filename)
            
            print('python ' + base_path + 'second_part.py ' + name_folder + " " + input_image_filename + " " + output_image_filename)
            
            if x == 0:       ## means success
                window.close()
            else:
                print("Second event fails to start")
                
        
        if event == 'Distance measurement estimation regarding the camera':
            print("Distance measurement estimation regarding the camera") 
            
            
            dir_imx = ""
            
            while True:
                
                dir_imx = input('\nInsert full directory: ')    ## fill in the browser
            
                if ".png" in dir_imx:
                    break
                else:
                     
                    if not dir_imx:
                        print("\nEmpty field\n Try again")
                    else:
                    
                        if "." in dir_imx:
                            print("\nThe extension of the filename has to be one of the following: \n {.png}")
                            
                        else:                
                            print("\nFilename should contain the extension")        
            
            
            
            dist_redBall_camera = input('\nDistance between red ball and camera (cm): ')
             
            real_diam_red_ball = input('\nDiameter of red ball in real life units (cm): ')
        
            
            x = os.system('python ' + base_path + 'third_part.py ' + dir_imx + " " + str(dist_redBall_camera) + " " + str(real_diam_red_ball))             
            
            print('python ' + base_path + 'third_part.py ' + dir_imx + " " + str(dist_redBall_camera) + " " + str(real_diam_red_ball))
            
            if x == 0:       ## means success
                window.close()
            else:
                print("Third event fails to start")
            
            
        if event == 'The real challenge: Run the algorithm in a video':
            print("The real challenge: Run the algorithm in a video")
            
            os.system('python ' + base_path + 'bossia_video_vc.py')  
            
        
window.close()



        




                          


