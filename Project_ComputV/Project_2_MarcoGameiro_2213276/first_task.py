# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 19:40:04 2022

@author: marco
"""

import csv
import datetime 
import numpy as np
import math
import os
from welford import Welford
from time import process_time

os.chdir("C:\\VisComp\\PL\\Project_2_MarcoGameiro_2213276\\")

print(os.getcwd())

from control_exec import gui_metadata_files, aux_guis_first_menu 
from control_exec import control_stream

# ###############################################################################
# ############################ Compute depth map ################################

adit_info = aux_guis_first_menu()
  
name_csv_file = adit_info[0]
dist_infrared_cameras = adit_info[1]

# dist_infrared_cameras = 5    ## 5 cm

# name_csv_file = "data_task_one"


meta_files = gui_metadata_files()

meta_left = meta_files[0]
meta_right = meta_files[1]

meta_left += '.csv'
meta_right += '.csv'

             
def write_data_to_csv_file(data_lab, name_csv_file):
     with open(name_csv_file + '.csv', 'a', encoding='utf-8') as f:
         
         line = ', '.join(data_lab)
         f.write(line + '\n')  
         
         print("One more line written to csv file ...") 
         


def extract_params_infrared_metadata(metadata, chdir_folder):
    
##    os.chdir("C:\\VisComp\\PL\\Project_2\\")
    
##    base_folder = os.getcwd()

    base_folder = chdir_folder
    
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
    
     
    return k 
    

## meta_left = "CV_D435_IR_left_calib_Infrared_metadata.csv"    
## meta_right = "CV_D435_IR_right_calib_Infrared_metadata.csv"





def calculate_ssd(img1, img2):
    """Computing the sum of squared differences (SSD) between two images."""
    if img1.shape != img2.shape:
        print("Images don't have the same shape.")
        return
    return np.sum((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32))**2)


def plot_graph(values_along_time):
       import matplotlib.pyplot as plt
       
       x = []
       
       len_data = len(values_along_time)
       
       for i in range(0, len_data):
           x.append(i)
           
       plt.figure()
       
       plt.plot(x, values_along_time)
       plt.xlabel('Number of image')
       plt.ylabel('SSD Value')
       plt.title("SSD values along time")    
       plt.show()
       
       


#########################################################
##########################################################
##########################################################

# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import argparse for command-line options
import argparse
# Import os.path for file path manipulation
import os.path

"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")
# Parse the command line arguments to an object
args = parser.parse_args() 
# Safety if no parameter have been given
if not args.input:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()
# Check if the given file have bag extension

print("Args: " + args.input)

if os.path.splitext(args.input)[1] != ".bag":
    print("The given file is not of correct file format.")
    print("Only .bag files are accepted")
    exit()
    
#    C:\\VisComp\\PL\\Project_2\\
    
full_path = args.input    
path_full_parts = full_path.split('/')
path_full_parts = path_full_parts[:-1]

new_path = ''

for part in path_full_parts:
    new_path += part + '/' 
    
new_path.replace('/', "\\")    
    
k_left = extract_params_infrared_metadata(meta_left, new_path)
k_right = extract_params_infrared_metadata(meta_right, new_path)

print("k_left: " + str(k_left))
print("k_right: " + str(k_right))

fx = k_left[0][0]
fy = k_right[0][0] 
 
disparity = 0


try:
    # Create pipeline 
    pipeline = rs.pipeline() 

    # Create a config object 
    config = rs.config()
    # Tell config that we will use a recorded device from filem to be used by
    # the pipeline through playback (comment this line if you want to use a
    # real camera).
    rs.config.enable_device_from_file(config, args.input)
    width = 848
    height = 480 
    fps = 15
    # Configure the pipeline to stream the depth image 
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    # Configure the pipeline to stream both infrared images 
    config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
    config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
    # Configure the pipeline to stream the color image
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    # Start streaming from file
    profile = pipeline.start(config)

   
    # Create colorizer object
    colorizer = rs.colorizer()

    # Retreive the stream and intrinsic properties for all cameras
    profiles = pipeline.get_active_profile()
    streams = {"color" : profiles.get_stream(rs.stream.color).as_video_stream_profile(),
               "left"  : profiles.get_stream(rs.stream.infrared, 1).as_video_stream_profile(),
               "right" : profiles.get_stream(rs.stream.infrared, 2).as_video_stream_profile(),
               "depth" : profiles.get_stream(rs.stream.depth).as_video_stream_profile()
              }
    intrinsics = {"color" : streams["color"].get_intrinsics(),
                  "left"  : streams["left"].get_intrinsics(),
                  "right" : streams["right"].get_intrinsics(),
                  "depth" : streams["depth"].get_intrinsics(),
                 }
    extrinsics = {"color2left" : get_extrinsics(streams["color"], streams["left"]),
                  "right2left" : get_extrinsics(streams["right"], streams["left"])}
    # We can retrieve the extrinsic parameters from the camera itself, but
    # since we are using a bagfile, force the baseline. We are assuming the cameras
    # to be parallel and simply displaced along the X coordinate by the baseine
    # distance (in meters). 
    baseline = 0.005  # extrinsics["right2left"][1][0]

    # Obtain the depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is [m/px_val]: " , depth_scale)
    
    # prev_color_frame = np.zeros((480,848, 3))
    # disp_map = np.zeros((480,848, 3))
    
    counter = 0   
    
    mean_welford_1 = []
    std_welford_1 = []
    
    mean_welford_2 = []
    std_welford_2 = []
    
    mean_welford_3 = []
    std_welford_3 = []
    
    times_computed_depth_maps = []
    
    # ssd_values = []
    
    # counter_for_best_ssd = 0
    
    data_lab = ['Mean value (B Channel)', 'STD value (B Channel)',
                'Mean value (G Channel)', 'STD value (G Channel)',
                'Mean value (R Channel)', 'STD value (R Channel)',
                'Processing time (sec.)']
    
    name_csv_file = new_path + name_csv_file
    
    write_data_to_csv_file(data_lab, name_csv_file) 
    
    #################################################
    
    tup_n_images = control_stream(fps)
    
    (resp_str, n_images) = tup_n_images
    
    if resp_str == 'Yes': 
        n_max_images = n_images
    
    #################################################

    # Streaming loop
    while True:
        # Get frameset of depth
        frames = pipeline.wait_for_frames() 

        # Get depth frame
        depth_frame = frames.get_depth_frame() 

        # Colorize depth frame to jet colormap
        depth_color_frame = colorizer.colorize(depth_frame)

        # Convert depth_frame to numpy array to render image in opencv
        depth_color_image = np.asanyarray(depth_color_frame.get_data())

        # Render image in opencv window
        cv2.imshow("Depth Stream", depth_color_image)
        cv2.imwrite("deph_map_camera_" + str(counter) + ".jpg", depth_color_image)

        # Left IR frame
        ir_left_frame = frames.get_infrared_frame(1)
        ir_left_image = np.asanyarray(ir_left_frame.get_data())
        # Render image in opencv window
   ##     cv2.imshow("Left IR Stream", ir_left_image)

        # Right IR frame
        ir_right_frame = frames.get_infrared_frame(2)
        ir_right_image = np.asanyarray(ir_right_frame.get_data())
        # Render image in opencv window 
   ##     cv2.imshow("Right IR Stream", ir_right_image)
 
        # Color frame
        color_frame = frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        # Render image in opencv window
         
        imx = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)        
        
   ##     cv2.imshow("Color Stream", imx)
        
  ##      print("Shape of image: " + str(imx.shape)) 
        
        ## disparity map is a representation of distances as pixel intensities
            
 ##       left_image = cv2.cvtColor(ir_left_image, cv2.COLOR_BGR2GRAY)
 ##       right_image = cv2.cvtColor(ir_right_image, cv2.COLOR_BGR2GRAY)
 
        t1_start = process_time() 
 
        print('loading images...')
        imgL = cv2.pyrDown(ir_left_image)  # downscale images for faster processing        ## aloeL
        imgR = cv2.pyrDown(ir_right_image)

        # disparity range is tuned for 'aloe' image pair
        window_size = 3
        min_disp = 16                     ## 16
        num_disp = 112-min_disp            ## 112
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 16,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        print('computing disparity...')
        disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        
   #     print("Type of disp object: " + str(type(disp)))        
   #     
   #     disp_b = np.zeros((int(disp.shape[1]), int(disp.shape[0])))
   #     
   #     width = int(disp.shape[1])
   #     height = int(disp.shape[0])        
   #     dim = (width, height)
   #       
   #     disp_b = cv2.resize(disp, dim, interpolation = cv2.INTER_AREA)
   #     disp = disp_b
   #    
   #     print("Transpose of disparity done")
       

        print('generating 3d point cloud...',)
        h, w = imgL.shape[:2]
   ##     f = 0.8*w                          # guess for focal length
   
        f = fx
        
        Q = np.float32([[1, 0, 0, -0.5*w],
                        [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                        [0, 0, 0,     -f], # so that y-axis looks up
                        [0, 0, 1,      0]])
        points = cv2.reprojectImageTo3D(disp, Q)
        colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2RGB)
        mask = disp > disp.min()
        out_points = points[mask]
        out_colors = colors[mask]
        out_fn = 'out.ply'
  #      write_ply(out_fn, out_points, out_colors)
        print('%s saved' % out_fn)
 
   ##     cv2.imshow('left', imgL)
        cv2.imwrite("left_changed.jpg", imgL)
      

        dispar = np.zeros_like(disp)
        dispar = cv2.cvtColor((disp-min_disp)/num_disp, cv2.COLOR_GRAY2RGB)
        
        dispar_new = cv2.cvtColor(dispar, cv2.COLOR_RGB2BGR)
        
        cv2.imshow('disparity', dispar_new)
        
        
        cv2.imwrite("disparity_" + str(counter) + ".jpg", dispar_new)
        
        # dispar_new = np.zeros_like(disp)
        # dispar_new = (disp-min_disp)/num_disp

        # cv2.imwrite("disp_map_" + str(counter) + ".jpg", dispar_new)
        
        Z = np.zeros_like(dispar_new)
        
        if fx == fy: 
            f = fx  
             
            B = dist_infrared_cameras 
            
   ##         for c in range(0,3):
            
            if True:
                for c in range(0,3):
                    for b in range(len(dispar_new[0])):
                        for a in range(len(dispar_new)):
                            if dispar_new[a,b,c] > 0:
                                Z[a,b,c] = (B*f)/dispar_new[a,b,c]               
            
        else:
            print("Focal lenght from x axis is different than the one from y axis")
            
        Z_res = np.zeros((2*len(Z),2*len(Z[0]),3))
            
 ##       Z_res = cv2.resize(Z, (2*len(Z[0]),2*len(Z),3), interpolation = cv2.INTER_AREA)
 
        for c in range(0,3):
            Z_res[:,:,c] = cv2.resize(Z[:,:,c], (2*len(Z[0]),2*len(Z)), interpolation = cv2.INTER_AREA)
        
        import matplotlib.pyplot as plt
        from PIL import Image
        
        
        print("Z_res shape: " + str(Z_res.shape)) 
        
        
#         cv2.imwrite("Z_res_aux.png", Z_res)
        
#  ##       Z_res = cv2.cvtColor(Z_res, cv2.COLOR_BGR2GRAY)
# ##        Z_res = Z_res[:,:,2]

#         Z_res = cv2.imread("Z_res_aux.png")
#         Z_res = cv2.cvtColor(Z_res, cv2.COLOR_BGR2GRAY)   

        
        # for c in range(0,3):
        #     for b in range(len(Z_res[0])):
        #         for a in range(len(Z_res)):
        #             Z_res[a,b] *= 4
                    
        cv2.imwrite("Z_res_aux.png", Z_res)
        
        Z_res = cv2.imread("Z_res_aux.png")
        Z_res = cv2.cvtColor(Z_res, cv2.COLOR_BGR2GRAY)                   
                    
        Z_res = cv2.applyColorMap(Z_res, cv2.COLORMAP_JET)
        
        Z_res_new = Z_res.copy()
        
        Z_res_new[:,:,0] = Z_res[:,:,2]
        Z_res_new[:,:,2] = Z_res[:,:,0]
        
        Z_res = Z_res_new
        
   ##     Z_res = cv2.bitwise_not(Z_res)
        
  #      Z_res = cv2.LUT(im_gray, lut)
       
        
 ##       Z_res = cv2.cvtColor(Z_res, cv2.COLOR_BGR2HSV)

        # # Get the color map by name:  
        # cm = plt.get_cmap('gist_rainbow')
        
        # # Apply the colormap like a function to any array:
        # colored_image = cm(Z_res) 
        
        # # Obtain a 4-channel image (R,G,B,A) in float [0, 1]
        # # But we want to convert to RGB in uint8 and save it:
        # Image.fromarray((colored_image[:, :, :3] * 255).astype(np.uint8)).save("depth_map_" + str(counter) + ".jpg")
        
        # Z_res = (colored_image[:, :, :3] * 255).astype(np.uint8) 
        
        
  #      Z_res = cv2.cvtColor(Z_res, cv2.COLOR_GRAY2RGB) 
        
        cv2.imwrite("depth_map_" + str(counter) + ".jpg", Z_res)
 
        
        cv2.imshow('Computed depth map', Z_res)
        
        print(Z_res.shape)
             
        t1_stop = process_time()   
        
        time_depth_map = abs(t1_stop-t1_start)
        print("Time to compute depth map for image " + str(counter) + " : " + str(time_depth_map) + " seconds")
        
        times_computed_depth_maps.append(time_depth_map)
        
       
        
  #      depth_color_image_grey = cv2.cvtColor(depth_color_image, cv2.COLOR_BGR2GRAY)
        
        print("Shape 1" + str(depth_color_image.shape))
 
        diff_depths = np.zeros_like(depth_color_image)
        
        print("Shape 2" + str(Z_res.shape))
        
        depths_bufList = []
        
        w_1 = Welford()
        w_2 = Welford()
        w_3 = Welford()        
        
        for c in range(0,3):
            for b in range(len(depth_color_image[0])):
                for a in range(len(depth_color_image)):
                    
                    if depth_color_image[a,b,c] != 0:
                        
                        diff_val = abs(Z_res[a,b,c] - depth_color_image[a,b,c])
                        
                        if diff_val <= 0:
                            diff_val = 0
                        if diff_val >= 255:
                            diff_val = 255
                            
                        depths_bufList.append(diff_val)
                        
                        if c == 0:
                            w_1.add(np.array([diff_val]))
                        elif c == 1:
                            w_2.add(np.array([diff_val]))
                        elif c == 2:
                            w_3.add(np.array([diff_val]))                                
                        
            
        diff_depths_buffer = np.array([depths_bufList])
        
        ######################################################################       
        ## Welford algorithm   
        
   ##     w_1.add(diff_depths_buffer)      
      
        
        print("Comparison between computed depth map and the one given by camera: ")
        
        mean_par = w_1.mean
        mean_par = mean_par[0]
        
        std_par = np.sqrt(w_1.var_s)
        std_par = std_par[0]
        
        mean_par_2 = w_2.mean
        mean_par_2 = mean_par_2[0]
        
        std_par_2 = np.sqrt(w_2.var_s)
        std_par_2 = std_par_2[0]
        
        mean_par_3 = w_3.mean
        mean_par_3 = mean_par_3[0]
        
        std_par_3 = np.sqrt(w_3.var_s) 
        std_par_3 = std_par_3[0]         
        
        
        print("-- Mean (B channel): " + str(mean_par))
        print("-- STD Value (B channel): " + str(std_par))
        
        print("-- Mean (G channel): " + str(mean_par_2))
        print("-- STD Value (G channel): " + str(std_par_2))
        
        print("-- Mean (R channel): " + str(mean_par_3))
        print("-- STD Value (R channel): " + str(std_par_3))
        
        mean_welford_1.append(mean_par)
        std_welford_1.append(std_par)

        mean_welford_2.append(mean_par_2)
        std_welford_2.append(std_par_2) 

        mean_welford_3.append(mean_par_3)
        std_welford_3.append(std_par_3)  
        
        ######################################################################        
        
        data_lab = [str(mean_par), str(std_par),
                    str(mean_par_2), str(std_par_2),
                    str(mean_par_3), str(std_par_3),
                    str(time_depth_map)] 
        
        write_data_to_csv_file(data_lab, name_csv_file) 
        
        ######################################################################
        
        counter += 1
        
        key = cv2.waitKey(1)
        # if pressed escape exit program
        if key == 27 or (resp_str == 'Yes' and counter == n_max_images):    
            
            mean_time_computing = np.mean(np.array([times_computed_depth_maps]))
            print("Mean time to compute a depth map: " + str(mean_time_computing) + " seconds")
            
            cv2.destroyAllWindows()
            break 

finally:
    pass





