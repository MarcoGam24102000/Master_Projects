
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

import datetime
import csv

from second_task_by_menu import second_task_sec 
from control_exec import control_stream
from eval_prec_sec import estim_prec
import time

"""
Returns R, T transform from src to dst
"""
def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(extrinsics.rotation, [3,3]).T
    T = np.array(extrinsics.translation)
    return (R, T)


# def list_out_ransac(clone, winH, winW):
#     from sklearn.linear_model import LinearRegression, RANSACRegressor
    
#     edge_pixels = []
#     n = 6
    
#     for i in range(0,winH+1):
#         edge_pixels.append(clone[x, y+i])
    
#     for j in range(0,winW+1):
#         edge_pixels.append(clone[x+j, y]) 
    
#     for j in range(0,winW+1):
#         edge_pixels.append(clone[x+j, y+WinH])
    
#     for i in range(0,winH+1):
#         edge_pixels.append(clone[x+WinW, y+i])
        
#     edge_pix_sel = random.sample(edge_pixels, n)
    
#     pos_edges_sel = []
    
#     for ed in edge_pix_sel:
#         pos = np.where(np.array([edge_pixels]) == ed)
#         pos_edges_sel.append(pos)
    
#     ransac = RANSACRegressor(base_estimator=LinearRegression(),
#                           min_samples=50, max_trials=100,
#                           loss='absolute_loss', random_state=42,
#                           residual_threshold=10)
#     #
#     # Fit the model
#     #
#     ransac.fit(pos_edges_sel, edge_pix_sel)
    
#     inlier_mask = ransac.inlier_mask_
#     outlier_mask = np.logical_not(inlier_mask)
    
#     # Create scatter plot for inlier datset
#     #
#     plt.figure(figsize=(8, 8))
#     plt.scatter(X[inlier_mask], y[inlier_mask],
#                 c='steelblue', edgecolor='white',
#                 marker='o', label='Inliers')
#     #
#     # Create scatter plot for outlier datset
#     #
#     plt.scatter(X[outlier_mask], y[outlier_mask],
#                   c='limegreen', edgecolor='white',
#                   marker='s', label='Outliers') 
#     #
    


# def analyse_image_window(clone, winH, winW):  
    
#           import random
#           from sklearn.linear_model import LinearRegression
    
#           edge_pixels = []
#           n = 6
          
#           for i in range(0,winH+1):
#               edge_pixels.append(clone[x, y+i])
          
#           for j in range(0,winW+1):
#               edge_pixels.append(clone[x+j, y]) 
          
#           for j in range(0,winW+1):
#               edge_pixels.append(clone[x+j, y+WinH])
          
#           for i in range(0,winH+1):
#               edge_pixels.append(clone[x+WinW, y+i])
              
#           edge_pix_sel = random.sample(edge_pixels, n)
          
#           pos_edges_sel = []
          
#           for ed in edge_pix_sel:
#               pos = np.where(np.array([edge_pixels]) == ed)
#               pos_edges_sel.append(pos)
          
#           lr = LinearRegression()    
#           lr.fit(pos_edges_sel, edge_pix_sel)
          
#           w = lr.coef_[0]
          
#           list_out = w*pos_edges_sel
          
#           return list_out      



# def find_elipses(grey_image):
    
#     # Find canny edges    
#     size_filter = (5,5)    
#     blurred = cv2.GaussianBlur(grey_image, size_filter, 0)    
#     mid = cv2.Canny(blurred, 30, 150)
    
#     # Sliding window    
#     from deepRed.pyimagesearch.helpers import pyramid
#     from deepRed.pyimagesearch.helpers import sliding_window    
    
#     (winW, winH) = (128, 128)    
 
#     for resized in pyramid(mid,scale=1.5):
#         for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
#             if (window.shape[0] != winH) or (window.shape[1] != winW):
#                 clone = resized.copy()
#                 list_out = analyse_image_window(clone, winH, winW)
#                 list_out_ransac(clone, winH, winW)
#                 cv2.rectangle(clone, (x,y), (x+winW, y+winH), (0,255,0), 2)
#                 cv2.imshow("Window", clone)
#                 cv2.waitkey(0)
#                 time.sleep(0.025)
    

def mog_bg_sub(rgb_image):
    backSub = cv2.createBackgroundSubtractorMOG2()
    
 ##   backSub = cv2.createBackgroundSubtractorKNN()


    # Update the background model
    fgMask = backSub.apply(rgb_image)

    # Get the frame number and write it on the current frame 
    cv2.rectangle(rgb_image, (10, 2), (100, 20), (255, 255, 255), -1)
##    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
##               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('FG Mask', fgMask)
     
    return fgMask

def getNumberContoursForImage(bin_image):
    cnts = cv2.findContours(bin_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     ## canny
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    imp_cnts = 0
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        
        if w> 150 and h> 150:
            imp_cnts += 1
    
    return imp_cnts

def track_move(diff_mog_image, prev_diff, width, height):
    
    white_pix_upper_actual = 0
    white_pix_upper_prev = 0
    white_pix_lower_actual = 0
    white_pix_lower_prev = 0
    
    for b in range(0, width):
        for a in range(0, height):
            if a >= int(width/2):
                if diff_mog_image[a,b] == 255:
                    white_pix_upper_actual += 1
                if prev_diff[a,b] == 255:
                    white_pix_upper_prev += 1
            else:
                if diff_mog_image[a,b] == 255:
                    white_pix_lower_actual += 1
                if prev_diff[a,b] == 255:
                    white_pix_lower_prev += 1 
    
    if (white_pix_upper_actual - white_pix_upper_prev) > 0 :    
        move = 'in'
    elif (white_pix_lower_prev - white_pix_lower_actual) < 0:   
        move = 'out'
    else:
        move = 'none'
        
    print(move)
     
    return move 

# Create object for parsing command-line options
parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                Remember to change the stream resolution, fps and format to match the recorded.")
# Add argument which takes path to a bag file as an input
parser.add_argument("-i", "--input", type=str, help="Path to the bag file")

parser.add_argument("-pi", "--peopleInside", type=str, help="Initial number of people inside lab")
parser.add_argument("-csv", "--csvFilename", type=str, help="Name of csv file, without extension, where data will be written")

# Parse the command line arguments to an object
args_all = parser.parse_args()
# Safety if no parameter have been given

args_all = list(vars(args_all).items())


if len(args_all) < 3:
    print("Too few given arguments")
elif len(args_all) > 3:
    print("Too much given arguments")
else:
    
    fps = 15
    
    exec_times_list = []
    
    input_par = args_all[0]
    (name_p, input_p) = input_par
     
    print("Input: " + str(input_p))
    
    init_people_inside = args_all[1]
    (name_p_2, init_people_within) = init_people_inside
    
    init_people_within = int(init_people_within)
    
    print("Init people inside: " + str(init_people_within))
    
    csv_filename = args_all[2]
    (name_p_3, name_csv_file) = csv_filename
    
    print("CSV Filename: " + str(csv_filename))

    args = args_all[0]      
    
    
    # if not input_p:
    #     print("No input paramater have been given.")
    #     print("For help type --help") 
    #     exit()
    
    # Check if the given file have bag extension
    if os.path.splitext(input_p)[1] != ".bag": 
        print("The given file is not of correct file format.")
        print("Only .bag files are accepted")
        exit()
        
    dir_path = input_p
    
    print("Dir Path: " + dir_path)  
    
    if '/' in dir_path:
        dir_parts = dir_path.split('/')
    elif "\\" in dir_path:
        dir_parts = dir_path.split("\\")
    
    dir_parts = dir_parts[:-1]
    
    dir_folder = ""
     
    for d in dir_parts:
        dir_folder += d + '/'    
        
    print("Dir Folder: " + dir_folder)
    
    #################################################
    
    tup_n_images = control_stream(fps)
    
    (resp_str, n_images) = tup_n_images
    
    if resp_str == 'Yes': 
        n_max_images = n_images
    
    #################################################
    
    
    
    data = []
##    name_csv_file = "data_lab"   
    
    ###############################################################################
    ## Compute first line of CSV file #############################################
    
    number_group = 1
    number_student = 2213276
    name_student = "Marco Gameiro"
    
    group_str = '# ' + 'Group ' + str(number_group)
    
    numberStudent_str = str(number_student)
    
    student_info = [group_str, numberStudent_str, name_student]
    
    
    def write_data_to_csv_file(name_csv_file, first_line, sec_line):
        with open(name_csv_file + '.csv', 'a', encoding='utf-8') as f:    ## w
           
            line = ', '.join(first_line)
            f.write(line + '\n')
            
            line = ', '.join(sec_line)
            f.write(line + '\n')       
            
                
    def write_data_to_csv_file_body(data_lab, name_csv_file):
        with open(name_csv_file + '.csv', 'a', encoding='utf-8') as f:
            
            line = ', '.join(data_lab)
            f.write(line + '\n') 
    
    
    ####################################################################################
    
    name_csv_file = dir_folder + name_csv_file 
    
    try:
        # Create pipeline
        pipeline = rs.pipeline() 
    
        # Create a config object
        config = rs.config()
        # Tell config that we will use a recorded device from filem to be used by
        # the pipeline through playback (comment this line if you want to use a
        # real camera).
        rs.config.enable_device_from_file(config, dir_path)
        width = 848
        height = 480
       
        # Configure the pipeline to stream the depth image
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        # Configure the pipeline to stream both infrared images
        config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        config.enable_stream(rs.stream.infrared, 2, width, height, rs.format.y8, fps)
        # Configure the pipeline to stream the color image
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    
        # Start streaming from file
        profile = pipeline.start(config)
    
        # Create opencv window to render image in
        # Uncomment if you want to be able to resize the window. The KEEPRATIO
        #parameters might not work.
        # cv2.namedWindow("Depth Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.namedWindow("Left IR Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.namedWindow("Right IR Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.namedWindow("Color Stream", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO )
        
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
        
        counter = 0
        lim_bin = 50
        
        track_p = False     
        
        frame_comp = np.zeros((480, 848))
        diff_mog_image = np.zeros((480, 848))
        prev_diff = np.zeros((480, 848))
         
        inter_count = 0
        
        people_in = 0
        people_out = 0
        n_people_lab = init_people_within   ## 0 
        
        ###############################################################################
        ## Compute second line of CSV file ############################################
    
        now = datetime.datetime.now()
        str_now = str(now)
        actual_time = str_now[11:19]  
        
        print("Actual time: " + actual_time)
    
        start_program = [actual_time, 'none', str(n_people_lab)]
        
        write_data_to_csv_file(name_csv_file, student_info, start_program)  
    
        ###############################################################################
        
        counter_in_outs = 0
        set_moves = []
        prec_list = []    
    
        moves_hough = []
        
        counter_for_meas = 0       
        
        circle_radixes = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160]           
            
        ind_circ_radix = 0 
        
        # Streaming loop
        
        centroid_prev = (0,0)
        counter_rep = 0
        
        other = False
        
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
 #           cv2.imshow("Depth Stream", depth_color_image)
    
            # Left IR frame
            ir_left_frame = frames.get_infrared_frame(1)
            ir_left_image = np.asanyarray(ir_left_frame.get_data())
            # Render image in opencv window
  #          cv2.imshow("Left IR Stream", ir_left_image)
    
            # Right IR frame
            ir_right_frame = frames.get_infrared_frame(2)
            ir_right_image = np.asanyarray(ir_right_frame.get_data())
            # Render image in opencv window
 #           cv2.imshow("Right IR Stream", ir_right_image)
    
            # Color frame
            color_frame = frames.get_color_frame()
            color_image = np.asanyarray(color_frame.get_data())
            # Render image in opencv window
            cv2.imshow("Color Stream", cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            
            startTime = time.time()          ## Start time
            
         ##   frame_result = mog_bg_sub(cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR))
            frame_result = mog_bg_sub(depth_color_image)   
            
            if counter >= 1:
            
                for b in range(0, width):
                    for a in range(0, height):
                        diff_mog_image[a,b] = abs(frame_result[a,b] - frame_comp[a,b]) 
                        
                        if diff_mog_image[a,b] > lim_bin:
                            diff_mog_image[a,b] = 255       ## Binarization
                        
                cv2.imwrite(dir_folder + "frame_result_" + str(counter) + ".png", diff_mog_image) 
                
                diff_mog_image_read = cv2.imread(dir_folder +  "frame_result_" + str(counter) + ".png")
                
                diff_mog_image = cv2.cvtColor(diff_mog_image_read, cv2.COLOR_BGR2GRAY)           
                
                len_cnts = getNumberContoursForImage(diff_mog_image)
                
                print("------")
                print("Number of contours detected: " + str(len_cnts))
                
                
                if track_p == True:
                    inter_count += 1
                
                if inter_count == 2:
                    inter_count = 0
                    track_p = False
                
                print("Hear")                   
                 
                
                # Assume binary_mask is a 2D numpy array with dtype bool
                centroid = np.mean(np.argwhere(diff_mog_image),axis=0)
                centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                
                print("Centroid: (" + str(centroid_x) + " , " + str(centroid_y) + ")")        
               
                
    
                if len_cnts > 0:        ##  and track_p == False
                    
                    move_direc = "none"
                    
                    msg = "none"                
                    
                    if True :      ## if not here check below
                        
                        counter_rep += 1
                        
                        if counter_rep == 1:
                        
                            print("Move detected")
                            
                            print("Centroid prev: (" + str(centroid_prev[0]) + " , " + str(centroid_prev[1]) + ")")
                            
                            if centroid_y < 220:
                                print("No move yet")
                                other = True
                            else:
                            
                                if (centroid_y - centroid_prev[1]) + (centroid_x - centroid_prev[0]) < 0 :
                                    print("Move is in")
                                    msg = "in"
                                elif (centroid_y - centroid_prev[1]) + (centroid_x - centroid_prev[0]) > 0:
                                    print("Move is out")
                                    msg = "out"
                                
                            if msg == "in" or msg == "out":                        
                                     
                                
                                    set_moves.append(msg) 
                                    
                                    move_direc = msg                                    
                                    
                            
                                    print("Length set moves: " + str(len(set_moves)))
                      
                                
                            
                            if len(set_moves) == 16:
                           ##     if counter_for_meas == 16:
                               
                                prec = estim_prec(set_moves) 
                                prec_list.append(prec)
                                set_moves = []
                                ind_circ_radix += 1
                                counter_for_meas = 0 
                                
                        elif counter_rep == 2:
                            counter_rep = 0   
                                
                    if other == True:                                  
                    
                        other = False
                        not_white_dots = False
                         
                        
                        print("A")
                        
                        output = diff_mog_image_read.copy() 
                        
                        print("B")
                        
                       ######################################################################################
                       ################## Try to find circles within image ##################################
                       # detect circles in the image                  
                        
    
                       
                        circles = cv2.HoughCircles(diff_mog_image, cv2.HOUGH_GRADIENT, 1.2, 45)  ## 100
                        circles_sec = cv2.HoughCircles(diff_mog_image, cv2.HOUGH_GRADIENT, 1.2, 85)  ## 100
                        
                        count_circles = 0
                        
                        # ensure at least some circles were found
                        if circles is not None and circles_sec is not None:                        
                        # convert the (x, y) coordinates and radius of the circles to integers
                        	circles = np.round(circles[0, :]).astype("int") 
     #                       circles_sec = np.round(circles_sec[0, :]).astype("int")                          
                            
                            
                        	# loop over the (x, y) coordinates and radius of the circles
                            
                        	for (x, y, r) in circles: 
                                
                                 if y > len(diff_mog_image)/5:   ## and x < (2/3)*len(diff_mog_image)
                                
                                    if x < len(diff_mog_image[0]):  
                                        move = 'out'
                                    else:
                                        move = 'in' 
                                    
                                    move_direc = move                                   
                                    
                                    
                                    not_white_dots = True
                                        
                                    print("Move detected by Hough transform: " + move)
                                        
                                        
                            		# draw the circle in the output image, then draw a rectangle 
                            		# corresponding to the center of the circle 
                                    
                                    count_circles += 1
                                    
                                    if count_circles <= 2:
                                        
                                        cv2.circle(output, (x, y), r, (0, 255, 0), 4)                      
                                        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
                                    
                                    
                                    
                                    moves_hough.append(move) 
                                    
                        	# show the output image
                   #     	cv2.imshow("output", np.hstack([diff_mog_image_read, output])) 
                   #     	cv2.waitKey(0)   
                        else:
                            print("No circles detected") 
                            
                        if not_white_dots == True:
                            
                            cv2.imshow("Mov. Det", output)
                            
                        ######################################################################################
                        ################## Try to find elipses within image ##################################
                         
                        # number_elipses = find_elipses(diff_mog_image)       
                        
                        # print(str(number_elipses) + " Elipses detected ...") 
                        
                        if not_white_dots == False:
                        
                       
                            move_direc = track_move(diff_mog_image, prev_diff, width, height)
                                       
                        
                        
                        executionTime = (time.time() - startTime)
                        print('Execution time in seconds: ' + str(executionTime))
                        
                        exec_times_list.append(executionTime)
                        
                        if ('in' in move_direc) or ('out' in move_direc):
                            counter_in_outs += 1 
                            
                            msg = ''
                            
                            if 'in' in move_direc:
                                msg = 'In'
                            elif 'out' in move_direc: 
                                msg = 'Out' 
                            
                        
                            if msg == "in" or msg == "out":                        
                                    
                                    set_moves.append(msg)  
                            
                                    print("Length set moves: " + str(len(set_moves)))
                            
                            if len(set_moves) == 16:
                           ##     if counter_for_meas == 16:
                               
                                prec = estim_prec(set_moves) 
                                prec_list.append(prec)
                                set_moves = []
                                ind_circ_radix += 1
                                counter_for_meas = 0  
                            
                               
                                
                            # if len(moves_hough) == 16:
                            
                            #     prec = estim_prec(moves_hough)  
                            #     prec_list.append(prec)
                            #     moves_hough = []                             
                                
                             
                    if move_direc != "none":        
                    
                        if 'in' in move_direc:
                             people_in += 1
                             n_people_lab += 1
                        elif 'out' in move_direc:
                             people_out += 1
                             n_people_lab -= 1
                                
                        now = datetime.datetime.now()
                        str_now = str(now)
                        actual_time = str_now[11:19]  
                            
                        data_lab = [actual_time, str(move_direc), str(n_people_lab)]
                        write_data_to_csv_file_body(data_lab, name_csv_file)
                            
                        if resp_str != 'Yes':
                              n_max_images = 5000
                            
                        people_within = second_task_sec(dir_path, people_in, people_out, n_people_lab, counter, n_max_images)
                        
                        track_p = True 
                
                prev_diff = diff_mog_image.copy() 
                
                counter_for_meas += 1
                
                centroid_prev = (centroid_x, centroid_y)   
            
            counter += 1 
            
            if resp_str == 'Yes':
                
                print("Current image: " + str(counter))
                
                print("Max Images: " + str(n_max_images)) 
            
                if counter == n_max_images: 
                ##    cv2.destroyAllWindows()                 
                    got_move = False
                    break
                else:
                        
                    frame_comp = frame_result.copy()                 
                        
                    key = cv2.waitKey(1) 
                    
            else:
        
                frame_comp = frame_result.copy()                 
                    
                key = cv2.waitKey(1) 
                
            # if pressed escape exit program  
            if key == 27 or (resp_str == 'Yes') and  (counter == n_max_images):   ##               (len(prec_list) == len(circle_radixes)
                ## (ind_circ_radix == len(circle_radixes)-1)
                prec_mean = np.mean(np.array([prec_list]))
                
                print("Mean precision: " + str(prec_mean) + " % ...") 
                
                prec_high = np.max(np.array([prec_list]))
                
                pos_higher_prec = np.where(np.array([prec_list]) == prec_high)[0].tolist()
                
                for p in pos_higher_prec:                
                    print("Higher precision, of about " + prec_high + " % at: " + str(p))
                 
                cv2.destroyAllWindows() 
                break
            
    finally:
        
        mean_time_per_image = np.mean(np.array([exec_times_list]))
        print("Mean per image: " + str(mean_time_per_image) + " seconds")
        
        pass





