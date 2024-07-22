# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 23:29:47 2022

@author: marco
"""

import cv2
import numpy as np

image = cv2.imread("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\data\\Boccia_balls.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def find_dist(gray, image):
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) #adaptive
    
    
    masked = cv2.bitwise_and(image, image, mask=thresh)
    ## cv2.imwrite('Masked.png', masked)
    
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    
    ##cv2.imwrite('Canny_image.png', canny)
      
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
    
    dims= []
    
    
    def find_dist_bet_centers(centerFirst, centerSec):
        first_parcel = (centerFirst[0] - centerSec[0])**2
        sec_parcel = (centerFirst[1] - centerSec[1])**2
        
        result_dist = np.sqrt(first_parcel + sec_parcel)
        
        return result_dist
         
        
    
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # x,y,w,h = 37, 625, 309, 28  
        ROI = thresh[y:y+h,x:x+w]
        
        if w > 50 and w < 500 :
            if h > 50 and h < 500:
                ## code to not create bounding boxes inside another already in that place
                if abs(w-h) <= 15 or abs(w-h) >= 25:
                    
                    if h > 50:
                        w = h
                        
                    gray_roi = gray[y:y+h,x:x+w]
                    
                    white_points = 0
                    
                    for b in range(len(gray_roi[0])):
                        for a in range(len(gray_roi)):
                            if gray_roi[a,b] > 200.:
                                white_points += 1
                    
        ##            print("Number of white points: " + str(white_points))
                      
                    if white_points < 0.15*(len(gray_roi[0])*(len(gray_roi))):
                        ## if not, move to the left until the requirement is satisfied
                        
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
                                ## cv2.imwrite('BOSSIA_BALL_ROI_CIRCLE_{}.png'.format(ROI_number), imCircle) 
                                
                                # draw filled circle in white on black background as mask
                                mask_circle = np.zeros_like(image)
                                mask_circle = cv2.circle(mask_circle, (int(x+w/2), int(y+h/2)), radius, (255,255,255), -1)
                                
                                result = cv2.bitwise_and(image, mask_circle)
                                
                                result_rois.append(result)
                                dims.append([x,y,w,h])
                                
                                ## cv2.imwrite('SEGMENTED_ROI_{}.png'.format(ROI_number), result) 
                                
                              ##  del imLines 
                        
                         ##       cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                               ## cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
                                ROI_number += 1
                        else:
                            dimensions_rois.append([x, y, w, h])
            
                            imLines = cv2.line(image, (x,y), (x+w,y), color, thickness)
                            imLines = cv2.line(imLines, (x+w,y), (x+w,y+h), color, thickness)
                            imLines = cv2.line(imLines, (x+w,y+h), (x,y+h), color, thickness)
                            imLines = cv2.line(imLines, (x,y+h), (x,y), color, thickness)
                            
                          ##  cv2.imwrite('BOSSIA_BALL_ROI_{}.png'.format(ROI_number), imLines) 
                            
                            dim_imLines = np.array([w, h]) 
                            
                            radius =  int((np.min(dim_imLines))/2)
                            
                            imCircle = cv2.circle(imLines, (int(x+w/2), int(y+h/2)), radius, color, thickness)
                         ##   cv2.imwrite('BOSSIA_BALL_ROI_CIRCLE_{}.png'.format(ROI_number), imCircle) 
                            
                            # draw filled circle in white on black background as mask
                            mask_circle = np.zeros_like(image)
                            mask_circle = cv2.circle(mask_circle, (int(x+w/2), int(y+h/2)), radius, (255,255,255), -1)
                            
                            result = cv2.bitwise_and(image, mask_circle)
                            
                            result_rois.append(result)
                            dims.append([x,y,w,h])
                            
                        ##    cv2.imwrite('SEGMENTED_ROI_{}.png'.format(ROI_number), result)
                            
                          ##  del imLines 
                    
                     ##       cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
                       ##     cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
                            ROI_number += 1
                            
    black_level = 117
    
    shadows_image = np.zeros_like(gray)
    
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
                   
    dist = 0
    
    if len(dims) > 2:
        print("More than 2 balls detected")   
    elif len(dims) == 2:
        dims_ball_one = dims[0]     
        dims_ball_two = dims[1] 
        
        center_one = (int(dims_ball_one[0] + dims_ball_one[2]/2), int(dims_ball_one[1] + dims_ball_one[3]/2))
        center_two = (int(dims_ball_two[0] + dims_ball_two[2]/2), int(dims_ball_two[1] + dims_ball_two[3]/2))
    
        dist = find_dist_bet_centers(center_one, center_two)
        dist = round(dist, 2)
        
    return dist, dims, roi_final_grey   ##    result_rois
        
dist, dims, result_rois = find_dist(gray, image)     

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    