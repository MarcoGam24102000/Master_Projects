# this_dir = os.getcwd()
# foldername_string = "CVis_2223_Assign1_MarcoGameiro"
# dir_parts = this_dir.split(foldername_string)
# base_dir = dir_parts[0] + foldername_string + "\\"
   
# #%%
   
# # working_dir = base_dir + "data\\Boccia_balls.jpg"

# # working_dir = working_dir.replace("\\", "/")
    
# ## image = cv2.imread(working_dir)

# image = cv2.imread('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/data/Boccia_balls.jpg')


# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imwrite("hsv_image.png", image_hsv)

# selected_image = image_hsv[:,:,2] 

# sobel_64 = cv2.Sobel(selected_image,cv2.CV_64F,1,0,ksize=3)
# abs_64 = np.absolute(sobel_64)
# sobel_8u = np.uint8(abs_64)

# cv2.imwrite("hsv_sobel_output_image.png", sobel_8u)

# mod_img = np.zeros_like(sobel_8u)    


# for b in range(len(sobel_8u[0])):
#     for a in range(len(sobel_8u)):
#         if sobel_8u[a,b] >= 125:
#             mod_img[a,b] = 255
#         else:
#             mod_img[a,b] = 0
    
# cv2.imwrite("hsv_mod_img.png", mod_img)

# ## Look closely

# th, im_th = cv2.threshold(sobel_8u, 220, 255, cv2.THRESH_BINARY_INV);

# cv2.imwrite("hsv_thresh.png", im_th)
 
# # Copy the thresholded image.
# im_floodfill = im_th.copy()  

# h, w = im_th.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)

# # Floodfill from point (0, 0)
# cv2.floodFill(im_floodfill, mask, (0,0), 255); 
  
# # Invert floodfilled image
# im_floodfill_inv = cv2.bitwise_not(im_floodfill)
 
# # Combine the two images to get the foreground.
# im_out = im_th | im_floodfill_inv 



# ##
# canny = cv2.Canny(sobel_8u, 30, 250)

# cv2.imwrite('Canny_image.png', canny)
# #

# cnts = cv2.findContours(mod_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
# dimensions_rois = []
# dims = []
    
# roi_final = np.zeros_like(image)
    
# not_include = False
    
# result_rois = []
    
# for c in cnts:
#     x,y,w,h = cv2.boundingRect(c)
    
#     if w > 50 and w < 500 :
#             if h > 50 and h < 500:
    
#                 dims.append([x,y,w,h])
                
                
# #  ##   ROI = thresh[y:y+h,x:x+w]

# ####################################################################

# image = cv2.imread('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/data/Boccia_balls.jpg')


# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imwrite("hsv_image.png", image_hsv)

# h, w = image_hsv.shape[:2]
# mask = np.zeros((h+2, w+2), np.uint8)

# # Floodfill from point (0, 0)
# hsv_tuple_filling = cv2.floodFill(image_hsv, mask, (0,0), 255);

# hsv_filling = hsv_tuple_filling[2]


# hsv_filled = hsv_tuple_filling[1]

# cv2.imwrite("hsv_filled_image.png", hsv_filled)

# hsv_filling_output = np.zeros_like(hsv_filling)

# for b in range(len(hsv_filling[0])):
#     for a in range(len(hsv_filling)):
#         if hsv_filling[a,b] == 0:
#             hsv_filling_output[a,b] = 0
#         elif hsv_filling[a,b] == 1:
#             hsv_filling_output[a,b] = 255
            
# cv2.imwrite("hsv_filled.png", hsv_filling_output)


# #######################################################################

# image = cv2.imread('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/data/Boccia_balls.jpg')
 

# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imwrite("hsv_image.png", image_hsv)

# kernel = np.ones((9,9),np.uint8)

# opened = cv2.morphologyEx (image_hsv, cv2.MORPH_CLOSE, kernel)

# cv2.imwrite("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\hsv_opened_image.png", opened)

            
# #####################################################################

# image = cv2.imread('C:/VisComp/PL/Project/CVis_2223_Assign1_MarcoGameiro/data/Boccia_balls.jpg')


# image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# cv2.imwrite("hsv_image.png", image_hsv)

# edges_hsv = cv2.Canny(image_hsv, 30, 100)

# median = cv2.medianBlur(edges_hsv,3)

# cv2.imwrite("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\median_hsv.png", median)

# ## medianImage_read = cv2.imread("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\median_hsv.png")


# element = np.array([[0,1,0],
#                     [1,1,1],
#                     [0,1,0]], np.uint8)

# dilated = cv2.dilate(median, element)

# ## eroded = median-eroded

# cv2.imwrite("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\dilated_hsv.png", dilated)


# ## Convert HSV to GREY

# thresh = cv2.adaptiveThreshold(image_hsv, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2) #adaptive

#     ## cv2.imwrite("Threshold_image.png", thresh) 

#     ## canny = cv2.Canny(thresh, 100, 200)

# canny = cv2.Canny(thresh, 30, 250)

# masked = cv2.bitwise_and(image_hsv, image_hsv, mask=thresh)

# img = cv2.imread("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\dilated_hsv.png")
# img_bw = 255*(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 5).astype('uint8')

# se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# mask = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, se1)
# mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)

# mask = np.dstack([mask, mask, mask]) / 255
# out = img * mask 

# # ROI_number = 0
# # thickness = 2
# # color = (0, 0, 255) 
# # cnts = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     ## canny
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
# # dimensions_rois = []
    
# # roi_final = np.zeros_like(image)   
    
    
# # result_rois = []

# # count = 0
    
# # for c in cnts:
# #     x,y,w,h = cv2.boundingRect(c)    
    
# #     print("x:" + str(x) + 
# #           "\ty:" + str(y) +
# #           "\tw:" + str(w) +
# #           "\th:" + str(h))
    
# #     dims = [x,y,w,h]
    
# #     dimensions_rois.append(dims)
    
    
# #     if w > 50 and w < 500 :
# #             if h > 50 and h < 500:
# #             ##     if abs(w-h) <= 15 or abs(w-h) >= 25:
# # ##                    print("\nExists ROI")

# #                   count += 1




# hsv_image = cv2.imread("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\hsv_image.png")
# hsv_thresh = cv2.imread("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\hsv_thresh.png")

# grey_img = hsv_image[:,:,2]
# grey_thresh = hsv_thresh[:,:,2]

# ## hsv_out = hsv_image+hsv_thresh 

# image_out = np.zeros_like(grey_img)

# for b in range(len(grey_img[0])): 
#     for a in range(len(grey_img)):
#         if grey_img[a,b] > 150  :   ## and grey_img[a,b] < 200
#             image_out[a,b] = 255
#         else:
#             image_out[a,b] = 0
            
# cv2.imwrite("C:\\VisComp\\PL\\Project\\CVis_2223_Assign1_MarcoGameiro\\code_python\\image_out_hsv.png", image_out)
            


# hsv_out = cv2.bitwise_and(grey_img, grey_img, mask=grey_thresh)

# # def rgb2gray(rgb): 

#     r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

#     return gray 


