import sys
import os
import cv2
import numpy as np
from myfunctions import edge_continuity, detect_surrounded_background


cv2.namedWindow("image", cv2.WINDOW_NORMAL)  
cv2.namedWindow("small_image", cv2.WINDOW_NORMAL)  
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)  

path_images = './data/train/0'
path_mask = './data/train_mask/0'
filelist= [file for file in os.listdir(path_images) if file.endswith('.jpg')]
threshold_pixels_1 = 0.0001 
threshold_pixels_2 = 0.0001
min_d = 5

for img_file in filelist:
    print(img_file)
    try:
      
      image = cv2.imread(os.path.join(path_images, img_file)) 
      height, width, _ = image.shape
      total_pixels = height*width

      n_pixels = np.count_nonzero((image == [255, 255, 255]).all(axis = 2))
      percentage_pixels = n_pixels/total_pixels

      if(percentage_pixels > threshold_pixels_1):

         dsize = (224, 224)#tamaño entrada cnn
         small_image = cv2.resize(image, dsize, interpolation = cv2.INTER_NEAREST)

         percentage_continuity = edge_continuity(small_image, min_d)
         #print('percentage_continuity: ', percentage_continuity)
         if(percentage_continuity > threshold_pixels_2):          
            background_pixels = detect_surrounded_background(small_image, min_d)
            list_coor = []
            mask = 255*np.ones((224, 224, 1), dtype = "uint8")  
            for pixel in background_pixels:
                y, x = pixel.point
                list_coor.append((y, x))
                small_image[y, x, :] = (255, 0, 0)
                mask[y, x] = 0
           
            
            # Saving the mask_image
            cv2.imwrite(os.path.join(path_mask, img_file), mask)

            cv2.imshow('mask', mask)
            cv2.imshow('small_image', small_image)
            cv2.imshow('image', image)
            if ord('q') == cv2.waitKey(1):
               break

    except:
      print('error', sys.stderr)
      




