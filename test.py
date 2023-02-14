import sys
import cv2
import pandas as pd
import numpy as np

from model import classifier
from myfunctions import read_image_from_url, evaluate_image, evaluate_image_and_return_mask


cv2.namedWindow("image", cv2.WINDOW_NORMAL)  
cv2.namedWindow("small_image", cv2.WINDOW_NORMAL)  
cv2.namedWindow("mask", cv2.WINDOW_NORMAL)  


threshold_pixels_1 = 0.0 #0.03 
threshold_pixels_2 = 0.0 #0.003
threshold_pixels_3 = 0.0 #0.2
min_d = 49 + 49 + 49
m_conf = np.zeros((2, 2), np.float32) #matrix confusión



df = pd.read_csv('./data/productive_data.csv')
get_url = lambda id: 'https://http2.mlstatic.com/D_' + id + '-F.jpg'

for index, row in df.iterrows():

    picture_id = row['picture_id']
    label = row['correct_background?']

    try:
      
      picture_url = get_url(picture_id)
      print('url: ', picture_url, 'label: ', label)
      
      image = read_image_from_url(picture_url)
      '''
      pred = evaluate_image(image, 
                            classifier, 
                            threshold_pixels_1, 
                            threshold_pixels_2, 
                            threshold_pixels_3, 
                            min_d)
      '''
      pred, small_image, mask = evaluate_image_and_return_mask(image, 
                                                               classifier, 
                                                               threshold_pixels_1, 
                                                               threshold_pixels_2, 
                                                               threshold_pixels_3, min_d)


      if(label == "1" or label == "0"):
         label = int(label)
         m_conf[pred][label] = m_conf[pred][label]  + 1 
         
      print(m_conf)
      
      if(True):
         cv2.imshow('mask', mask)
         cv2.imshow('small_image', small_image)
         cv2.imshow('image', image)
         if ord('q') == cv2.waitKey(1):
            break
      
    except:
      print('error', sys.stderr)


