import cv2
import numpy as np
from skimage import io, img_as_ubyte

class BackGroundPixel:
      def  __init__(self, point):
           self.point = point

def norm_l2(color_a, color_b):
    d = color_a - color_b
    l2 = np.linalg.norm(d)
    return l2

def is_background(image, point, min_d, color = (255, 255, 255)):

    y, x = point
    height, width, _ = image.shape
    
    flag = False
    posible_pixels = []

    if(norm_l2(image[y, x], color) < min_d):
       deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
       for delta in deltas:
           xd = x + delta[0]
           yd = y + delta[1]
           if(xd < 0 or xd >= width or yd < 0 or yd >= height):
              continue
           if(norm_l2(image[y, x], color) < min_d):
              pixel = BackGroundPixel((yd, xd)) 
              posible_pixels.append(pixel)
    
    if(len(posible_pixels) > 0):
       flag = True

    return posible_pixels, flag

def detect_surrounded_background(image, min_d):
    height, width, _ = image.shape
    background_pixels = []
    mask_background_pixels = [[False for j in range(0, width)] for i in range(0, height)]
    possible_background_pixels = []

    for j in range(0, width):
        pixel = BackGroundPixel((0, j))
        possible_background_pixels.append(pixel)
        mask_background_pixels[0][j] = True

    for j in range(0, width):
        pixel = BackGroundPixel((height - 1, j)) 
        possible_background_pixels.append(pixel)
        mask_background_pixels[height - 1][j] = True

    for i in range(1, height - 1):
        pixel = BackGroundPixel((i, 0)) 
        possible_background_pixels.append(pixel)
        mask_background_pixels[i][0] = True
        
    for i in range(1, height - 1):
        pixel = BackGroundPixel((i, width - 1)) 
        possible_background_pixels.append(pixel)
        mask_background_pixels[i][width - 1] = True

    
    while(len(possible_background_pixels) > 0):
    
        new_possible_background_pixel = []
        for i, pixel in enumerate(possible_background_pixels):
            posible_pixels, flag = is_background(image, pixel.point, min_d)
            if(flag):
               background_pixels.append(pixel)
               for posible_pixel in posible_pixels:
                   y, x = posible_pixel.point
                   if(not mask_background_pixels[y][x]):
                     new_possible_background_pixel.append(posible_pixel)
                     mask_background_pixels[y][x] = True
            
            possible_background_pixels[i] = None

        possible_background_pixels = [item for item in possible_background_pixels if item is not None]
        possible_background_pixels = possible_background_pixels + new_possible_background_pixel
    
    return background_pixels
   

def edge_continuity(image, min_d, color=(255, 255, 255)):
    height, width, _ = image.shape
    nh = 2
    list_pixels = []
    nocontinuous_0x = []
    nocontinuous_h_1x = []
    nocontinuous_y0 = []
    nocontinuous_yw_1 = []

    for x in range(0, width - nh, nh + 1):
        pxcolor0 = image[0, x]
        pxcolor1 = image[0, x + 1]
        pxcolor2 = image[0, x + 2]
        if(norm_l2(pxcolor0, color) < min_d and norm_l2(pxcolor1, color) < min_d and norm_l2(pxcolor2, color) < min_d ):
           list_pixels.append((0, x))
           list_pixels.append((0, x + 1))
           list_pixels.append((0, x + 2))
               

    for x in range(0, width - nh, nh + 1):
        pxcolor0 = image[height - 1, x]
        pxcolor1 = image[height - 1, x + 1]
        pxcolor2 = image[height - 1, x + 2]
        if(norm_l2(pxcolor0, color) < min_d and norm_l2(pxcolor1, color) < min_d and norm_l2(pxcolor2, color) < min_d):
           list_pixels.append((height - 1, x))
           list_pixels.append((height - 1, x + 1))
           list_pixels.append((height - 1, x + 2))


    for y in range(0, height - nh, nh + 1):
        pxcolor0 = image[y, 0]
        pxcolor1 = image[y + 1, 0]
        pxcolor2 = image[y + 2, 0]
        if(norm_l2(pxcolor0, color) < min_d and norm_l2(pxcolor1, color) < min_d and norm_l2(pxcolor2, color) < min_d ):
           list_pixels.append((y, 0))
           list_pixels.append((y + 1, 0))
           list_pixels.append((y + 2, 0))

    for y in range(0, height - nh, nh + 1):
        pxcolor0 = image[y, width - 1]
        pxcolor1 = image[y + 1, width - 1]
        pxcolor2 = image[y + 2, width - 1]
        if(norm_l2(pxcolor0, color) < min_d and norm_l2(pxcolor1, color) < min_d and norm_l2(pxcolor2, color) < min_d ):
           list_pixels.append((y, width - 1))
           list_pixels.append((y + 1, width - 1))
           list_pixels.append((y + 2, width - 1))

    total_pixles = 2*height + 2*width - 4
    percentage_continuity = len(list_pixels)/total_pixles
    return percentage_continuity


def read_image_from_url(url):
    image = img_as_ubyte(io.imread(url))
    if(image.ndim == 3):
       image = image[:, :, ::-1]
    else:#I assume gray
       image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image




def evaluate_image(image, classifier, threshold_pixels_1, threshold_pixels_2, threshold_pixels_3, min_d):
    pred = 0    
    height, width, _ = image.shape
    total_pixels = height*width
    n_pixels = np.count_nonzero((image == [255, 255, 255]).all(axis = 2))
    percentage_pixels = n_pixels/total_pixels
    #print('Percentage pixels color: ', percentage_pixels)
    if(percentage_pixels >= threshold_pixels_1):# zero inclusive
       dsize = (224, 224)#(width, height)
       small_image = cv2.resize(image, dsize, interpolation = cv2.INTER_NEAREST)
       percentage_continuity = edge_continuity(small_image, min_d)
       #print('percentage_continuity: ', percentage_continuity)
       if(percentage_continuity >= threshold_pixels_2):# zero inclusive
          background_pixels = detect_surrounded_background(small_image, min_d)
          list_coor = []
          mask = 255*np.ones((dsize[1], dsize[0], 3), dtype = "uint8")  
          for pixel in background_pixels:
              y, x = pixel.point
              list_coor.append((y, x))
              small_image[y, x, :] = (255, 0, 0)
              mask[y, x] = (0, 0, 0)
           
          n_pixels = len(list_coor)
          total_pixels = dsize[0]*dsize[1]
          percentage_pixels = n_pixels/total_pixels
          #print('porcentaje surrounded_background: ', percentage_pixels)
          if(percentage_pixels >= threshold_pixels_3):# zero inclusive
             pred = classifier.evaluate(mask) 

    return pred
    
def evaluate_image_and_return_mask(image, classifier, threshold_pixels_1, threshold_pixels_2, threshold_pixels_3, min_d):
    pred = 0
    small_image = None
    mask = None
    height, width, _ = image.shape
    total_pixels = height*width
    n_pixels = np.count_nonzero((image == [255, 255, 255]).all(axis = 2))
    percentage_pixels = n_pixels/total_pixels
    #print('Percentage pixels color: ', percentage_pixels)
    if(percentage_pixels >= threshold_pixels_1):# zero inclusive
       dsize = (224, 224)#(width, height)
       small_image = cv2.resize(image, dsize, interpolation = cv2.INTER_NEAREST)
       percentage_continuity = edge_continuity(small_image, min_d)
       #print('percentage_continuity: ', percentage_continuity)
       if(percentage_continuity >= threshold_pixels_2):# zero inclusive
          background_pixels = detect_surrounded_background(small_image, min_d)
          list_coor = []
          mask = 255*np.ones((dsize[1], dsize[0], 3), dtype = "uint8")  
          for pixel in background_pixels:
              y, x = pixel.point
              list_coor.append((y, x))
              small_image[y, x, :] = (255, 0, 0)
              mask[y, x] = (0, 0, 0)
           
          n_pixels = len(list_coor)
          total_pixels = dsize[0]*dsize[1]
          percentage_pixels = n_pixels/total_pixels
          #print('porcentaje surrounded_background: ', percentage_pixels)
          if(percentage_pixels >= threshold_pixels_3):# zero inclusive
             pred = classifier.evaluate(mask) 

    return pred, small_image, mask


    
