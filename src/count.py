# -*- coding: utf-8 -*-
'''
Cell counting.
@author     Matthias Moulin
@version    1.0
'''

import cv2
import cv2.cv
import numpy as np

########################################################################################################################################################################################
#PARAMETERS
########################################################################################################################################################################################

#params Canny
canny_upper_lower_ratio = 2                                           #Canny recommended an upper:lower ratio between 2:1 and 3:1
canny_high_threshold = 100                                            #Used to find initial segments of strong edges
canny_low_threshold = canny_high_threshold / canny_upper_lower_ratio  #Used for edge linking

#params Hough
dp =  1  
minDist = 40                                                          #Minimum distance between the centers of the detected circles.
param1 = canny_high_threshold                                         #The higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
param2 = 12                                                           #The accumulator threshold for the circle centers at the detection stage.
minRadius = 30                                                        #Minimum circle radius.
maxRadius = 60                                                        #Maximum circle radius.

#params Feature elimination
colour_model = 1                                                      #BGR=0 | HSV=1
fv_high_threshold = np.array([165, 51, 211])                          #The higher threshold values for feature elimination. [HSV]
fv_low_threshold = np.array([90, 20, 160])                           #The lower threshold values for feature elimination. [HSV]

#-> exloit the saturation (S) value in HSV model for eliminating the blue cells

#params general
option = 0                                                            #All=0 | Canny=1 | Hough=2 (Canny already incl.) | Text=3 
img_name = 'img'                                                      #The name of the image for diplaying.
img_png = 'img.png'                                                   #The name of the image for writing to disk.

########################################################################################################################################################################################
#HSV (Wikipedia)
########################################################################################################################################################################################
#Hue (Nederlands: tint),                                             is wat we gewoonlijk 'kleur' noemen, zeg een punt op de regenboog.
#                                                                    In het HSV-model wordt de kleur uitgezet op een cirkel, en wordt
#                                                                    de plek aangeduid in graden: Hue loopt dus van 0 tot 360 (graden).
#Saturation (Nederlands: verzadiging)                                dit geeft een hoeveelheid (of felheid) van een kleur aan.
#                                                                    Wordt uitgedrukt in procenten, en loopt van 0% (flets, grijs) naar 100% (volle kleur).
#Value of Brightness (Nederlands: intensiteit):                      staat voor de lichtheid van de kleur.
#                                                                    Wordt uitgedrukt in procenten, en loopt van 0% (zwart) naar 100% (wit).

#For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255]. 


def detect(img):
    '''
    Do the detection.
    @param     img          the image   
    '''
    #You can use a global variable in other functions by declaring it
    #as global in each function that assigns to it.
    global img_name, img_png
    global canny_upper_lower_ratio, canny_high_threshold, canny_low_threshold, options
    
    #create a gray scale version of the image, with as type an unsigned 8bit integer
    img_g = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    img_g[:,:] = img[:,:,0]

    ########################################################################################################################################################################################
    #CANNY
    ########################################################################################################################################################################################
    #1. Do canny (determine the right parameters) on the gray scale image
    edges = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    #image          – single-channel 8-bit input image.
    #edges          – output edge map; it has the same size and type as image .
    #threshold1     – first threshold for the hysteresis procedure.
    #threshold2     – second threshold for the hysteresis procedure.
    #apertureSize   – aperture size for the Sobel() operator.
    #L2gradient     – a flag, indicating whether a more accurate
    #L_2 norm should be used to calculate the image gradient magnitude ( L2gradient=true ),
    #or whether the default  L_1 norm  is enough ( L2gradient=false ).
    cv2.Canny(img_g, canny_low_threshold, canny_high_threshold, edges)
    
    #Show the results of canny
    if option <= 1:
        #an array copy of the given object
        canny_result = np.copy(img_g)
        #copy of the array, cast to a specified type.
        canny_result[edges.astype(np.bool)]=0
        img_name = 'canny'
        img_png = 'canny.png'
        cv2.imshow(img_name, canny_result)
        cv2.waitKey(0)
        cv2.imwrite(img_png, canny_result) 
        if option == 1: return

    ########################################################################################################################################################################################
    #HOUGH
    ########################################################################################################################################################################################
    #2. Do hough transform on the gray scale image
    
    #You can use a global variable in other functions by declaring it
    #as global in each function that assigns to it.
    global dp, minDist, param1, param2, minRadius, maxRadius
    
    #image          – 8-bit, single-channel, grayscale input image.
    #circles        – Output vector of found circles. Each vector is encoded as a 3-element floating-point vector  (x, y, radius) .
    #method         – Detection method to use. Currently, the only implemented method is CV_HOUGH_GRADIENT
    #dp             – Inverse ratio of the accumulator resolution to the image resolution. 
    #                (For example, if dp=1 , the accumulator has the same resolution as the input image.
    #                If dp=2 , the accumulator has half as big width and height.)
    #minDist        – Minimum distance between the centers of the detected circles. 
    #                If the parameter is too small, multiple neighbor circles may be falsely detected in addition to a true one. 
    #                If it is too large, some circles may be missed.
    #param1         – In case of CV_HOUGH_GRADIENT , it is the higher threshold of the two passed to the Canny() edge detector (the lower one is twice smaller).
    #param2         – In case of CV_HOUGH_GRADIENT , it is the accumulator threshold for the circle centers at the detection stage.
    #                 The smaller it is, the more false circles may be detected. Circles, corresponding to the larger accumulator values, will be returned first.
    #minRadius      – Minimum circle radius.
    #maxRadius      – Maximum circle radius.
    circles = cv2.HoughCircles(img_g, cv2.cv.CV_HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    #print circles.shape = e.g.: (1L, 164L, 3L)
    circles = circles[0,:,:]
    #print circles.shape = e.g.: (164L, 3L)
    
    #Show hough transform result
    if option <= 2:
        img_name = 'hough'
        img_png = 'hough.png'  
        showCircles(img, circles)
        if option == 2: return
    
    ########################################################################################################################################################################################
    #Feature vectors
    ########################################################################################################################################################################################
    #3.a Get a feature vector (the average color) for each circle
    
    #BGR -> HSV conversion
    global colour_model
    if colour_model == 1 : 
        img_m = np.copy(img)
        img_m = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    else : 
        img_m = img
    
    nbCircles = circles.shape[0]
    features = np.zeros((nbCircles,3), dtype=np.int)
    for i in range(nbCircles):
        features[i,:] = getAverageColorInCircle(img_m , int(circles[i,0]), int(circles[i,1]), int(circles[i,2]))
    
    #3.b Show the image with the features (just to provide some help with selecting the parameters)
    if option <= 3:
        img_name = 'info'
        img_png = 'info.png'
        showCircles(img, circles, [str(features[i,:]) for i in range(nbCircles)])
        if option == 3: return

    #3.c Remove circles based on the features
    selectedCircles = np.zeros((nbCircles), np.bool)
    for i in range(nbCircles):
        if ((fv_low_threshold < features[i,:]).all() and 
            (fv_high_threshold > features[i,:]).all()):
            selectedCircles[i] = 1
    circles = circles[selectedCircles]

    #Show final result
    img_name = 'result'
    img_png = 'result.png'
    showCircles(img, circles)    
    return circles
        
    
def getAverageColorInCircle(img, cx, cy, radius):
    '''
    Get the average color of img inside the circle located at (cx,cy) with radius.
    @param     img          the image  
    @param     cx           the horizontal coordinate of the center of the circle
    @param     cy           the vertical coordinate of the center of the circle
    @param     radius       the radius of the circle
    '''
    maxy, maxx, channels = img.shape
    
    #nbVoxels = 0
    
    patch, mask =  truncate(img, getCircleMask(radius), cx, cy, radius)
    C = np.zeros((3))
    for c in range(channels):
        #Only False values are taken into account for the mean
        C[c] = np.mean(np.ma.masked_array(patch[:,:,c], mask))
    return C

def getValueCircleMask(radius):
    '''
    Get a (value) mask for a circle with given radius.
    @param     radius       the radius of the circle
    '''
    #just for quickly getting the row and column
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x*x + y*y
    return mask

def getCircleMask(radius):
    '''
    Get a (boolean) mask for a circle with given radius.
    @param     radius       the radius of the circle
    '''
    #False means inside the circle, True means outside the circle
    return getValueCircleMask(radius) > radius*radius
    
def truncate(img, mask, cx, cy, radius):
    '''
    Truncate the given mask and circle in the image.
    @param     img          the image  
    @param     mask         the mask
    @param     cx           the horizontal coordinate of the center of the circle
    @param     cy           the vertical coordinate of the center of the circle
    @param     radius       the radius of the circle
    '''
    img_x = img.shape[1]
    img_y = img.shape[0]
    xmin, xmax, ymin, ymax = 0, 0, 0, 0
    
    #  axis:
    #         (y+)*****
    #         |***img**
    #         |********
    # (x-)---(0)---(x+)
    #        |
    #        |
    #      (y-)
    
    #truncating the mask
    #mind the Python inclusions and exclusions
    if cx + radius >= img_x:
        xmax = cx + radius - img_x + 1
        mask = mask[:,:-xmax]
    if cx - radius < 0:
        xmin = radius - cx
        mask = mask[:,xmin:]
    if cy + radius >= img_y:
        ymax = cy + radius - img_y + 1
        mask = mask[:-ymax,:]
    if cy - radius < 0:
        ymin = radius - cy
        mask = mask[ymin:,:]
        
    #truncating the circle in the image
    #mind the Python inclusions and exclusions
    patch = img[max(0,cy-radius) : min(img_y,cy+radius+1), max(0,cx-radius) : min(img_x,cx+radius+1), :]
    
    return patch, mask
    
def showCircles(img, circles, text=None):
    '''
    Show circles on an image.
    @param img:     numpy array
    @param circles: numpy array 
                    shape = (nb_circles, 3)
                    contains for each circle: center_x, center_y, radius
    @param text:    optional parameter, list of strings to be plotted in the circles
    '''
    global img_name, img_png
    
    #make a copy of img
    img = np.copy(img)
    #draw the circles
    nbCircles = circles.shape[0]
    for i in range(nbCircles):
        cv2.circle(img, (int(circles[i,0]), int(circles[i,1])), int(circles[i,2]), cv2.cv.CV_RGB(255, 0, 0), 2, 8, 0)
    #draw text
    if text!=None:
        for i in range(nbCircles):
            cv2.putText(img, text[i], (int(circles[i,0]), int(circles[i,1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cv2.cv.CV_RGB(0, 0, 255))
    #show the result
    #everything is placed on img!
    cv2.imshow(img_name, img)
    cv2.waitKey(0)  
    cv2.imwrite(img_png, img)
        
if __name__ == '__main__':
    #read an image
    img = cv2.imread('normal.jpg')
    
    #print the dimension of the image
    print img.shape
    
    #show the image
    cv2.imshow('img',img)
    cv2.waitKey(0)
    
    #do detection
    circles = detect(img)
    
    #155 cells
    print "We counted "+str(circles.shape[0])+ " cells."