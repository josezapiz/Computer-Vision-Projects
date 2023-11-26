#python3.9
import cv2 #4.4.0
import numpy as np
import os #for saving the pictures
import random


def rand():
    rdn = random.random()
    return rdn


def noise(image,prob):
    prob = prob/100
    output = np.zeros(image.shape,np.uint8) #making sure the output is int
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i][j] = image[i][j] + prob*rand()*image[i][j] - prob*rand()*image[i][j]
    return output


def saveImage(image,value):
    savefolder = r'path'
    filename = 'Result with noise = ' +str(value) + '.jpg'
    os.chdir(savefolder)
    cv2.imwrite(filename,image)

def saveFilteredImage(image,value):
    savefolder = r'path'
    filename = 'FILTERED Result with noise = ' +str(value) + '.jpg'
    os.chdir(savefolder)
    cv2.imwrite(filename,image)



#________READING IMAGE______#
folder = r'path'
os.chdir(folder)
original = cv2.imread('swords.jpg')

#_________________________APPLYING NOISE_________________________#
for x in range(0,21,5):
 results = noise(original,x)
 cv2.imshow('Swords with noise = ' + str(x) , results)
 cv2.waitKey(0)
 cv2.destroyAllWindows()


 #load the template image we look for
 template = cv2.imread('shape.jpg',0)
 w, h = template.shape[::-1]

 #run the template matching
 results_gray = cv2.cvtColor(results, cv2.COLOR_BGR2GRAY)
 res = cv2.matchTemplate(results_gray,template,cv2.TM_CCOEFF_NORMED)
 threshold = 0.70
 loc = np.where( res >= threshold)

 #mark the corresponding location(s)
 
 for pt in zip(*loc[::-1]):
   cv2.rectangle(results, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
  
 cv2.imshow('Detected shapes with noise = ' + str(x), results)
 cv2.waitKey(0)
 cv2.destroyAllWindows()


 saveImage(results,x)
 os.chdir(folder)


#_________________________APPLYING NOISE_________________________#
for x in range(0,21,5):
 results = noise(original,x)
 cv2.imshow('Swords with noise = ' + str(x) , results)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

 #___filter___#
 results = cv2.GaussianBlur(results,(3,3),0)
 #load the template image we look for
 template = cv2.imread('shape.jpg',0)
 w, h = template.shape[::-1]

 #run the template matching
 results_gray = cv2.cvtColor(results, cv2.COLOR_BGR2GRAY)
 res = cv2.matchTemplate(results_gray,template,cv2.TM_CCOEFF_NORMED)
 threshold = 0.70
 loc = np.where( res >= threshold)

 #mark the corresponding location(s)
 
 for pt in zip(*loc[::-1]):
   cv2.rectangle(results, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
  
 cv2.imshow('Detected shapes with noise = ' + str(x), results)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

 saveFilteredImage(results,x)
 os.chdir(folder)
