import cv2
import numpy as np
import os

folder = r'path'
resultFolder = r'path'
os.chdir(folder)

#_______________________________________CORREL METHOD________________________________________#

original = cv2.imread('swords.jpg')
original2 = cv2.imread('swords.jpg')
height_original, width_original = original.shape[:2]

template = cv2.imread('shape.jpg')
height_template, width_template = template.shape[:2]

#___template histogram and threashold___#
template_hist = cv2.calcHist([template], [0], None, [256], [0,256])
scores = np.zeros((height_original - height_template, width_original - width_template))
location = []
mask = np.ones((height_template, width_template, 3))
threshold = 0.60

#___scaning__#
for m in range(0, height_original - height_template):
  for n in range(0, width_original - width_template):
     object = original[m: m + height_template, n: n + width_template]
     histOriginal = cv2.calcHist([object], [0], None, [256], [0, 256])
     scores[m, n] = cv2.compareHist(template_hist, histOriginal, cv2.HISTCMP_CORREL) 
     if scores[m, n] > threshold:
       offset = np.array((m, n))  
       original[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
       (max_Y, max_X) = (m, n)
       location.append((max_X, max_Y)) 

#___creating rectangle___#
for i in range(0, len(location)):
   loc = location[i]
   cv2.rectangle(original2, loc, (loc[0] + width_template, loc[1] + height_template), (255, 0, 0), 3)

cv2.imshow("Result with CORREL METHOD", original2)
os.chdir(resultFolder)
cv2.imwrite("Result with CORREL METHOD.jpg", original2)
cv2.waitKey(0)
cv2.destroyAllWindows()
os.chdir(folder)
#___________________________________________________________________________________________#



#__________________________________BHATTACHARYYA METHOD_____________________________________#
original3 = cv2.imread('swords.jpg')

#___template histogram and threashold___#
template_hist = cv2.calcHist([template], [0], None, [256], [0,256])
scores = np.zeros((height_original - height_template, width_original - width_template))
location = []
mask = np.ones((height_template, width_template, 3))
threshold = 0.45

#___scanning__#
for m in range(0, height_original - height_template):
  for n in range(0, width_original - width_template):
     object = original[m: m + height_template, n: n + width_template]  
     histOriginal = cv2.calcHist([object], [0], None, [256], [0, 256])
     scores[m, n] = cv2.compareHist(template_hist, histOriginal, cv2.HISTCMP_BHATTACHARYYA)  
     if scores[m, n] < threshold: 
       offset = np.array((m, n))  
       original[offset[0]:offset[0] + mask.shape[0], offset[1]:offset[1] + mask.shape[1]] = mask
       (max_Y, max_X) = (m, n)
       location.append((max_X, max_Y)) 

#___creating rectangle___#
for i in range(0, len(location)): 
   loc = location[i]
   cv2.rectangle(original3, loc, (loc[0] + width_template, loc[1] + height_template), (0, 0, 0), 3)

cv2.imshow("Results with BHATTACHARYYA method", original3)
os.chdir(resultFolder)
cv2.imwrite("Result with BHATTACHARYYA method.jpg", original3)
cv2.waitKey(0)
cv2.destroyAllWindows()
#___________________________________________________________________________________________#