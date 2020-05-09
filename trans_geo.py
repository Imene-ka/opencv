import cv2
import sys
import math
from math import floor
import numpy as np


def fonction(img,d):
  p=img.shape
  l=p[0]
  print(l)
  c=p[1]
  print(c)
  i=0
  result = np.zeros((l+c,c+l, 3), np.uint8)
  while i<l :
      j=0
      while j<c :
          pixel = img[i,j]
          i_n=-floor(i*math.cos(d)-j*math.sin(d))
          j_n=-floor(j*math.cos(d)+i*math.sin(d))
          #if i_n < l and j_n < c :
          result[i_n,j_n]= pixel
          j=j+1
      i=i+1
  cv2.imshow("new image",result)
  cv2.waitKey(0)


src = cv2.imread("/home/client/Images/a2.png")
if src.size == 0 :
    sys.exit("Error: the image has not been correctly loaded.")
fonction(src,10)