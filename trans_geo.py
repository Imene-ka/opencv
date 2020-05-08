import cv2
import sys
import math
from math import floor
import numpy as np


def fonction(img,d):
  p=img.shape
  l=p[0]
  c=p[1]
  i=0
  result = np.zeros((l,c, 3), np.uint8)
  while i<l :
      j=0
      while j<c :
          (r,g,b) = img[i,j]
          i_n=floor(i*math.cos(d)-j*math.sin(d))
          j_n=floor(j*math.cos(d)+i*math.sin(d))
          result[i_n,j_n]= (r,g,b)
          j=j+1
      i=i+1
  cv2.imshow("new image",result)


src = cv2.imread("/home/client/Images/a2.png")
if src.size == 0 :
    sys.exit("Error: the image has not been correctly loaded.")
fonction(src,40)