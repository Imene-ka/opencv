import cv2
import sys

img = cv2.imread("/home/client/Images/a2.png")
if img.size == 0 :
    sys.exit("Error: the image has not been correctly loaded.")


