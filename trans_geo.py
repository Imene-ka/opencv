import cv2
import sys

src = cv2.imread("/home/client/Images/a2.png")
if src.size == 0 :
    sys.exit("Error: the image has not been correctly loaded.")
des=cv2.Mat()
desize=cv2.Size(src.rows,src.cols)
center=cv2.point(src.cols/2,src.rows /2)
m=cv2.getRotationMatrix2D(center,45,1)
cv2.warpAffine(src,des,m,desize,cv2.INTER_LINEAR,cv2.BORDER_CONSTANT,cv2.Scalar())
cv.imshow("new windows",des)
src.delete()
des.delete()
m.delete()

