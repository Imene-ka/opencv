import cv2
import sys

def filtre_médian(src,result) :
    while True:
        cv2.imshow("old image", src)
        cv2.waitKey(0)
        cv2.imshow("new image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rep = input("svp entrez 'y' si vous voulez repetez le filtre sinon 'n' :")
        if rep =="n" or rep =="N" :
            break

def filtre_gaussien(src,result) :
    while True:
        cv2.imshow("old image", src)
        cv2.waitKey(0)
        cv2.imshow("new image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rep = input("svp entrez 'y' si vous voulez repetez le filtre sinon 'n' :")
        if rep =="n" or rep =="N" :
            break

def filtre_prewitt(src,result) :
    while True:
        cv2.imshow("old image", src)
        cv2.waitKey(0)
        cv2.imshow("new image", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rep = input("svp entrez 'y' si vous voulez repetez le filtre sinon 'n' :")
        if rep =="n" or rep =="N" :
           break

src = cv2.imread("unnamed.jpg", cv2.IMREAD_GRAYSCALE)
if src.size == 0 :
    sys.exit("Error: the image has not been correctly loaded.")
shape=src.shape
filtre_médian(src,src)
filtre_gaussien(src,src)
filtre_prewitt(src,src)

