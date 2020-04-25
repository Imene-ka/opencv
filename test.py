import sys
import cv2

# We load the image from disk
img_couleur = cv2.imread("/home/client/Images/a2.png")
img_gris = cv2.imread("/home/client/Images/a2.png", cv2.IMREAD_GRAYSCALE)

# We check that our image has been correctly loaded
if img_couleur.size == 0 & img_gris.size == 0:
    sys.exit("Error: the image has not been correctly loaded.")

R = img_couleur[:, :, 0]
V = img_couleur[:, :, 1]
B = img_couleur[:, :, 2]

# demention de l'image
print(img_couleur.shape)
# We display our image and ask the program to wait until a key is pressed
# rouge
cv2.imshow("R", R)
cv2.waitKey(0)
# close the windows
cv2.destroyAllWindows()
# VERT
cv2.imshow("V", V)
cv2.waitKey(0)
# close the windows
cv2.destroyAllWindows()
# BLEU
cv2.imshow("B", B)
cv2.waitKey(0)
# close the windows
cv2.destroyAllWindows()
# afficher l'image en couleur
cv2.imshow("image en couleur", img_couleur)
cv2.waitKey(0)
# close the windows
cv2.destroyAllWindows()
# afficher l'image en gris
cv2.imshow("Mimage en gris", img_gris)
cv2.waitKey(0)
# close the windows
cv2.destroyAllWindows()