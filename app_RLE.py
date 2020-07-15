import sys
import numpy as np
import cv2

import algo_RLE


def to_bin(mat,l,c):
    str_mat=""
    i = 0
    while i < l:
        j = 0
        while j < c:
            chaine=str.replace(bin(mat[i][j]),"0b","")
            if len(chaine) < 8 :
                chaine=chaine.zfill(8)
            str_mat=str_mat+chaine
            j = j + 1
        i = i + 1
    return str_mat

def codage_image_RLE(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    RR = algo_RLE.rle_binaire(to_bin(R, R.shape[0], R.shape[1]))
    GG = algo_RLE.rle_binaire(to_bin(G, G.shape[0], G.shape[1]))
    BB = algo_RLE.rle_binaire(to_bin(B, B.shape[0], B.shape[1]))

    return RR,GG,BB

def to_dec(s,l,c):
    mat=np.zeros((l,c))
    h=0
    i = 0
    while i < l:
        j = 0
        while j < c:
            k=s[h:h+8]
            mat[i][j]=int(k,2)
            j = j + 1
            h=h+8
        i = i + 1

    return mat

def decodage(RR,GG,BB,l,c):

    R=to_dec(algo_RLE.rle_binaire_inverse(RR),l,c)
    G=to_dec(algo_RLE.rle_binaire_inverse(GG),l,c)
    B=to_dec(algo_RLE.rle_binaire_inverse(BB),l,c)

    result=np.empty((l,c,3),np.uint8)

    result[:, :, 0]=R
    result[:, :, 1]=G
    result[:, :, 2]=B

    return result

#lecteur de l'image en couleur
img_couleur = cv2.imread("a2.png")
# verifier esq la lecture de l'image fait correctement
if img_couleur.size == 0 :
    sys.exit("Error: the image has not been correctly loaded.")
#codage d'une image
RR,GG,BB=codage_image_RLE(img_couleur)
#decodage de l'image
result=decodage(RR,GG,BB,img_couleur.shape[0],img_couleur.shape[1])
cv2.imshow("result",result)
cv2.waitKey(0)
cv2.destroyAllWindows()
#calculer le taux
taux=(sys.getsizeof(RR)+sys.getsizeof(GG)+sys.getsizeof(BB))/sys.getsizeof(img_couleur)
print(taux)


