import math
import algo_RLE
import algo_quantif
import cv2
import pywt
import matplotlib.pyplot as plt
import huffman
import numpy as np


def calcule_entropie(list) :
    freq={}
    val=0
    for c in list:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
    for c,m in freq:
        freq[c]=freq[c]/len(list)
        val=val+(math.log2(freq[c])*freq[c])

    return val

def calcule_taille_rle(list):
    maximum=max(list)
    minimum=min(list)
    taille=(len(str.replace(bin(minimum),"0b","")))*len(list)
    return taille

def calcule_taille_imgColor(img):
    taille=img.shape[0]*img.shape[1]*3
    return taille

def array_list(mat,l,c):
    i=0
    list=[]
    while i<l :
        j=0
        while j<c:
            list.append(str(mat[i][j]))
            j=j+1
        i=i+1

    return list
def list_array(s,l,c):
    mat=np.zeros((l,c))
    i = 0
    k = 0
    while i < l:
        j = 0
        while j < c:
            mat[i][j]=s[k]
            j = j + 1
            k = k+1
        i = i + 1

    return mat

"""codec = HuffmanCodec.from_frequencies({'e': 100, 'n':20, 'x':1, 'i': 40, 'q':3})
encoded = codec.encode('exeneeeexniqneieini')
codec.print_code_table()"""

img_color=cv2.imread("camera.jpeg")
img_gris=cv2.imread("camera.jpeg",cv2.IMREAD_GRAYSCALE)

shape1=img_color.shape
shape2=img_gris.shape

"""R = img_color[:, :, 0]
G = img_color[:, :, 1]
B = img_color[:, :, 2]

#codage huffman /img_color

R_huff= huff.string_code(array_list(R,shape1[0],shape1[1]))
G_huff= huff.string_code(array_list(G,shape1[0],shape1[1]))
B_huff= huff.string_code(array_list(B,shape1[0],shape1[1]))

#calcule la taille sans rle binaire / img_color

taux1=(len(R_huff)+len(G_huff)+len(B_huff))/calcule_taille_imgColor(img_color)
print("taux de compression avec huffman sans rle : ")
print(taux1)

#rle binaire suite des 0 et 1 -> la liste / img_color

R_huff_rle=algo_RLE.rle_binaire(R_huff)
G_huff_rle=algo_RLE.rle_binaire(G_huff)
B_huff_rle=algo_RLE.rle_binaire(B_huff)

#calcule la taille apres rle /img_color

taille_R_huff_rle = calcule_taille_rle(R_huff_rle)
taille_G_huff_rle = calcule_taille_rle(G_huff_rle)
taille_B_huff_rle = calcule_taille_rle(B_huff_rle)

print("le taux de compression avec huffman - > rle : ")
taux2=(taille_B_huff_rle+taille_G_huff_rle+taille_R_huff_rle)/calcule_taille_imgColor(img_color)
print(taux2)

#--------------------------------------------------

# ondelette -> quantif -> huffman -> rle_binaire

#algirithme quantification

a,(h,v,d)=algo_quantif.app_quantif(img_gris,20)

#huffman

h_huff=huff.string_code(array_list(h,h.shape[0],h.shape[1]))
v_huff=huff.string_code(array_list(v,v.shape[0],v.shape[1]))
d_huff=huff.string_code(array_list(d,d.shape[0],d.shape[1]))

#rle binaire

h_huff_rle=algo_RLE.rle_binaire(h_huff)
v_huff_rle=algo_RLE.rle_binaire(v_huff)
d_huff_rle=algo_RLE.rle_binaire(d_huff)

# calcule la taille

taux3=(calcule_taille_rle(h_huff_rle)+calcule_taille_rle(v_huff_rle)+calcule_taille_rle(d_huff_rle))/(shape2[0]*shape2[1])
print("le taux de compression ondelette -> qauntif -> huffman -> rle :")
print(taux3)

# calcule des taux """

val=[5,10,15,20,25,30,35,40,50]
img=["a2.png","bebe.jpg","camera.jpeg","camera.jpeg","test1.jpg","unnamed.jpg","squirrel.jpg","cats.JPG","dog.jpg"]

for i,v in  enumerate(val):
    img_ = cv2.imread(img[i],cv2.IMREAD_GRAYSCALE)

    #compression
    #algo_quatif
    coef = algo_quantif.app_quantif(img_,v)
    a,(h,v,d)= coef

    # huffman

    tree1,h_huff = huffman.huffman_encoding_func(array_list(h, h.shape[0], h.shape[1]))
    tree2,v_huff = huffman.huffman_encoding_func(array_list(v, v.shape[0], v.shape[1]))
    tree3,d_huff = huffman.huffman_encoding_func(array_list(d, d.shape[0], d.shape[1]))

    # rle binaire

    h_huff_rle = algo_RLE.rle_binaire(h_huff)
    v_huff_rle = algo_RLE.rle_binaire(v_huff)
    d_huff_rle = algo_RLE.rle_binaire(d_huff)

    #calcule taux

    taux_ = (calcule_taille_rle(h_huff_rle) + calcule_taille_rle(v_huff_rle) + calcule_taille_rle(d_huff_rle)) / (
                img_.shape[0] * img_.shape[1])

    #decompression
    #rel inverse

    h_rle_inverse= algo_RLE.rle_binaire_inverse(h_huff_rle)
    v_rle_inverse = algo_RLE.rle_binaire_inverse(v_huff_rle)
    d_rle_inverse = algo_RLE.rle_binaire_inverse(d_huff_rle)

    # huffman inverse

    h_huff_inverse = huffman.huffman_decoding_func(h_rle_inverse,tree1)
    v_huff_inverse = huffman.huffman_decoding_func(h_rle_inverse,tree2)
    d_huff_inverse = huffman.huffman_decoding_func(d_rle_inverse,tree3)
    # ondelette -> image

    h = list_array(h_huff_inverse,h.shape[0],h.shape[1])
    v = list_array(v_huff_inverse,v.shape[0],v.shape[1])
    d = list_array(d_huff_inverse,d.shape[0],d.shape[1])
    coef = a,(h,v,d)
    image = pywt.idwt2(coef,"db1")

    # affichage de l'image

    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.text(0,0,"taux de compression = "+str(taux_)+"  pas = "+str(val[i]))
    plt.xticks([])
    plt.yticks([])
    plt.show()



