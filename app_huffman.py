import math
import algo_RLE
import algo_quantif
import cv2
import pywt
import matplotlib.pyplot as plt
import huff
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
    coeffs = algo_quantif.app_quantif(img_,v)
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]

    # huffman

    code11,h1_huff = huff.huffman_encoding_func(array_list(h1, h1.shape[0], h1.shape[1]))
    code12,v1_huff = huff.huffman_encoding_func(array_list(v1, v1.shape[0], v1.shape[1]))
    code13,d1_huff = huff.huffman_encoding_func(array_list(d1, d1.shape[0], d1.shape[1]))

    code21,h2_huff = huff.huffman_encoding_func(array_list(h2, h2.shape[0], h2.shape[1]))
    code22,v2_huff = huff.huffman_encoding_func(array_list(v2, v2.shape[0], v2.shape[1]))
    code23,d2_huff = huff.huffman_encoding_func(array_list(d2, d2.shape[0], d2.shape[1]))

    code31,h3_huff = huff.huffman_encoding_func(array_list(h3, h3.shape[0], h3.shape[1]))
    code32,v3_huff = huff.huffman_encoding_func(array_list(v3, v3.shape[0], v3.shape[1]))
    code33,d3_huff = huff.huffman_encoding_func(array_list(d3, d3.shape[0], d3.shape[1]))

    # rle binaire

    h1_huff_rle = algo_RLE.rle_binaire(code11)
    v1_huff_rle = algo_RLE.rle_binaire(code12)
    d1_huff_rle = algo_RLE.rle_binaire(code13)

    h2_huff_rle = algo_RLE.rle_binaire(code21)
    v2_huff_rle = algo_RLE.rle_binaire(code22)
    d2_huff_rle = algo_RLE.rle_binaire(code23)

    h3_huff_rle = algo_RLE.rle_binaire(code31)
    v3_huff_rle = algo_RLE.rle_binaire(code32)
    d3_huff_rle = algo_RLE.rle_binaire(code33)

    #calcule taux

    t1=calcule_taille_rle(h1_huff_rle) + calcule_taille_rle(v1_huff_rle) + calcule_taille_rle(d1_huff_rle)
    t2=calcule_taille_rle(h2_huff_rle) + calcule_taille_rle(v2_huff_rle) + calcule_taille_rle(d2_huff_rle)
    t3=calcule_taille_rle(h3_huff_rle) + calcule_taille_rle(v3_huff_rle) + calcule_taille_rle(d3_huff_rle)
    taux_ = ((t1+t2+t3) / (img_.shape[0] * img_.shape[1]))
    print(taux_,"bit/pixel")
    #decompression
    #rel inverse

    a1_rle_inverse = algo_RLE.rle_binaire_inverse(a1_huff_rle)
    h1_rle_inverse = algo_RLE.rle_binaire_inverse(h1_huff_rle)
    v1_rle_inverse = algo_RLE.rle_binaire_inverse(v1_huff_rle)
    d1_rle_inverse = algo_RLE.rle_binaire_inverse(d1_huff_rle)


    h2_rle_inverse = algo_RLE.rle_binaire_inverse(h2_huff_rle)
    v2_rle_inverse = algo_RLE.rle_binaire_inverse(v2_huff_rle)
    d2_rle_inverse = algo_RLE.rle_binaire_inverse(d2_huff_rle)


    h3_rle_inverse = algo_RLE.rle_binaire_inverse(h3_huff_rle)
    v3_rle_inverse = algo_RLE.rle_binaire_inverse(v3_huff_rle)
    d3_rle_inverse = algo_RLE.rle_binaire_inverse(d3_huff_rle)


    # huffman inverse

    a1_huff_inverse = huff.huffman_decoding_func(a1_rle_inverse,a1_huff)

    h1_huff_inverse = huff.huffman_decoding_func(h1_rle_inverse,h1_huff)
    v1_huff_inverse = huff.huffman_decoding_func(v1_rle_inverse,v1_huff)
    d1_huff_inverse = huff.huffman_decoding_func(d1_rle_inverse,d1_huff)


    h2_huff_inverse = huff.huffman_decoding_func(h2_rle_inverse,h2_huff)
    v2_huff_inverse = huff.huffman_decoding_func(v2_rle_inverse,v2_huff)
    d2_huff_inverse = huff.huffman_decoding_func(d2_rle_inverse,d2_huff)


    h3_huff_inverse = huff.huffman_decoding_func(h1_rle_inverse,h3_huff)
    v3_huff_inverse = huff.huffman_decoding_func(v3_rle_inverse,v3_huff)
    d3_huff_inverse = huff.huffman_decoding_func(d3_rle_inverse,d3_huff)


    # ondelette -> image
    print("decomprese")
    h1 = list_array(h1_huff_inverse,h1.shape[0],h1.shape[1])
    v1 = list_array(v1_huff_inverse,v1.shape[0],v1.shape[1])
    d1 = list_array(d1_huff_inverse,d1.shape[0],d1.shape[1])

    h2 = list_array(h2_huff_inverse, h2.shape[0], h2.shape[1])
    v2 = list_array(v2_huff_inverse, v2.shape[0], v2.shape[1])
    d2 = list_array(d2_huff_inverse, d2.shape[0], d2.shape[1])

    h3 = list_array(h3_huff_inverse, h3.shape[0], h3.shape[1])
    v3 = list_array(v3_huff_inverse, v3.shape[0], v3.shape[1])
    d3 = list_array(d3_huff_inverse, d3.shape[0], d3.shape[1])

    coeffs[0] = a1
    coeffs[-1] = (h1, v1, d1)
    coeffs[-2] = (h2, v2, d2)
    coeffs[-3] = (h3, v3, d3)
    print("ondelette inverse")
    image = pywt.waverec2(coeffs,"db1")
    # affichage de l'image

    plt.figure()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.text(0,0,"taux de compression = "+str(taux_)+"  pas = "+str(val[i]))
    plt.xticks([])
    plt.yticks([])
    plt.show()




