import cv2
import numpy as np
import pywt
import bandelette
import bandelette_inverse
import huff
import algo_RLE
import quantif_scalaire

def calcule_taille_rle(list):
    taille= 8 * len(list)
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

def old_compression(coeffs) :
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]
    q = 5
    h1_quantif = quantif_scalaire.quantif(h1, q)
    v1_quantif = quantif_scalaire.quantif(v1, q)
    d1_quantif = quantif_scalaire.quantif(d1, q)

    h2_quantif = quantif_scalaire.quantif(h2, q)
    v2_quantif = quantif_scalaire.quantif(v2, q)
    d2_quantif = quantif_scalaire.quantif(d2, q)

    h3_quantif = quantif_scalaire.quantif(h3, q)
    v3_quantif = quantif_scalaire.quantif(v3, q)
    d3_quantif = quantif_scalaire.quantif(d3, q)

    # huffman

    code11, h1_huff = huff.huffman_encoding_func(array_list(h1_quantif, h1.shape[0], h1.shape[1]))
    code12, v1_huff = huff.huffman_encoding_func(array_list(v1_quantif, v1_quantif.shape[0], v1_quantif.shape[1]))
    code13, d1_huff = huff.huffman_encoding_func(array_list(d1, d1.shape[0], d1.shape[1]))

    code21, h2_huff = huff.huffman_encoding_func(array_list(h2_quantif, h2_quantif.shape[0], h2_quantif.shape[1]))
    code22, v2_huff = huff.huffman_encoding_func(array_list(v2_quantif, v2_quantif.shape[0], v2_quantif.shape[1]))
    code23, d2_huff = huff.huffman_encoding_func(array_list(d2_quantif, d2_quantif.shape[0], d2_quantif.shape[1]))

    code31, h3_huff = huff.huffman_encoding_func(array_list(h3_quantif, h3_quantif.shape[0], h3_quantif.shape[1]))
    code32, v3_huff = huff.huffman_encoding_func(array_list(v3_quantif, v3_quantif.shape[0], v3_quantif.shape[1]))
    code33, d3_huff = huff.huffman_encoding_func(array_list(d3_quantif, d3_quantif.shape[0], d3_quantif.shape[1]))

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

    # calcule taux

    t1 = calcule_taille_rle(h1_huff_rle) + calcule_taille_rle(v1_huff_rle) + calcule_taille_rle(d1_huff_rle)
    t2 = calcule_taille_rle(h2_huff_rle) + calcule_taille_rle(v2_huff_rle) + calcule_taille_rle(d2_huff_rle)
    t3 = calcule_taille_rle(h3_huff_rle) + calcule_taille_rle(v3_huff_rle) + calcule_taille_rle(d3_huff_rle)

    return t1 + t2 + t3 ,

def new_compression(coeffs) :
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]
    q = 5
    # bandelette

    bandelette1 = bandelette.bandelette(h1, v1, d1, q)
    print("finish1")
    # h1_bandelette = bandelette.bandelette(h1,h1.shape[0],h1.shape[1],q)
    # v1_bandelette = bandelette.bandelette(v1,v1.shape[0],v1.shape[1],q)
    # d1_bandelette = bandelette.bandelette(d1,d1.shape[0],d1.shape[1],q)

    bandelette2 = bandelette.bandelette(h2, v2, d2, q)

    # h2_bandelette = bandelette.bandelette(h2,h2.shape[0],h2.shape[1],q)
    # v2_bandelette = bandelette.bandelette(v2,v2.shape[0],v2.shape[1],q)
    # d2_bandelette = bandelette.bandelette(d2,d2.shape[0],d2.shape[1],q)

    bandelette3 = bandelette.bandelette(h1, v1, d1, q)

    # h3_bandelette = bandelette.bandelette(h3,h3.shape[0],h3.shape[1],q)
    # v3_bandelette = bandelette.bandelette(v3,v3.shape[0],v3.shape[1],q)
    # d3_bandelette = bandelette.bandelette(d3,d3.shape[0],d3.shape[1],q)

    # huffman

    code11, h1_huff = huff.huffman_encoding_func(
        array_list(bandelette1[0], bandelette1[0].shape[0], bandelette1[0].shape[1]))
    code12, v1_huff = huff.huffman_encoding_func(
        array_list(bandelette1[1], bandelette1[1].shape[0], bandelette1[1].shape[1]))
    code13, d1_huff = huff.huffman_encoding_func(
        array_list(bandelette1[2], bandelette1[2].shape[0], bandelette1[2].shape[1]))

    code21, h2_huff = huff.huffman_encoding_func(
        array_list(bandelette2[0], bandelette2[0].shape[0], bandelette2[0].shape[1]))
    code22, v2_huff = huff.huffman_encoding_func(
        array_list(bandelette2[1], bandelette2[1].shape[0], bandelette2[1].shape[1]))
    code23, d2_huff = huff.huffman_encoding_func(
        array_list(bandelette2[2], bandelette2[2].shape[0], bandelette2[2].shape[1]))

    code31, h3_huff = huff.huffman_encoding_func(
        array_list(bandelette3[0], bandelette3[0].shape[0], bandelette3[0].shape[1]))
    code32, v3_huff = huff.huffman_encoding_func(
        array_list(bandelette3[1], bandelette3[1].shape[0], bandelette3[1].shape[1]))
    code33, d3_huff = huff.huffman_encoding_func(
        array_list(bandelette3[2], bandelette3[2].shape[0], bandelette3[2].shape[1]))

    # rle binaire
    rle = algo_RLE.rle_binaire(code11 + code12 + code13 + code21 + code22 + code23 + code31 + code32 + code33)

    """


    h1_huff_rle = algo_RLE.rle_binaire(code11)
    v1_huff_rle = algo_RLE.rle_binaire(code12)
    d1_huff_rle = algo_RLE.rle_binaire(code13)

    h2_huff_rle = algo_RLE.rle_binaire(code21)
    v2_huff_rle = algo_RLE.rle_binaire(code22)
    d2_huff_rle = algo_RLE.rle_binaire(code23)

    h3_huff_rle = algo_RLE.rle_binaire(code31)
    v3_huff_rle = algo_RLE.rle_binaire(code32)
    d3_huff_rle = algo_RLE.rle_binaire(code33)

    #calcule de taux
    #"""
    t = calcule_taille_rle(rle)
    """
    t1=calcule_taille_rle(h1_huff_rle) + calcule_taille_rle(v1_huff_rle) + calcule_taille_rle(d1_huff_rle)
    t2=calcule_taille_rle(h2_huff_rle) + calcule_taille_rle(v2_huff_rle) + calcule_taille_rle(d2_huff_rle)
    t3=calcule_taille_rle(h3_huff_rle) + calcule_taille_rle(v3_huff_rle) + calcule_taille_rle(d3_huff_rle) """
    taux_ = t / (image.shape[0] * image.shape[1])
    # ((t1+t2+t3) / (image.shape[0] * image.shape[1])) #bit/pixel """
    # rle = algo_RLE.rle_binaire(code00+code11+code12+code13+code21+code22+code23+code31+code32+code33)
    # taux_ = calcule_taille_rle(rle)/(image.shape[0] * image.shape[1])
    print(taux_, " bit/pixel")
    return  rle



image = cv2.imread("a2.png",cv2.IMREAD_GRAYSCALE)
#ondelette
coeffs = pywt.wavedec2(image, "db1", mode="periodization", level=3)

# decompression
# rel inverse
"""


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
# print("decomprese")

h1 = list_array(h1_huff_inverse,h1.shape[0],h1.shape[1])
v1 = list_array(v1_huff_inverse,v1.shape[0],v1.shape[1])
d1 = list_array(d1_huff_inverse,d1.shape[0],d1.shape[1])

h2 = list_array(h2_huff_inverse, h2.shape[0], h2.shape[1])
v2 = list_array(v2_huff_inverse, v2.shape[0], v2.shape[1])
d2 = list_array(d2_huff_inverse, d2.shape[0], d2.shape[1])

h3 = list_array(h3_huff_inverse, h3.shape[0], h3.shape[1])
v3 = list_array(v3_huff_inverse, v3.shape[0], v3.shape[1])
d3 = list_array(d3_huff_inverse, d3.shape[0], d3.shape[1])

# bandelette inverse

h1_b_inv = d3_huff

coeffs[0] = a1
coeffs[-1] = (h1, v1, d1)
coeffs[-2] = (h2, v2, d2)
coeffs[-3] = (h3, v3, d3)
print("ondelette inverse")
image = pywt.waverec2(coeffs,"db1")

image = cv2.imshow("image aprés la compression",image)

"""