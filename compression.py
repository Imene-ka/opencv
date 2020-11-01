import cv2
import numpy as np
import pywt
import bandelette_1
import bandelette_inverse
import huff
import algo_RLE
import quantif_scalaire

def calcule_taille_rle(list):
    taille= 8 * len(list)
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
            mat[i][j] = s[k]
            j = j + 1
            k = k+1
            len(s)
        i = i + 1


    return mat

def old_compression(coeffs,shape) :
    print(type(coeffs))
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]
    q = 20
    h1_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(h1, q),q)
    v1_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(v1, q),q)
    d1_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(d1, q),q)

    h2_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(h2, q),q)
    v2_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(v2, q),q)
    d2_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(d2, q),q)

    h3_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(h3, q),q)
    v3_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(v3, q),q)
    d3_quantif = quantif_scalaire.val_quantif(quantif_scalaire.quantif(d3, q),q)

    # huffman

    code11, h1_huff = huff.huffman_encoding_func(array_list(h1_quantif, h1.shape[0], h1.shape[1]))
    code12, v1_huff = huff.huffman_encoding_func(array_list(v1_quantif, v1_quantif.shape[0], v1_quantif.shape[1]))
    code13, d1_huff = huff.huffman_encoding_func(array_list(d1_quantif, d1_quantif.shape[0], d1_quantif.shape[1]))

    code21, h2_huff = huff.huffman_encoding_func(array_list(h2_quantif, h2_quantif.shape[0], h2_quantif.shape[1]))
    code22, v2_huff = huff.huffman_encoding_func(array_list(v2_quantif, v2_quantif.shape[0], v2_quantif.shape[1]))
    code23, d2_huff = huff.huffman_encoding_func(array_list(d2_quantif, d2_quantif.shape[0], d2_quantif.shape[1]))

    code31, h3_huff = huff.huffman_encoding_func(array_list(h3_quantif, h3_quantif.shape[0], h3_quantif.shape[1]))
    code32, v3_huff = huff.huffman_encoding_func(array_list(v3_quantif, v3_quantif.shape[0], v3_quantif.shape[1]))
    code33, d3_huff = huff.huffman_encoding_func(array_list(d3_quantif, d3_quantif.shape[0], d3_quantif.shape[1]))
    huff_tree = [h1_huff,v1_huff,d1_huff,h2_huff,v2_huff,d2_huff,h3_huff,v3_huff,d3_huff]
    # rle binaire

    rle = algo_RLE.rle_binaire(code11+code12+code13+code21+code22+code23+code31+code32+code33)

    #h1_huff_rle = algo_RLE.rle_binaire(code11)
    #v1_huff_rle = algo_RLE.rle_binaire(code12)
    #d1_huff_rle = algo_RLE.rle_binaire(code13)

    #h2_huff_rle = algo_RLE.rle_binaire(code21)
    #v2_huff_rle = algo_RLE.rle_binaire(code22)
    #d2_huff_rle = algo_RLE.rle_binaire(code23)

    #h3_huff_rle = algo_RLE.rle_binaire(code31)
    #v3_huff_rle = algo_RLE.rle_binaire(code32)
    #d3_huff_rle = algo_RLE.rle_binaire(code33)

    # calcule taux

    #t1 = calcule_taille_rle(h1_huff_rle) + calcule_taille_rle(v1_huff_rle) + calcule_taille_rle(d1_huff_rle)
    #t2 = calcule_taille_rle(h2_huff_rle) + calcule_taille_rle(v2_huff_rle) + calcule_taille_rle(d2_huff_rle)
    #t3 = calcule_taille_rle(h3_huff_rle) + calcule_taille_rle(v3_huff_rle) + calcule_taille_rle(d3_huff_rle)

    taux = calcule_taille_rle(rle) / (shape[0]*shape[1]*8)
    print(taux)
    r = [rle,len(code11),len(code12),len(code13),len(code21),len(code22),len(code23),len(code31),len(code32),len(code33),h1.shape,h2.shape,h3.shape,huff_tree,coeffs]
    return r

def old_decompression(binaire) :
    # rel inverse
    print("decompression")
    rle = binaire[0]
    rle_inverse = algo_RLE.rle_binaire_inverse(rle)

    l = binaire[1]
    h1_rle_inverse = rle_inverse[0:l]
    l += binaire[2]
    v1_rle_inverse = rle_inverse[binaire[1]:l]
    l += binaire[3]
    d1_rle_inverse = rle_inverse[binaire[2]:l]

    l += binaire[4]

    h2_rle_inverse = rle_inverse[binaire[3]:l]
    l += binaire[5]
    v2_rle_inverse = rle_inverse[binaire[4]:l]
    l += binaire[6]
    d2_rle_inverse = rle_inverse[binaire[5]:l]

    l+= binaire[7]
    h3_rle_inverse = rle_inverse[binaire[6]:l]
    l += binaire[8]
    v3_rle_inverse = rle_inverse[binaire[7]:l]
    l += binaire[9]
    d3_rle_inverse = rle_inverse[binaire[8]:l]

    # huffman inverse
    huff_tree = binaire[13]

    h1_huff_inverse = huff.huffman_decoding_func(h1_rle_inverse,huff_tree[0])
    v1_huff_inverse = huff.huffman_decoding_func(v1_rle_inverse,huff_tree[1])
    d1_huff_inverse = huff.huffman_decoding_func(d1_rle_inverse,huff_tree[2])

    h2_huff_inverse = huff.huffman_decoding_func(h2_rle_inverse,huff_tree[3])
    v2_huff_inverse = huff.huffman_decoding_func(v2_rle_inverse,huff_tree[4])
    d2_huff_inverse = huff.huffman_decoding_func(d2_rle_inverse,huff_tree[5])

    h3_huff_inverse = huff.huffman_decoding_func(h3_rle_inverse,huff_tree[6])
    v3_huff_inverse = huff.huffman_decoding_func(v3_rle_inverse,huff_tree[7])
    d3_huff_inverse = huff.huffman_decoding_func(d3_rle_inverse,huff_tree[8])

    # print("decomprese")
    shape1 = binaire[10]
    shape2 = binaire[11]
    shape3 = binaire[12]

    h1 = list_array(h1_huff_inverse,shape1[0],shape1[1])
    v1 = list_array(v1_huff_inverse,shape1[0],shape1[1])
    d1 = list_array(d1_huff_inverse,shape1[0],shape1[1])


    h2 = list_array(h2_huff_inverse, shape2[0], shape2[1])
    v2 = list_array(v2_huff_inverse, shape2[0], shape2[1])
    d2 = list_array(d2_huff_inverse, shape2[0], shape2[1])


    h3 = list_array(h3_huff_inverse, shape3[0], shape3[1])
    v3 = list_array(v3_huff_inverse, shape3[0], shape3[1])
    d3 = list_array(d3_huff_inverse, shape3[0], shape3[1])

    coeff = binaire[14]
    coeff[0]
    coeff[-3] = (h3, v3, d3)
    coeff[-2] = (h2, v2, d2)
    coeff.append((h1, v1, d1))



    image = pywt.waverec2(coeffs,"db2")

    cv2.imshow("image aprés compression",image)


def new_compression(coeffs,shape) :
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]
    q = 20
    # bandelette

    bandelette1 = bandelette_1.bandelette(h1, v1, d1, q)
    print("finish1")
    # h1_bandelette = bandelette.bandelette(h1,h1.shape[0],h1.shape[1],q)
    # v1_bandelette = bandelette.bandelette(v1,v1.shape[0],v1.shape[1],q)
    # d1_bandelette = bandelette.bandelette(d1,d1.shape[0],d1.shape[1],q)

    bandelette2 = bandelette_1.bandelette(h2, v2, d2, q)
    print("finish2")
    # h2_bandelette = bandelette.bandelette(h2,h2.shape[0],h2.shape[1],q)
    # v2_bandelette = bandelette.bandelette(v2,v2.shape[0],v2.shape[1],q)
    # d2_bandelette = bandelette.bandelette(d2,d2.shape[0],d2.shape[1],q)

    bandelette3 = bandelette_1.bandelette(h3, v3, d3, q)
    print("finish3")
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
    taux_ = t / (shape[0] * shape[1]*8)
    # ((t1+t2+t3) / (image.shape[0] * image.shape[1])) #bit/pixel """
    # rle = algo_RLE.rle_binaire(code00+code11+code12+code13+code21+code22+code23+code31+code32+code33)
    # taux_ = calcule_taille_rle(rle)/(image.shape[0] * image.shape[1])
    print(taux_, " bit/pixel")
    return  rle



image = cv2.imread("1.bmp",cv2.IMREAD_GRAYSCALE)
#ondelette
coeffs = pywt.wavedec2(image, "db2", mode="periodization", level=3)

binaire = old_compression(coeffs,image.shape)

old_decompression(binaire)

new_compression(coeffs,image.shape)
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