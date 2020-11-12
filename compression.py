import cv2
import math
import numpy as np
import pywt
import bandelette_1
import bandelette_inverse
import huff
import algo_RLE
import quantif_scalaire
import matplotlib.pyplot as plt

def afficher(img) :
    i = 0
    shape = img.shape
    while i < shape[0] :
        j = 0
        while j < shape[1] :
            print("img[",i,"]","[",j,"]",img[i][j])
            if img[i][j] != 0 :
                return False
            j += 1
        i += 1
    return True
def calcule_PSNR(max,mse) :
    pnsr = 10 * math.log(math.pow(max,2)/mse,10)
    return pnsr
def s(i1,i2) :
    i=0
    val = 0
    shape = i1.shape
    while i < abs(shape[0]) :
        j = 0
        while j < abs(shape[1]) :
            val += pow(i1[i][j]-i2[i][j],2)
            j += 1
        i += 1
    return val
def calcule_mse(img1,img2,n):
    mse = s(img1,img2)/n
    return  mse
def calcule_taille_rle(list):
    taille=  len(list) * 8
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
            mat[i][j] = float(s[k])
            j = j + 1
            k = k+1
        i = i + 1
    return mat

def old_compression(coeffs,shape,q) :
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]

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

    code11, h1_huff = huff.huffman_encoding_func(array_list(h1_quantif, h1_quantif.shape[0], h1_quantif.shape[1]))
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

    taux =  (calcule_taille_rle(rle)) / (shape[0]*shape[1])
    print("********old**************")
    print("old taux bpp ", taux)
    print("old taux % ", (1 - taux) * 100)
    r = [rle,len(code11),len(code12),len(code13),len(code21),len(code22),len(code23),len(code31),len(code32),len(code33),h1.shape,h2.shape,h3.shape,huff_tree,a1]
    return r

def old_decompression(binaire,q) :
    # rel inverse

    rle = binaire[0]
    rle_inverse = algo_RLE.rle_binaire_inverse(rle)

    l = binaire[1]

    h1_rle_inverse = rle_inverse[0:l]

    v1_rle_inverse = rle_inverse[l:l+binaire[2]]

    l += binaire[2]

    d1_rle_inverse = rle_inverse[l:l+binaire[3]]

    l += binaire[3]

    h2_rle_inverse = rle_inverse[l:l+binaire[4]]
    l += binaire[4]
    v2_rle_inverse = rle_inverse[l:l+binaire[5]]
    l += binaire[5]
    d2_rle_inverse = rle_inverse[l:l+binaire[6]]

    l+= binaire[6]
    h3_rle_inverse = rle_inverse[l:l+binaire[7]]
    l += binaire[7]
    v3_rle_inverse = rle_inverse[l:l+binaire[8]]
    l += binaire[8]
    d3_rle_inverse = rle_inverse[l:l+binaire[9]]

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

    image = pywt.waverec2([binaire[14],(h3,v3,d3),(h2,v2,d2),(h1,v1,d1)],"db3")

    #print(image)
    #plt.imshow(image, cmap=plt.cm.gray)
    #plt.xticks([])
    #plt.yticks([])
    #plt.show()

    return image

def new_compression(coeffs,shape,q) :
    a1 = coeffs[0]
    (h1, v1, d1) = coeffs[-1]
    (h2, v2, d2) = coeffs[-2]
    (h3, v3, d3) = coeffs[-3]

    # bandelette

    bandelette1,base1 = bandelette_1.bandelette(h1, v1, d1, q)

    # h1_bandelette = bandelette.bandelette(h1,h1.shape[0],h1.shape[1],q)
    # v1_bandelette = bandelette.bandelette(v1,v1.shape[0],v1.shape[1],q)
    # d1_bandelette = bandelette.bandelette(d1,d1.shape[0],d1.shape[1],q)
    bandelette2,base2 = bandelette_1.bandelette(h2, v2, d2, q)
    # h2_bandelette = bandelette.bandelette(h2,h2.shape[0],h2.shape[1],q)
    # v2_bandelette = bandelette.bandelette(v2,v2.shape[0],v2.shape[1],q)
    # d2_bandelette = bandelette.bandelette(d2,d2.shape[0],d2.shape[1],q)

    bandelette3,base3 = bandelette_1.bandelette(h3, v3, d3, q)
    # h3_bandelette = bandelette.bandelette(h3,h3.shape[0],h3.shape[1],q)
    # v3_bandelette = bandelette.bandelette(v3,v3.shape[0],v3.shape[1],q)
    # d3_bandelette = bandelette.bandelette(d3,d3.shape[0],d3.shape[1],q)
    base = [base1,base2,base3]
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
    huff_tree = [h1_huff , v1_huff , d1_huff , h2_huff , v2_huff , d2_huff , h3_huff ,  v3_huff , d3_huff]
    # rle binaire
    rle = algo_RLE.rle_binaire(code11 + code12 + code13 + code21 + code22 + code23 + code31 + code32 + code33)

    taux =    (calcule_taille_rle(rle)) / (shape[0]*shape[1])
    print("********neew**************")
    print("new taux bpp ",taux)
    print("new taux % ",(1 - taux)*100)

    return  [rle,len(code11),len(code12),len(code13),len(code21),len(code22),len(code23),len(code31),len(code32),len(code33),h1.shape,h2.shape,h3.shape,huff_tree,a1,base, bandelette1[0].shape, bandelette2[0].shape, bandelette3[0].shape]

def new_decompression(binaire) :
    # rel inverse
    rle = binaire[0]
    rle_inverse = algo_RLE.rle_binaire_inverse(rle)
    l = binaire[1]
    h1_rle_inverse = rle_inverse[0:l]

    v1_rle_inverse = rle_inverse[l:l+binaire[2]]

    l += binaire[2]

    d1_rle_inverse = rle_inverse[l:l+binaire[3]]

    l += binaire[3]

    h2_rle_inverse = rle_inverse[l:l+binaire[4]]

    l += binaire[4]

    v2_rle_inverse = rle_inverse[l:l+binaire[5]]

    l += binaire[5]

    d2_rle_inverse = rle_inverse[l:l+binaire[6]]

    l += binaire[6]

    h3_rle_inverse = rle_inverse[l:l+binaire[7]]

    l += binaire[7]

    v3_rle_inverse = rle_inverse[l:l+binaire[8]]

    l += binaire[8]

    d3_rle_inverse = rle_inverse[l:l+binaire[9]]

    # huffman inverse
    huff_tree = binaire[13]

    h1_huff_inverse = huff.huffman_decoding_func(h1_rle_inverse, huff_tree[0])
    v1_huff_inverse = huff.huffman_decoding_func(v1_rle_inverse, huff_tree[1])
    d1_huff_inverse = huff.huffman_decoding_func(d1_rle_inverse, huff_tree[2])

    h2_huff_inverse = huff.huffman_decoding_func(h2_rle_inverse, huff_tree[3])
    v2_huff_inverse = huff.huffman_decoding_func(v2_rle_inverse, huff_tree[4])
    d2_huff_inverse = huff.huffman_decoding_func(d2_rle_inverse, huff_tree[5])

    h3_huff_inverse = huff.huffman_decoding_func(h3_rle_inverse, huff_tree[6])
    v3_huff_inverse = huff.huffman_decoding_func(v3_rle_inverse, huff_tree[7])
    d3_huff_inverse = huff.huffman_decoding_func(d3_rle_inverse, huff_tree[8])

    # print("decomprese")
    shape1 = binaire[16]
    shape2 = binaire[17]
    shape3 = binaire[18]

    h1_bandelette = list_array(h1_huff_inverse, shape1[0], shape1[1])
    v1_bandelette = list_array(v1_huff_inverse, shape1[0], shape1[1])
    d1_bandelette = list_array(d1_huff_inverse, shape1[0], shape1[1])



    h2_bandelette = list_array(h2_huff_inverse, shape2[0], shape2[1])
    v2_bandelette = list_array(v2_huff_inverse, shape2[0], shape2[1])
    d2_bandelette = list_array(d2_huff_inverse, shape2[0], shape2[1])


    h3_bandelette = list_array(h3_huff_inverse, shape3[0], shape3[1])
    v3_bandelette = list_array(v3_huff_inverse, shape3[0], shape3[1])
    d3_bandelette = list_array(d3_huff_inverse, shape3[0], shape3[1])



    base = binaire[15]
    shape1_o = binaire[10]
    shape2_o = binaire[11]
    shape3_o = binaire[12]

    h1, v1, d1 = bandelette_inverse.inverse(h1_bandelette,v1_bandelette, d1_bandelette,base[0],shape1_o)
    h2, v2, d2 = bandelette_inverse.inverse(h2_bandelette, v2_bandelette, d2_bandelette, base[1],shape2_o)
    h3, v3, d3 = bandelette_inverse.inverse(h3_bandelette, v3_bandelette, d3_bandelette, base[2],shape3_o)

    image = pywt.waverec2([binaire[14], (h3, v3, d3), (h2, v2, d2), (h1, v1, d1)], "db3")

    #afficher(image)
    plt.imshow(image,cmap="gray")
    plt.show()

    return image

image1 = cv2.imread("2.bmp",cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("3.bmp",cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread("5.bmp",cv2.IMREAD_GRAYSCALE)

"""
#plt.imshow(image, cmap=plt.cm.gray)
#plt.xticks([])
#plt.yticks([])
#plt.show()
#ondelette """
coeffs1 = pywt.wavedec2(image1,"db3", level=3)
coeffs2 = pywt.wavedec2(image2,"db3", level=3)
coeffs3 = pywt.wavedec2(image3,"db3", level=3)
"""
arr, coeff = pywt.coeffs_to_array(coeffs)

plt.figure(figsize=(20, 20))
plt.imshow(arr, cmap=plt.cm.gray)
plt.xticks([])
plt.yticks([])
plt.show()
"""
#

img = [image1,image2,image3]
img_test = [coeffs1,coeffs2,coeffs3]
val_quantif =[10,15,20,25,30,35,40,45,50]
#val_quantif = [55,60]
"""
binaire = old_compression(coeffs1,image1.shape, 30)
image_dec1 = old_decompression(binaire,25)
mse1 = calcule_mse(image1,image_dec1, abs(image1.shape[0]) * abs(image1.shape[1]))
print("old mse ", mse1)
print("old psnr ", calcule_PSNR(image1.max(), mse1))
binaire = new_compression(coeffs1, image1.shape, 30)
image_dec2 = new_decompression(binaire)
mse2 = calcule_mse(image1, image_dec2, abs(image1.shape[0]) * abs(image1.shape[1]))
print("new mse ", mse2)
print("new psnr ", calcule_PSNR(image1.max(), mse2))

"""
for coeffs,image in zip(img_test,img) :
    print("**********************************************")
    print("**********************",image.shape,"************************")
    print("**********************************************")
    for pas in val_quantif:
        print("************** ", pas, " **************")
        """binaire = old_compression(coeffs,image.shape, pas)
        image_dec1 = old_decompression(binaire,pas)
        mse1 = calcule_mse(image,image_dec1, abs(image.shape[0]) * abs(image.shape[1]))
        print("old mse ", mse1)
        print("old psnr ", calcule_PSNR(image.max(), mse1))"""
        binaire = new_compression(coeffs, image.shape, pas)
        image_dec2 = new_decompression(binaire)
        mse2 = calcule_mse(image, image_dec2, abs(image.shape[0]) * abs(image.shape[1]))
        print("new mse ", mse2)
        print("new psnr ", calcule_PSNR(image.max(), mse2))


