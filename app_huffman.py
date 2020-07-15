import math
import huff
import algo_RLE
import algo_quantif
import cv2

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

"""codec = HuffmanCodec.from_frequencies({'e': 100, 'n':20, 'x':1, 'i': 40, 'q':3})
encoded = codec.encode('exeneeeexniqneieini')
codec.print_code_table()"""

img_color=cv2.imread("camera.jpeg")
img_gris=cv2.imread("camera.jpeg",cv2.IMREAD_GRAYSCALE)

shape1=img_color.shape
shape2=img_gris.shape

R = img_color[:, :, 0]
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

(a,h,v,d)=algo_quantif.app_quantif(img_gris)

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

# calcule de l'entropie
#
#print("entrepie = %f",entrepie1)

