import pywt
import matplotlib.pyplot as plt
import numpy as np
import cv2

def AB(min,max) :
    a =255/(max-min)
    b =255-((255*max)/(max-min))
    return (a,b)
def val_min(mat,l,c) :
    min=mat[0][0]
    i = 0
    while i < l:
        j = 0
        while j < c:
            if mat[i][j]<min :
                min=mat[i][j]
            j = j + 1
        i = i + 1
    return min
def val_max(mat,l,c) :
    max=mat[0][0]
    i=0
    while i < l:
        j = 0
        while j < c:
            if mat[i][j]>max :
                max=mat[i][j]
            j = j + 1
        i = i + 1
    return max
def normaliser(mat,c,l) :
    i = 0
    result = np.zeros((l, c))
    min=val_min(mat,l,c)
    max=val_max(mat,l,c)
    (a,b)=AB(min,max)
    while i < l:
        j = 0
        while j < c:
            result[i][j]=int(mat[i][j]*a+b)
            j=j+1
        i = i + 1
    return result

def verif(pixel,min,max,pas):
    i=min
    while i<=max :
       if pixel>=i and pixel<=i+pas :
           pixel=(i+i+pas)/2
           break
       i=i+pas
    return pixel

def quantification_pas_constant(img,c,l,pas):
    i=0
    while i<c :
        j=0
        while j<l :
            #print(img[i][j])
            img[i][j]=verif(img[i][j],np.min(img),np.max(img),pas)
            j=j+1
        i=i+1
    return img

def app_quantif(img) :
    pas = (int)(input("entrez le pas :"))
    cof0 = pywt.dwt2(img, "db1")
    a1, (h1, v1, d1) = cof0
    for a in [a1, h1, v1, d1]:
        a = quantification_pas_constant(a, a.shape[0], a.shape[1], pas)

    return a1,h1,v1,d1


"""sous_titre=['Approximation', ' Horizontal', 'Vertical ', 'Diagonal ']
original = pywt.data.camera()
#cv2.imshow("R",original)
shape=original.shape
pas=(int) (input("entrez le pas :"))
cof0 = pywt.dwt2(original,"db1",)
print(cof0)
a1, (h1, v1, d1) = cof0
for a in [a1,h1,v1,d1] :
    a=quantification_pas_constant(a,a.shape[0],a.shape[1],pas)"""

"""fig = plt.figure("algo", figsize=(12,3))

for i, a in enumerate([a1,h1, v1,d1]):
    s=a.shape
    ax = fig.add_subplot(2,2,i+1)
    a=normaliser(a,s[0],s[1])
    ax.imshow(a,cmap=plt.cm.gray)
    ax.set_title(sous_titre[i])
    ax.set_xticks([])
    ax.set_yticks([])

plt.show()"""



