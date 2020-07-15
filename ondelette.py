import matplotlib.pyplot as plt
import pywt
import cv2
import numpy as np
from pywt._doc_utils import draw_2d_wp_basis, wavedec2_keys


#import numpy as np
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

# Load image
#original = pywt.data.camera()
original=cv2.imread("camera.jpeg",cv2.IMREAD_GRAYSCALE)
shape=original.shape
# Wavelet transform of image, and plot approximation and details
sous_titre=['Approximation', ' Horizontal', 'Vertical ', 'Diagonal ']
ond = ['db1', 'db2','haar','gaus2']
titre = ['Daubechies 1', 'Daubechies 2','haar', 'Gaussian']
print(original)
i=0
while i<3 :
    coeffs = pywt.wavedec2(original,ond[i],mode="periodization", level=4)
    a1 = coeffs[0]
    (h2, v2, d2) = coeffs[-1]
    (h3, v3, d3) = coeffs[-2]
    (h4, v4, d4) = coeffs[-3]

    coeffs[0]=normaliser(a1,a1.shape[1],a1.shape[0])
    coeffs[-1]=(normaliser(h2,h2.shape[1],h2.shape[0]),normaliser(v2,v2.shape[1],v2.shape[0]),normaliser(d2,d2.shape[1],d2.shape[0]))
    coeffs[-2] = (normaliser(h3, h3.shape[1], h3.shape[0]), normaliser(v3, v3.shape[1], v3.shape[0]),normaliser(d3, d3.shape[1], d3.shape[0]))
    coeffs[-3] = (normaliser(h4, h4.shape[1], h4.shape[0]), normaliser(v4, v4.shape[1], v4.shape[0]),normaliser(d4, d4.shape[1], d4.shape[0]))

    arr, coeff = pywt.coeffs_to_array(coeffs)

    cv2.imshow("R", arr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #plt.figure(titre[i],figsize=(20, 20))
    #plt.imshow(arr, cmap=plt.cm.gray)
    #plt.title(titre[i])
    # plt.set_xticks([])
    # plt.set_yticks([])
    #plt.show()
    i=i+1

""""coeffs = pywt.wavedec2(original,ond[i],mode="zero", level=4)
a1 = coeffs[0]
(h2, v2, d2) = coeffs[-1]
(h3, v3, d3) = coeffs[-2]
(h4, v4, d4) = coeffs[-3]
arr, coeff = pywt.coeffs_to_array(coeffs)
#arr = normaliser_arr(arr,arr.shape[0],arr.shape[1])
plt.figure(titre[i],figsize=(20, 20))
plt.imshow(arr, cmap=plt.cm.gray)
plt.title(titre[i])
# plt.set_xticks([])
# plt.set_yticks([])
plt.show()"""
