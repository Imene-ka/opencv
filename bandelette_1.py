import const_bases
import quantif_scalaire
import numpy as np
import math
import cv2

def copier(r1, img, l, c):
    i = 0
    while i < l:
        j = 0
        while j < c:
            r1[i][j] = img[i][j]
            j = j + 1
        i = i + 1

    return r1

def sub_divise(img, l, c):
    if l % 4 != 0:
        h = 1
        m = l % 4
        val = 4 - m   # 2
        k = l   # k=50
        r1 = np.zeros((l + val, c))
        r1 = copier(r1, img, l, c)
        while k < l + val :
            n = 0
            while n < c:
                r1[k][n] = img[l - h][n]
                h = h + 1
                n = n + 1
            k = k + 1
            h = 1
        img = r1

    if c % 4 != 0:
        m = c % 4
        val = 4 - m
        k = 0
        r2 = np.zeros((img.shape[0], c + val))
        r2 = copier(r2, img, img.shape[0], img.shape[1])
        while k < l:
            n = c
            h = 1
            while n < c + val:
                r2[k][n] = img[k][c - h]
                h = h + 1
                n = n + 1
            k = k + 1
        img = r2

    i = 0
    result = []
    cmp = 0
    while i < img.shape[0] :

        j = 0
        while j < img.shape[1] :
            h = 0
            if cmp != 16:
                r = np.zeros((4, 4))
            while h < 4:
                k = 0
                while k < 4:
                    r[h][k] = img[i + h][j + k]
                    cmp = cmp + 1

                    k = k + 1

                h = h + 1

            j = j + 4
            if cmp == 16:
                result.append(r)
                cmp = 0
        i = i + 4

    return (result,img.shape)

def combiner(r,shape) :
    result = np.zeros(shape)
    dc =0
    dl =0
    for e in r :
        i = 0
        while i < 4:
            j = 0
            while j < 4:
                result[i+dl][j+dc] = e[i][j]
                j = j + 1

            i = i + 1

        dc = dc + 4
        if dc == shape[1] :
            dc = 0
            dl = dl + 4



    return result

def produit_scalaire(f1,f2,f3, b):
    h = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    while h < 4:
        k = 0
        while k < 4:
            sum1 = sum1 + (f1[h][k] * b[h][k])
            sum2 = sum2 + (f2[h][k] * b[h][k])
            sum3 = sum3 + (f3[h][k] * b[h][k])
            k = k + 1

        h = h + 1
    return sum1,sum2,sum3

def fct(f1,f2,f3,b) :
    produit1 = np.zeros((4, 4))
    produit2 = np.zeros((4, 4))
    produit3 = np.zeros((4, 4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            produit1[i][j],produit2[i][j],produit3[i][j] = produit_scalaire(f1,f2,f3,b[str(i) + str(j)])
            #produit2[i][j] = produit_scalaire(f, b[str(i) + str(j)])
            #produit3[i][j] = produit_scalaire(f, b[str(i) + str(j)])
            j = j + 1
        i = i + 1

    return  produit1,produit2,produit3

def distorsion(f,fq):
    d = 0
    i = 0
    while i < 4 :
        j = 0
        while j < 4 :
            d = d + math.pow(f[i][j]-fq[i][j],2)
            j = j + 1
        i = i + 1
    return d

def Rb(i):
    if i == 16 :
        Prb = 1/2
    else :
        Prb = 0.5/16

    rb = - math.log2(Prb)

    return rb

def proba(val,sous_band) :
    p = np.sum(sous_band==val)
    return p/(sous_band.shape[0]*sous_band.shape[1])

def Rc(b,sous_band) :
    rc = 0
    i = 0
    while i < 4 :
        j = 0
        while j < 4 :
            if str(b[i][j]) in sous_band :
                rc = rc - sous_band[str(b[i][j])]
            j= j + 1
        i = i +1

    return rc

def best_base(block_h,block_v,block_d,h_h,v_h,d_h,pas,bases) :
    L_h = []
    L_v = []
    L_d = []
    fs_h = []
    fs_v = []
    fs_d = []
    λ0 = 6.5
    # multiplicateur de Lagrange
    λ = (3 / (4 * λ0)) * math.pow(pas, 2)
    for b,i in bases :
        #produit scalaire entre la base et le block
        block_hh,block_vv,block_dd = fct(block_h,block_v,block_d,b)
        #block_vv = fct(block_v,b)
        #block_dd = fct(block_d,b)
        print("produit scalaire",i)
        # quatification
        fh = quantif_scalaire.quantif(block_hh, pas)
        fv = quantif_scalaire.quantif(block_vv,pas)
        fd = quantif_scalaire.quantif(block_dd,pas)

        rb = Rb(i)

        L_h.append(distorsion(block_hh,quantif_scalaire.val_quantif(fh,pas)) + λ * (Rc(fh,h_h)) + rb)
        fs_h.append(fh)
        #*******************
        L_v.append(distorsion(block_vv, quantif_scalaire.val_quantif(fv,pas)) + λ * (Rc(fv,v_h)) + rb)
        fs_v.append(fv)
        #**********************
        L_d.append(distorsion(block_dd, quantif_scalaire.val_quantif(fd,pas)) + λ * (Rc(fd,d_h)) + rb)
        fs_d.append(fd)

    index_h = L_h.index(min(L_h)) + 1
    M_block_h = fs_h[index_h-1]
    #**************
    index_v = L_v.index(min(L_v)) + 1
    M_block_v = fs_v[index_v - 1]
    #*************************
    index_d = L_d.index(min(L_d)) + 1
    M_block_d = fs_d[index_d - 1]
    print("terminéé")
    return M_block_h , M_block_v , M_block_d , index_h , index_v , index_d
def pro_log(sous_band) :
    result = {}
    i = 0
    while i < sous_band.shape[0] :
        j = 0
        while j < sous_band.shape[1] :
            if str(sous_band[i][j]) in result :
                result[str(sous_band[i][j])] += 1
            else:
                result[str(sous_band[i][j])] = 1
            j = j + 1
        i = i + 1
    for cle,val in result.items() :
        result[cle] = math.log(val/(sous_band.shape[0]*sous_band.shape[1]),2)
    return result
def histogramme(h1,v1,d1,q) :
    h1_quantif = quantif_scalaire.quantif(h1, q)
    v1_quantif = quantif_scalaire.quantif(v1, q)
    d1_quantif = quantif_scalaire.quantif(d1, q)

    h_h = pro_log(h1_quantif)
    h_v = pro_log(v1_quantif)
    h_d = pro_log(d1_quantif)

    return h_h , h_v , h_d
#input les coefs de ondelette -> coefs transformées et les bases associé
def bandelette(h,v,d,q) :
    base_dire = const_bases.bases_directionnel()
    # h1 , h2 = const_bases.base_H()
    # dct = const_bases.base_DCT()
    # calcule le histogramme

    h_b = []
    h_index =[]
    v_b = []
    v_index = []
    d_b = []
    d_index = []
    #subdivise les blocks

    blocks_h, shape_h = sub_divise(h, h.shape[0],h.shape[1])
    blocks_v, shape_v = sub_divise(v, v.shape[0], v.shape[1])
    blocks_d, shape_d = sub_divise(d, d.shape[0], d.shape[1])

    h_h, v_h, d_h = histogramme(h, v, d, q)

    #print("finish subdivision ",d.shape,v.shape)
    for bh,bv,bd in zip(blocks_h,blocks_v,blocks_d) :
        bh_m , bv_m ,bd_m , ih ,iv ,  id = best_base(bh,bv,bd,h_h,v_h,d_h,q,base_dire)
        #bh_m , ih = best_base(bh,h,q,base_dire)
        #bv_m , iv = best_base(bv,v,q,base_dire)
        #bd_m , id = best_base(bd,d,q,base_dire)
        h_b.append(bh_m)
        h_index.append(ih)
        v_b.append(bv_m)
        v_index.append(iv)
        d_b.append(bd_m)
        d_index.append(id)
       # print("c bon")

    h_b = combiner(h_b,shape_h)
    v_b = combiner(v_b,shape_v)
    d_b = combiner(d_b,shape_d)

    return h_b,v_b,d_b


#test bandelette
#a = np.array([[1,2,3,4,5],[6,7,1,9,10],[11,12,1,14,12],[16,17,18,19,20]])

f = [[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,1]]
#subdivision
#s,shape = sub_divise(a,a.shape[0],a.shape[1])
#print(s)
#print(shape)
#c = combiner(s,shape)
#print(c)
"""a = cv2.imread("bebe.jpg",cv2.IMREAD_GRAYSCALE)
print(a)
s = bandelette(a,a.shape[0],a.shape[1],20)
print(s)"""