import const_bases
import quantif_scalaire
import numpy as np
import math

def copier(r1, img ):
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            r1[i][j] = img[i][j]
            j += 1
        i += 1

    return r1

def sub_divise(img, l, c):
    if l % 4 != 0:
        h = 1
        m = l % 4 # 1
        val = 4 - m   # 3
        k = l   # k= 317
        r1 = np.zeros((l + val, c))
        r1 = copier(r1, img)
        while k < l + val :
            n = 0
            while n < c:
                r1[k][n] = img[l - h][n]
                n = n + 1
            k = k + 1
            h = h + 1

        img = r1

    if c % 4 != 0:
        m = c % 4
        val = 4 - m
        k = 0
        r2 = np.zeros((img.shape[0], c + val))
        r2 = copier(r2, img)
        while k < img.shape[0]:
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
                j += 1

            i = i + 1

        dc = dc + 4
        if dc == shape[1] :
            dc = 0
            dl = dl + 4

    return result

def produit_scalaire(f1,f2,f3, b):
    i = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    while i < 4:
        j = 0
        while j < 4:
            sum1 += (f1[i][j] * b[i][j])
            sum2 += (f2[i][j] * b[i][j])
            sum3 += (f3[i][j] * b[i][j])
            j += 1

        i += 1
    return sum1,sum2,sum3

def fct(f1,f2,f3,b) :
    produit1 = np.zeros((4, 4))
    produit2 = np.zeros((4, 4))
    produit3 = np.zeros((4, 4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
             val1, val2 , val3 = produit_scalaire(f1,f2,f3,b[str(i) + str(j)])
             produit1[i][j] = val1
             produit2[i][j] = val2
             produit3[i][j] = val3

             j += 1
        i += 1


    return  produit1,produit2,produit3

def distorsion(f1,f2,f3,fq1,fq2,fq3):
    d1 = d2 = d3 = 0
    i = 0
    while i < 4 :
        j = 0
        while j < 4 :
            d1 += math.pow(f1[i][j] - fq1[i][j], 2)
            d2 += math.pow(f2[i][j] - fq2[i][j], 2)
            d3 += math.pow(f3[i][j] - fq3[i][j], 2)
            j +=  1
        i += 1
    return d1 , d2 , d3

def Rb(i):
    if i == 16 :
        Prb = 1/2
    else :
        Prb = 0.5/16

    rb = - math.log2(Prb)

    return rb

def Rc(b1,b2,b3,h1,h2,h3) :
    rc1 = rc2 = rc3 = 0
    i = 0
    while i < 4 :
        j = 0
        while j < 4 :
            if str(b1[i][j]) in h1 :
                rc1 -= h1[str(b1[i][j])]
            if str(b2[i][j]) in h2 :
                rc2 -= h2[str(b2[i][j])]
            if str(b3[i][j]) in h3 :
                rc3 -= h3[str(b3[i][j])]
            j += 1
        i += 1

    return rc1,rc2,rc3
def somme(m1,m2) :
    s = np.zeros((4,4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            s[i][j] =m1[i][j] + m2[i][j]
            j = j + 1
        i = i + 1
    return s
def projection_inverse(fb1,base1) :
    f1 = np.zeros((4, 4))
    val = np.zeros((4, 4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            # produit
            # debut
            k = 0
            b1 = base1[str(i) + str(j)]
            while k < 4:
                l = 0
                while l < 4:
                    val[k][l] = b1[k][l] * fb1[i][j]
                    l = l + 1
                k = k + 1

            f1 = somme(f1, val)
            # fin
            j = j + 1
        i = i + 1

    return f1
def best_base(block_h,block_v,block_d,h_h,v_h,d_h,pas,bases) :
    L_h = []
    L_v = []
    L_d = []
    fs_h = []
    fs_v = []
    fs_d = []
    λ0 = 6.5
    # multiplicateur de Lagrange
    λ = (3 / 4 * λ0) * math.pow(pas, 2)
    #λ = math.log()
    #λ = 0.15 * math.pow(pas, 2)
    for (b,i) in bases :
        """if i == 1 :
            print(b)
            print(block_h)"""
        #produit scalaire entre la base et le block
        block_hb,block_vb,block_db = fct(block_h,block_v,block_d,b)
        #print(block_h)
        #print(block_hb)
        #print("****************")
        """if i == 1:
            print(block_hb)
            print(projection_inverse(block_hb,b))"""
        #print(block_h)
        #print(block_hb)
        #print("****************")
        # quatification
        fh = quantif_scalaire.quantif(block_hb, pas)
        fv = quantif_scalaire.quantif(block_vb, pas)
        fd = quantif_scalaire.quantif(block_db, pas)
        #print(fh)
        rb = Rb(i)
        #****************
        d1 , d2 , d3 = distorsion(block_hb,block_vb,block_db,quantif_scalaire.val_quantif(fh,pas),quantif_scalaire.val_quantif(fv,pas),quantif_scalaire.val_quantif(fd,pas))
        #************
        rc1 ,rc2 ,rc3 = Rc(fh,fv,fd,h_h,v_h,d_h)

        #*************
        L_h.append(d1 +(λ * (rb + rc1)) )
        fs_h.append(fh)
        #*******************
        L_v.append(d2 + (λ *(rb + rc2)))
        fs_v.append(fv)
        #**********************
        L_d.append(d3 + (λ *(rb + rc3)))
        fs_d.append(fd)

    index_h = L_h.index(min(L_h)) + 1
    M_block_h = fs_h[index_h-1]

    #**************
    index_v = L_v.index(min(L_v)) +1
    M_block_v = fs_v[index_v-1]
    #*************************
    index_d = L_d.index(min(L_d)) +1
    M_block_d = fs_d[index_d-1]
    return M_block_h , M_block_v , M_block_d ,index_h , index_v , index_d

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
    # les bases derictionnel
    base_dire = const_bases.bases_directionnel()
    """for be,ix in base_dire :
        print("*****************     base numero ",ix,"   *******************")
        print(be["00"])
        print(be["01"])
        print(be["02"])
        print(be["03"])
        print(be["10"])
        print(be["11"])
        print(be["12"])
        print(be["13"])
        print(be["20"])
        print(be["21"])
        print(be["22"])
        print(be["23"])
        print(be["30"])
        print(be["31"])
        print(be["32"])
        print(be["33"])"""

    # h1 , h2 = const_bases.base_H()
    # dct = const_bases.base_DCT()
    # calcule le histogramme true

    h_h, v_h, d_h = histogramme(h, v, d, q)

    h_b = []
    h_index =[]
    v_b = []
    v_index = []
    d_b = []
    d_index = []

    #subdivise les blocks true

    blocks_h, shape_h = sub_divise(h, h.shape[0], h.shape[1])
    blocks_v, shape_v = sub_divise(v, v.shape[0], v.shape[1])
    blocks_d, shape_d = sub_divise(d, d.shape[0], d.shape[1])

    for bh,bv,bd in zip(blocks_h,blocks_v,blocks_d) :

        bh_m , bv_m ,bd_m , ih ,iv , id = best_base(bh,bv,bd,h_h,v_h,d_h,q,base_dire)

        h_b.append(bh_m)
        h_index.append(ih)

        v_b.append(bv_m)
        v_index.append(iv)

        d_b.append(bd_m)
        d_index.append(id)

    h_b = quantif_scalaire.val_quantif(combiner(h_b,shape_h),q)
    v_b = quantif_scalaire.val_quantif(combiner(v_b,shape_v),q)
    d_b = quantif_scalaire.val_quantif(combiner(d_b,shape_d),q)



    return (h_b,v_b,d_b),(h_index,v_index,d_index)


#test bandelette
a = np.array([[1,2,3,4],[6,7,1,9],[11,12,1,14],[16,17,18,19]])
#print(a)
#f = [[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,1]]
#subdivision
s,shape = sub_divise(a,a.shape[0],a.shape[1])
"""for e in s :
    print("**************************************************")
    print(e)"""
#print(s)
#print(shape)
#c = combiner(s,shape)
#print(c)
"""
print("**************************************************")
print("***********        subdivise       ***************")
print("**************************************************")
for e in s :
    print("**************************************************")
    print(e)"""
#print(c)
#a = cv2.imread("bebe.jpg",cv2.IMREAD_GRAYSCALE)
#print(a)
#print(s)
"""h1,h2,h3 = histogramme(a,a,a,3)
print("**************************************************")
print("***********          histogramme   ***************")
print("**************************************************")
print(h1)
"""
"""
band, bases = bandelette(a,a,a,3)
print("**************************************************")
print("***********          result        ***************")
print("**************************************************")
print(band[0])
print(bases[0])
print("**************************************************")
print(band[1])
print(bases[1])
print("**************************************************")
print(band[2])
print(bases[2])"""
