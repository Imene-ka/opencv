import numpy as np
import quantif_scalaire
import const_bases

def redimentioner(block,shape) :
    b_r = np.zeros(shape)
    i = 0
    while i < shape[0] :
        j= 0
        while j < shape[1] :
            b_r[i][j] = block[i][j]
            j += 1
        i += 1
    return b_r
def copier(r1, img, l, c):
    i = 0
    while i < l:
        j = 0
        while j < c:
            r1[i][j] = img[i][j]
            j += 1
        i += 1

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
def projection_inverse(fb1,fb2,fb3,base1,base2,base3) :
    f1 = np.zeros((4, 4))
    var1 = np.zeros((4, 4))
    f2 = np.zeros((4, 4))
    var2 = np.zeros((4, 4))
    f3 = np.zeros((4, 4))
    var3 = np.zeros((4, 4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            # produit
            # debut
            k = 0
            b1 = base1[0][str(i) + str(j)]
            b2 = base2[0][str(i) + str(j)]
            b3 = base3[0][str(i) + str(j)]

            while k < 4:
                l = 0
                while l < 4:
                    var1[k][l] = b1[k][l] * fb1[i][j]
                    var2[k][l] = b2[k][l] * fb2[i][j]
                    var3[k][l] = b3[k][l] * fb3[i][j]
                    l = l + 1
                k = k + 1

            f1 = somme(f1, var1)
            f2 = somme(f2, var2)
            f3 = somme(f3, var3)
            # fin
            j = j + 1
        i = i + 1

    return f1,f2,f3
def inverse(h_b,v_b,d_b,b,s) :
    shape = h_b.shape
    k = 0
    h = []
    v = []
    d = []

    hh_b , shape1  = sub_divise(h_b, shape[0], shape[1])
    vv_b , shape2 =  sub_divise(v_b, shape[0], shape[1])
    dd_b , shape3 =  sub_divise(d_b, shape[0], shape[1])

    bases = const_bases.bases_directionnel()
    for hh,vv,dd in zip(hh_b,vv_b,dd_b) :

        h_origine,v_origine,d_origine = projection_inverse(hh,vv,dd,bases[b[0][k]-1],bases[b[1][k]-1],bases[b[2][k]-1])
        h.append(h_origine)
        v.append(v_origine)
        d.append(d_origine)
        k = k + 1

    return redimentioner(combiner(h,shape1),s), redimentioner(combiner(v,shape2),s),redimentioner(combiner(d,shape3),s)

"""
#verification de tout les bases
a = [[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,1]]
band = bandelette.app_produit_scalaire(a)
for (b,i) in band :
    print("****************** la base numero ", i ,"**********************")
    print(projection_inverse(b,i))

#"""