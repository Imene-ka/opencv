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

def produit_scalaire(f, b):
    h = 0
    sum = 0
    while h < 4:
        k = 0
        while k < 4:
            sum = sum + (f[h][k] * b[h][k])
            k = k + 1

        h = h + 1
    return sum

def fct(f,b) :
    produit = np.zeros((4, 4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            produit[i][j] = produit_scalaire(f,b[str(i) + str(j)])
            j = j + 1
        i = i + 1

    return  produit

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

def proba(val,sous_band,shape) :
    p = np.sum(sous_band == val )
    return p/(shape[0]*shape[1])

def Rc(b,sous_band,shape) :
    rc = 0
    i = 0
    while i < 4 :
        j = 0
        while j < 4 :
            pr = proba(b[i][j],sous_band,shape)
            rc = rc - (math.log2(pr))
            j= j + 1
        i = i +1

    return rc

#input coef / output lagrange de tt ls blocks
def calcul_lagrange(dis0,dis1,dis2,coefH,coefV,coefD,base,q,shape) :
    L_h = []
    L_v = []
    L_d = []
    λ0 = 6.5
    # multiplicateur de Lagrange
    λ = (3 / (4 * λ0)) * math.pow(q, 2)
    for i,b in enumerate(zip(coefH,coefV,coefD)) :
        L_h.append(dis0[i] + λ * (Rc(b[0],coefH,shape) + Rb(base)))
        L_v.append(dis1[i] + λ * (Rc(b[1], coefV, shape) + Rb(base)))
        L_d.append(dis2[i] + λ * (Rc(b[2], coefD, shape) + Rb(base)))


    return L_h , L_v ,L_d

def choixMB(d, h_b, v_b, d_b, q, shape) :
    i = 0
    l_h = []
    l_v = []
    l_d = []

    for h,v,dd in zip(h_b,v_b,d_b) :
        dis = d[i]
        val1,val2,val3 = calcul_lagrange(dis[0],dis[1],dis[2],h,v,dd,i,q,shape)
        l_h.append(val1)
        l_v.append(val2)
        l_d.append(val3)
        #l_v.append(calcul_lagrange(dis[1],v,i,q,shape))
        #l_d.append(calcul_lagrange(dis[2],dd,i,q,shape))
        print("calcule lagrange")
        i = i + 1

    l = [h_b,v_b,d_b]
    r = []
    for k,e in enumerate([l_h,l_v,l_d]) :
        result = []
        e1 = e[0]
        e2 = e[1]
        e3 = e[2]
        e4 = e[3]
        e5 = e[4]
        e6 = e[5]
        e7 = e[6]
        e8 = e[7]
        e9 = e[8]
        e10 = e[9]
        e11 = e[10]
        e12 = e[11]
        for i , it in enumerate(zip(e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12)) :
            list_L_blocks = [it[0],it[1],it[2],it[3],it[4],it[5],it[6],it[7],it[8],it[9],it[10],it[11],it[12]]
            index_min = list_L_blocks.index(min(list_L_blocks))
            # l'indice i represente l'indice de block
            # l'indice de min represente la base
            bl = l[k]
            result.append(bl[i])

        r.append((combiner(result,shape)))

    return r[0],r[1],r[2]

# input coef / output coef -> transformation bandelette - > quantification
def coef_base(h,v,d,b,q) :
    eb_h = []
    eb_v = []
    eb_d = []
    for e1,e2,e3 in zip(h,v,d):
        eb_h.append(quantif_scalaire.quantif(fct(e1,b),q))
        eb_v.append(quantif_scalaire.quantif(fct(e2, b), q))
        eb_d.append(quantif_scalaire.quantif(fct(e3, b), q))
    return eb_h , eb_v , eb_d

# input coef originale et quantifier -> list de distorsion pour chaque block
def calcule_distorsion(bhs,bvs,bds,q_h,q_v,q_d) :
    d_h = []
    d_v =[]
    d_d = []
    for bh,bv,bd,q1,q2,q3 in zip(bhs,bvs,bds,q_h,q_v,q_d) :
        d_h.append(distorsion(bh, q1))
        d_v.append(distorsion(bv, q2))
        d_d.append(distorsion(bd, q3))

    return d_h , d_v , d_d

#input les coefs de ondelette -> coefs transformées et les bases associé
def bandelette(h,v,dd,q) :
    base_dire = const_bases.bases_directionnel()
    # subdivise les blocks
    blocks_h, shape_h = sub_divise(h, h.shape[0], h.shape[1])
    blocks_v, shape_v = sub_divise(v, v.shape[0], v.shape[1])
    blocks_d, shape_d = sub_divise(dd, dd.shape[0], dd.shape[1])

    #h1 , h2 = const_bases.base_H()
    #dct = const_bases.base_DCT()
    h_b = []
    v_b = []
    d_b = []
    d = []
    for b,i in base_dire :
        b1 , b2 , b3 = coef_base(blocks_h,blocks_v,blocks_d,b,q)
        h_b.append(b1)
        v_b.append(b2)
        d_b.append(b3)

        d.append(calcule_distorsion(blocks_h,blocks_v,blocks_d,quantif_scalaire.val_quantif2(b1,q),quantif_scalaire.val_quantif2(b2,q),
                                    quantif_scalaire.val_quantif2(b3,q)))

    #ajouter les instruction pour les bases complementaires
    print("finsh distortion")
    h_b , v_b , d_b = choixMB(d,h_b,v_b,d_b,q,shape_h)

    return h_b,v_b,d_b


#test bandelette
#a = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15],[16,17,18,19,20]])

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



