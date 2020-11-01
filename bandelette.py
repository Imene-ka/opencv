import const_bases
import quantif_scalaire
import numpy as np
import math
import class_thread
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
            produit[i][j] = np.vdot(f,b[str(i) + str(j)])
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

def Rc(b1,b2,b3,sous_band1,sous_band2,sous_band3,shape) :
    rc1 = 0
    rc2 = 0
    rc3 = 0
    i = 0
    while i < 4 :
        j = 0
        while j < 4 :
            pr1 = (np.sum(sous_band1 == b1[i][j]))/(shape[0]*shape[1])
            rc1 = rc1 - (math.log2(pr1))
            pr2 = (np.sum(sous_band2 == b2[i][j]))/(shape[0]*shape[1])
            rc2 = rc2 - (math.log2(pr2))
            pr3 = (np.sum(sous_band3 == b3[i][j]))/(shape[0]*shape[1])
            rc3 = rc3 - (math.log2(pr3))
            j= j + 1
        i = i +1

    return rc1,rc2,rc3

def calcul_l_old_version(dis0,dis1,dis2,coefH,coefV,coefD,base,q,shape):
    L_h = []
    L_v = []
    L_d = []
    λ0 = 6.5
    # multiplicateur de Lagrange
    λ = (3 / (4 * λ0)) * math.pow(q, 2)
    rb = Rb(base)
    for i, b in enumerate(zip(coefH, coefV, coefD)):  # taille de liste
        rc1, rc2, rc3 = Rc(b[0], b[1], b[2], coefH, coefV, coefD, shape)

        L_h.append(dis0[i] + (λ * (rc1 + rb)))
        L_v.append(dis1[i] + (λ * (rc2 + rb)))
        L_d.append(dis2[i] + (λ * (rc3 + rb)))
        # L_h.append(dis0[i] + λ * (Rc(b[0],coefH,shape) + Rb(base)))
        # L_v.append(dis1[i] + λ * (Rc(b[1], coefV, shape) + Rb(base)))
        # L_d.append(dis2[i] + λ * (Rc(b[2], coefD, shape) + Rb(base)))
#input coef / output lagrange de tt ls blocks
def calcul_lagrange(dis0,dis1,dis2,coefH,coefV,coefD,base,q,shape) :
    L_h = []
    L_v = []
    L_d = []
    λ0 = 6.5
    # multiplicateur de Lagrange
    λ = (3 / (4 * λ0)) * math.pow(q, 2)
    rb = Rb(base)
    for i, b in enumerate(zip(coefH, coefV, coefD)):  # taille de liste

        rc1, rc2, rc3 = Rc(b[0], b[1], b[2], coefH, coefV, coefD, shape)

        L_h.append(dis0[i] + (λ * (rc1 + rb)))
        L_v.append(dis1[i] + (λ * (rc2 + rb)))
        L_d.append(dis2[i] + (λ * (rc3 + rb)))
        # L_h.append(dis0[i] + λ * (Rc(b[0],coefH,shape) + Rb(base)))
        # L_v.append(dis1[i] + λ * (Rc(b[1], coefV, shape) + Rb(base)))
        # L_d.append(dis2[i] + λ * (Rc(b[2], coefD, shape) + Rb(base)))

    return L_h , L_v ,L_d

def choixMB(d, h_b, v_b, d_b, q, shape) :
    i = 0
    l_h = []
    l_v =[]
    l_d = []

    """
    th1 =  class_thread.MonThread(d[0][0], d[0][1], d[0][2], h_b[0],v_b[0],d_b[0],1,q, shape)
    th1.start()
    th2 =  class_thread.MonThread(d[1][0], d[1][1], d[1][2], h_b[1], v_b[1], d_b[1], 2, q, shape)
    th2.start()
    th3 =  class_thread.MonThread(d[2][0], d[2][1], d[2][2], h_b[2], v_b[2], d_b[2], 3, q, shape)
    th3.start()
    th4 =  class_thread.MonThread(d[3][0], d[3][1], d[3][2], h_b[3], v_b[3], d_b[3], 4, q, shape)
    th4.start()
    th5 =  class_thread.MonThread(d[4][0], d[4][1], d[4][2], h_b[4], v_b[4], d_b[4], 5, q, shape)
    th5.start()
    th6 =  class_thread.MonThread(d[5][0], d[5][1], d[5][2], h_b[5], v_b[5], d_b[5], 6, q, shape)
    th6.start()
    th7 =  class_thread.MonThread(d[6][0], d[6][1], d[6][2], h_b[6], v_b[6], d_b[6], 7, q, shape)
    th7.start()
    th8 =  class_thread.MonThread(d[7][0], d[7][1], d[7][2], h_b[7], v_b[7], d_b[7], 8, q, shape)
    th8.start()
    th9 =  class_thread.MonThread(d[8][0], d[8][1], d[8][2], h_b[8], v_b[8], d_b[8], 9, q, shape)
    th9.start()
    th10 = class_thread.MonThread(d[9][0], d[9][1], d[9][2], h_b[9], v_b[9], d_b[9], 10, q, shape)
    th10.start()
    th11 = class_thread.MonThread(d[10][0], d[10][1], d[10][2], h_b[10], v_b[10], d_b[10],11, q, shape)
    th11.start()
    th12 = class_thread.MonThread(d[11][0], d[11][1], d[11][2], h_b[11], v_b[11], d_b[11], 12, q, shape)
    th12.start()

    val11, val12, val13 = th1.join()
    val21,val22,val23 = th2.join()
    val31,val32,val33 = th3.join()
    val41,val42,val43 = th4.join()
    val51,val52,val53 = th5.join()
    val61,val62,val63 = th6.join()
    val71,val72,val73 = th7.join()
    val81,val82,val83 = th8.join()
    val91,val92,val93 = th9.join()
    val101,val102,val103 = th10.join()
    val111,val112,val113 = th11.join()
    val121,val122,val123 = th12.join()

    l_h = [val11,val21,val31,val41,val51,val61,val71,val81,val91,val101,val111,val121]
    l_v = [val12,val22,val32,val42,val52,val62,val72,val82,val92,val102,val112,val122]
    l_d = [val13,val23,val33,val43,val53,val63,val73,val83,val93,val103,val113,val123]
    """
    for h,v,dd in zip(h_b,v_b,d_b) : # 12 iteration
        dis = d[i]
        val1,val2,val3 = calcul_lagrange(dis[0],dis[1],dis[2],h,v,dd,i+1,q,shape)
        l_h.append(val1)
        l_v.append(val2)
        l_d.append(val3)
        i = i + 1
        #l_v.append(calcul_lagrange(dis[1],v,i,q,shape))
        #l_d.append(calcul_lagrange(dis[2],dd,i,q,shape))
        print("calcule lagrange",i)


    l = [h_b,v_b,d_b]
    r = []

    for k,e in enumerate([l_h,l_v,l_d]) : # nbr des block
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
            list_L_blocks = [it[0],it[1],it[2],it[3],it[4],it[5],it[6],it[7],it[8],it[9],it[10],it[11]]
            index_min = list_L_blocks.index(min(list_L_blocks))
            # l'indice i represente l'indice de block
            # l'indice de min represente la base
            bl = l[k]
            result.append(bl[index_min][i])

        r.append((combiner(result,shape)))

    return r[0],r[1],r[2]

# input coeffs / output coeffs -> transformation bandelette - > quantification
def coef_base(h,v,d,b,q) :
    eb_h = []
    eb_v = []
    eb_d = []
    for e1,e2,e3 in zip(h,v,d):
        eb_h.append(quantif_scalaire.quantif(fct(e1,b),q))
        eb_v.append(quantif_scalaire.quantif(fct(e2,b),q))
        eb_d.append(quantif_scalaire.quantif(fct(e3,b),q))
    return eb_h , eb_v , eb_d

# input coef originale et quantifier -> list de distorsion pour chaque block
def calcule_distorsion(bhs,bvs,bds,q_h,q_v,q_d) :
    d_h = []
    d_v =[]
    d_d = []
    for bh,bv,bd,q1,q2,q3 in zip(bhs,bvs,bds,q_h,q_v,q_d) : # nombre de blocks iteration
        d_h.append(distorsion(bh, q1))
        d_v.append(distorsion(bv, q2))
        d_d.append(distorsion(bd, q3))

    return d_h , d_v , d_d

#input les coefs de ondelette -> coefs transformées et les bases associé
def bandelette(h,v,dd,q) :

    base_dire = const_bases.bases_directionnel()
    # h1 , h2 = const_bases.base_H()
    # dct = const_bases.base_DCT()

    # subdivise les blocks
    blocks_h, shape_h = sub_divise(h, h.shape[0], h.shape[1])
    blocks_v, shape_v = sub_divise(v, v.shape[0], v.shape[1])
    blocks_d, shape_d = sub_divise(dd, dd.shape[0], dd.shape[1])

    # declaration
    h_b = []
    v_b = []
    d_b = []
    d = []

    # les block dans les bases
    for b,i in base_dire : # 12 iteration
        b1 , b2 , b3 = coef_base(blocks_h,blocks_v,blocks_d,b,q)
        h_b.append(b1)
        v_b.append(b2)
        d_b.append(b3)

        d.append(calcule_distorsion(blocks_h,blocks_v,blocks_d,quantif_scalaire.val_quantif2(b1,q),quantif_scalaire.val_quantif2(b2,q),
                                    quantif_scalaire.val_quantif2(b3,q)))

    #ajouter les instruction pour les bases complementaires
    print("finsh distortion")
    #calcule de lagrange
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



