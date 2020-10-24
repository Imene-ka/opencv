import numpy as np
import bandelette
import const_bases

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
def projection_inverse(fb,indice) :
    cb = const_bases.bases_directionnel()
    base,range = cb[indice-1]
    f = np.zeros((4, 4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            # produit
            # debut
            k = 0
            b = base[str(i) + str(j)]

            while k < 4:
                l = 0
                while l < 4:
                    b[k][l] = b[k][l] * fb[i][j]
                    l = l + 1
                k = k + 1

            f = somme(f, b)
            # fin
            j = j + 1
        i = i + 1

    return f
"""
#verification de tout les bases
a = [[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,1]]
band = bandelette.app_produit_scalaire(a)
for (b,i) in band :
    print("****************** la base numero ", i ,"**********************")
    print(projection_inverse(b,i))

#"""