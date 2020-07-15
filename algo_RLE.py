import numpy as np

def rle(donnee,pre,sortie) :
    cmp=0
    for n in donnee :
        if n=="1" :
           if pre=="1" :
               cmp = cmp+1
               pre="1"
           else :
               if pre !="":
                   sortie.append(cmp)
               cmp = 1
               pre = "1"
        if n=="0" :
           if pre=="0" :
               cmp = cmp+1
               pre="0"
           else :
               if pre !="":
                   sortie.append(cmp)
               cmp=1
               pre = "0"
    sortie.append(cmp)

    return sortie
def rle_binaire(donnee):
    sortie=[]
    if donnee[0]=="0" :
        sortie=rle(donnee,"",sortie)
    else :
        sortie.append(0)
        sortie=rle(donnee,"",sortie)
    return sortie

def rle_binaire_inverse(donnee) :
    i=0
    pre=""
    sortie=str()
    while i<len(donnee) :
        if pre=="" or pre=="0" :
             sortie = sortie + ("0" * donnee[i])
             pre="1"
        else :
            sortie = sortie + ("1" * donnee[i])
            pre="0"
        i=i+1
    return sortie
"""#codage
sortie1=rle_binaire("10001100010")
print(sortie1)
sortie2=rle_binaire("000110000010")
print(sortie2)
#decodage
donnee1=rle_binaire_inverse(sortie1)
print(donnee1)
donnee2=rle_binaire_inverse(sortie2)
print(donnee2)"""










