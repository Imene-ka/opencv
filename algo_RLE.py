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

def true(list) :
    for e in list :
        if e >= 255 :
            return True
    return False

def code(sortie) :
    i = 0
    result = []
    while i < len(sortie):
        if sortie[i] < 255:
            result.append(sortie[i])
        else:
            if sortie[i] == 255 :
                result.append(255)
                result.append(0)
            else :
                val = sortie[i]
                while val > 255 :
                    result.append(255)
                    val = val - 255
                    if val <= 255 :
                        result.append(val)
                        if val == 255 :
                           result.append(0)
                        break
        i = i + 1

    return result

def recuperation(d) :
    result = []
    val = 0
    i = 0
    while i < len(d) :
        if d[i] == 255 :
            while d[i] == 255 :
                 val = val + d[i]
                 i = i + 1
            result.append(val+d[i])
            val = 0
        else :
            result.append(d[i])
        i = i + 1
    return result
def rle_binaire(donnee):
    sortie=[]
    if donnee[0]=="0" :
        sortie=rle(donnee,"",sortie)
    else :
        sortie.append(0)
        sortie=rle(donnee,"",sortie)

    return code(sortie)
def rle_binaire_inverse(d) :
    i=0
    pre=""
    sortie=""
    donnee = recuperation(d)
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
print(donnee2)
"""

"""
print([500,21,1050,12])
s1 = code([500,21,1050,12])
print(s1)
s2 = recuperation(s1)
print(s2)
"""






