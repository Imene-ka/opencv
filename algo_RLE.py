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
        if e > 255 :
            return True
    return False

def code(sortie) :
    val = sortie
    cmp = 0
    result = []
    while true(val) :
        result = []
        for e in val:
            if e < 255:
                result.append(e)
            else:
                if e > 255:
                    result.append(255)
                    result.append(e - 255)
                else:
                    if cmp != 0 :
                        result.append(e)
                    else :
                        result.append(255)
                        result.append(0)
                        cmp = 1
        val = result

    return result

def recuperation(d) :
    result = []
    i = 0
    while i < len(d) :
        if d[i] == 255 :
            result.append(d[i]+d[i+1])
            i = i + 2
        else :
            result.append(d[i])

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
print(donnee2)"""
"""
s1 = code([255,21,650,12])
print(s1)
s2 = recuperation(s1)
print(s2)
"""







