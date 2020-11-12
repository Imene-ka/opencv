import numpy as np

def val_quantif2(l,q) :
    result = []
    for e in l :
        result.append(val_quantif(e,q))
    return result
def val_quantif(b,q):
    result = np.zeros(b.shape)
    i = 0
    while i < b.shape[0]:
        j = 0
        while j < b.shape[1]:
            if b[i][j] == 0 :
                result[i][j] = 0
            else :
                result[i][j] = (b[i][j] + 0.5 ) * q
            j += 1

        i += 1

    return result

def quantif1(b1,b2,b3,q) :
    result1 = np.zeros((4,4))
    result2 = np.zeros((4,4))
    result3 = np.zeros((4,4))
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            if b1[i][j] > -q and b1[i][j] < q:
                result1[i][j] = 0
            else:
                result1[i][j] = int(b1[i][j] / q)

            if b2[i][j] > -q and b2[i][j] < q:
                result2[i][j] = 0
            else:
                result2[i][j] = int(b2[i][j] / q)

            if b3[i][j] > -q and b3[i][j] < q:
                result3[i][j] = 0
            else:
                result3[i][j] = int(b3[i][j] / q)

            j = j + 1

        i = i + 1

    return result1,result2,result3
def quantif(b,q) :
    result = np.zeros((b.shape[0],b.shape[1]))
    i = 0
    while i< b.shape[0] :
        j = 0
        while j < b.shape[1] :
            if b[i][j] > -q and b[i][j] < q :
                result[i][j] = 0
            else :
                result[i][j] = int(b[i][j] / q)

            j += 1

        i += 1

    return result

"""
#test quntifiction
a = np.array([[320,-500.30,-74,-899.4,255],[-77,-10,10,0,20],[47,33,255,145,151],[163,178,189,19,200]])
print(a)
qa = quantif(a,20)
print(qa)
qq = val_quantif(qa,20)
print(qq)
#"""