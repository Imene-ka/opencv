import numpy as np

def val_quantif2(l,q) :
    result = []
    for e in l :
        result.append(val_quantif(e,q))
    return result
def val_quantif(b,q):
    result = np.zeros((b.shape[0], b.shape[1]))
    i = 0
    while i < b.shape[0]:
        j = 0
        while j < b.shape[1]:
            if b[i][j] == 0 :
                result[i][j] = 0
            else :
                result[i][j] = (q *(b[i][j]) ) + (q / 2)
            j = j + 1

        i = i + 1

    return result

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

            j = j +1

        i = i +1

    return result

"""
#test quntifiction
a = np.array([[320,-500,-74,-899,255],[-77,-10,10,0,20],[47,33,255,145,151],[163,178,189,19,200]])
print(a)
qa = quantif(a,20)
print(qa)
qq = val_quantif(qa,20)
print(qq)
#"""