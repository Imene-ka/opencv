import math
import matplotlib.pyplot as plt
import numpy as np
import groupe



def position():
    pos = []
    i = 0
    while i < 4:
        j = 0
        while j < 4:
            pos.append((i, j))
            j = j + 1
        i = i + 1

    return pos


def directions():
    pos = position()

    v0 = [1]

    v1 = [1 / math.sqrt(2), -1 / math.sqrt(2)]
    v2 = [1 / math.sqrt(2), 1 / math.sqrt(2)]

    v3 = [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)]
    v4 = [-1 / math.sqrt(2), 0, -1 / math.sqrt(2)]
    v5 = [-1 / math.sqrt(6), 2 / math.sqrt(6), -1 / math.sqrt(6)]

    v6 = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
    v7 = [-1 / 2, 1 / 2, 1 / 2, -1 / 2]
    v8 = [(math.sqrt(2) - 2) / (2 * math.sqrt(2)), math.sqrt((2 * math.sqrt(2)) - 1) / 2,
          -math.sqrt((2 * math.sqrt(2)) - 1) / 2, (2 - math.sqrt(2)) / (2 * math.sqrt(2))]
    v9 = [-math.sqrt((2 * math.sqrt(2)) - 1) / 2, (math.sqrt(2) - 2) / (2 * math.sqrt(2)),
          (2 - math.sqrt(2)) / (2 * math.sqrt(2)), math.sqrt((2 * math.sqrt(2)) - 1) / 2]

    d1 = [groupe.groupe(3, [v3, v4, v5], [pos[0], pos[1], pos[2]]), groupe.groupe(2, [v1, v2], [pos[3], pos[4]]),
          groupe.groupe(2, [v1, v2], [pos[5], pos[6]]), groupe.groupe(2, [v1, v2], [pos[7], pos[8]]),
          groupe.groupe(2, [v1, v2], [pos[9], pos[10]]), groupe.groupe(2, [v1, v2], [pos[11], pos[12]]),
          groupe.groupe(3, [v3, v4, v5], [pos[13], pos[14], pos[15]])]

    d2 = [groupe.groupe(2, [v1, v2], [pos[0], pos[1]]), groupe.groupe(2, [v1, v2], [pos[2], pos[4]]),
          groupe.groupe(2, [v1, v2], [pos[3], pos[5]]), groupe.groupe(2, [v1, v2], [pos[6], pos[8]]),
          groupe.groupe(2, [v1, v2], [pos[7], pos[9]]), groupe.groupe(2, [v1, v2], [pos[10], pos[12]]),
          groupe.groupe(2, [v1, v2], [pos[11], pos[13]]), groupe.groupe(2, [v1, v2], [pos[14], pos[15]])]

    d3 = [groupe.groupe(1, [v0], [pos[0]]), groupe.groupe(2, [v1, v2], [pos[1], pos[4]]),
          groupe.groupe(3, [v3, v4, v5], [pos[2], pos[5], pos[8]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[3], pos[6], pos[9], pos[12]]),
          groupe.groupe(3, [v3, v4, v5], [pos[7], pos[10], pos[13]]),
          groupe.groupe(2, [v1, v2], [pos[11], pos[14]]), groupe.groupe(1, [v0], [pos[15]])]

    d4 = [groupe.groupe(2, [v1, v2], [pos[0], pos[4]]), groupe.groupe(2, [v1, v2], [pos[1], pos[8]]),
          groupe.groupe(2, [v1, v2], [pos[2], pos[9]]), groupe.groupe(2, [v1, v2], [pos[3], pos[10]]),
          groupe.groupe(2, [v1, v2], [pos[5], pos[12]]), groupe.groupe(2, [v1, v2], [pos[6], pos[13]]),
          groupe.groupe(2, [v1, v2], [pos[7], pos[14]]), groupe.groupe(2, [v1, v2], [pos[11], pos[15]])]

    d5 = [groupe.groupe(3, [v3, v4, v5], [pos[0], pos[4], pos[8]]), groupe.groupe(2, [v1, v2], [pos[1], pos[12]]),
          groupe.groupe(2, [v1, v2], [pos[5], pos[9]]), groupe.groupe(2, [v1, v2], [pos[2], pos[13]]),
          groupe.groupe(2, [v1, v2], [pos[6], pos[10]]), groupe.groupe(2, [v1, v2], [pos[3], pos[14]]),
          groupe.groupe(3, [v3, v4, v5], [pos[7], pos[11], pos[15]])]

    d6 = [groupe.groupe(4, [v6, v7, v8, v9], [pos[0], pos[4], pos[8], pos[12]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[1], pos[5], pos[9], pos[13]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[2], pos[6], pos[10], pos[14]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[3], pos[7], pos[11], pos[15]])]

    d7 = [groupe.groupe(2, [v1, v2], [pos[0], pos[13]]), groupe.groupe(2, [v1, v2], [pos[1], pos[14]]),
          groupe.groupe(2, [v1, v2], [pos[2], pos[15]]), groupe.groupe(3, [v3, v4, v5], [pos[3], pos[7], pos[11]]),
          groupe.groupe(3, [v3, v4, v5], [pos[4], pos[8], pos[12]]), groupe.groupe(2, [v1, v2], [pos[5], pos[9]]),
          groupe.groupe(2, [v1, v2], [pos[6], pos[10]])]

    d8 = [groupe.groupe(2, [v1, v2], [pos[0], pos[9]]), groupe.groupe(2, [v1, v2], [pos[1], pos[10]]),
          groupe.groupe(2, [v1, v2], [pos[2], pos[11]]), groupe.groupe(2, [v1, v2], [pos[3], pos[7]]),
          groupe.groupe(2, [v1, v2], [pos[4], pos[13]]), groupe.groupe(2, [v1, v2], [pos[5], pos[14]]),
          groupe.groupe(2, [v1, v2], [pos[6], pos[15]]), groupe.groupe(2, [v1, v2], [pos[8], pos[12]])]

    d9 = [groupe.groupe(4, [v6, v7, v8, v9], [pos[0], pos[5], pos[10], pos[15]]),
          groupe.groupe(3, [v3, v4, v5], [pos[1], pos[6], pos[11]]), groupe.groupe(2, [v1, v2], [pos[2], pos[7]]),
          groupe.groupe(1, [v0], [pos[3]]),
          groupe.groupe(3, [v3, v4, v5], [pos[4], pos[9], pos[14]]), groupe.groupe(2, [v1, v2], [pos[8], pos[13]]),
          groupe.groupe(1, [v0], [pos[12]])]

    d10 = [groupe.groupe(2, [v1, v2], [pos[0], pos[6]]), groupe.groupe(2, [v1, v2], [pos[1], pos[7]]),
           groupe.groupe(2, [v1, v2], [pos[2], pos[3]]), groupe.groupe(2, [v1, v2], [pos[4], pos[10]]),
           groupe.groupe(2, [v1, v2], [pos[5], pos[11]]), groupe.groupe(2, [v1, v2], [pos[8], pos[14]]),
           groupe.groupe(2, [v1, v2], [pos[9], pos[15]]), groupe.groupe(2, [v1, v2], [pos[12], pos[13]])]

    d11 = [groupe.groupe(2, [v1, v2], [pos[0], pos[7]]), groupe.groupe(3, [v3, v4, v5], [pos[1], pos[2], pos[3]]),
           groupe.groupe(2, [v1, v2], [pos[4], pos[11]]), groupe.groupe(2, [v1, v2], [pos[5], pos[6]]),
           groupe.groupe(2, [v1, v2], [pos[8], pos[15]]), groupe.groupe(2, [v1, v2], [pos[9], pos[10]]),
           groupe.groupe(3, [v3, v4, v5], [pos[12], pos[13], pos[14]])]

    d12 = [groupe.groupe(4, [v6, v7, v8, v9], [pos[0], pos[1], pos[2], pos[3]]),
           groupe.groupe(4, [v6, v7, v8, v9], [pos[4], pos[5], pos[6], pos[7]]),
           groupe.groupe(4, [v6, v7, v8, v9], [pos[8], pos[9], pos[10], pos[11]]),
           groupe.groupe(4, [v6, v7, v8, v9], [pos[12], pos[13], pos[14], pos[15]])]

    return d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12


def create_base(d):
    base = {}
    for g in d:
        #gi = g.gi
        pos = g.pos
        vecteur = g.vecteur

        for s, p in enumerate(pos):
            b = np.zeros((4, 4))
            (m, n) = p
            v = vecteur[s]
            for k, pp in enumerate(pos):
                (i, j) = pp
                b[i][j] = v[k]

            base[str(m)+str(n)] = b

    return base


def bases_directionnel():
    base = []
    ldirection = directions()

    for i,d in enumerate(ldirection):
        base.append((create_base(d),i+1))
    return base


def base_DCT():

    dct00 = [[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5],[0.5,0.5,0.5,0.5]]
    dct01 = [[1,1,1,1],[0.5,0.5,0.5,0.5],[-0.5,-0.5,-0.5,-0.5],[-1,-1,-1,-1]]
    dct02 = [[0.5,0.5,0.5,0.5],[-0.5,-0.5,-0.5,-0.5],[-0.5,-0.5,-0.5,-0.5],[0.5,0.5,0.5,0.5]]
    dct03 = [[0.2,0.2,0.2,0.2],[-1,-1,-1,-1],[1,1,1,1],[-0.2,-0.2,-0.2,-0.2]]
    dct10 = np.transpose(dct01)
    dct11 = [[1, 0.5 ,-0.5 ,-1],[0.5,0,-0.2,-0.5],[-0.5,-0.2,0,0.5],[-1,-0.5,0.5,1]]
    dct12 = [[1,0.5,-0.5,-1],[-1,-0.5,0.5,1],[-1,-0.5,0.5,1],[1,0.5,-0.5,-1]]
    dct13 = [[0.5,0,-0.2,-0.5],[-1,-0.5,0.5,1],[1,0.5,-0.5,-1],[-0.5,-0.2,0,0.5]]
    dct20 = np.transpose(dct02)
    dct21 = np.transpose(dct12)
    dct22 = [[0.5,-0.5,-0.5,0.5],[-0.5,0.5,0.5,-0.5],[-0.5,0.5,0.5,-0.5],[0.5,-0.5,-0.5,0.5]]
    dct23 = [[0,-0.5 ,-0.5,0],[-0.7,1,1,-0.7],[1,-0.7,-0.7,1],[-0.5,0,0,-0.5]]
    dct30 = np.transpose(dct03)
    dct31 = np.transpose(dct13)
    dct32 = np.transpose(dct23)
    dct33 = [[0,-0.5,0,-0.5],[-0.5,1,-1,0],[0,-1,1,-0.5],[-0.5,0,-0.5,0]]

    dct = {'00': dct00 , '01' : dct01 , '02' : dct02 , '03' : dct03 , '10' : dct10 , '11' : dct11 , '12' : dct12 ,
           '13' : dct13 , '20' : dct20 , '21' : dct21 , '22' : dct22 , '23' : dct23 , '30' : dct30 , '32' : dct32 ,
           '31' : dct31 , '33' : dct33}

    return dct,13


def base_H():
    H0 = {
        '00': [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
        '01': [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.75, 0.75, 0.75, 0.75], [0.75, 0.75, 0.75, 0.75]],
        '02': [[1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '03': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5]],
        '10': [[0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75]],
        '11': [[0.25, 0.25, 0.75, 0.75], [0.25, 0.25, 0.75, 0.75], [0.75, 0.75, 0.25, 0.25], [0.75, 0.75, 0.25, 0.25]],
        '12': [[0.5, 0.5, 1, 1], [0.5, 0.5, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '13': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 1], [0.5, 0.5, 0, 0]],
        '20': [[1, 0, 0.5, 0.5], [1, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '21': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 0, 0.5, 0.5], [1, 0, 0.5, 0.5]],
        '22': [[1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '23': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5]],
        '30': [[0.5, 0.5, 1, 0], [0.5, 0.5, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '31': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 0], [0.5, 0.5, 1, 0]],
        '32': [[0.5, 0.5, 1, 0], [0.5, 0.5, 0, 1], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '33': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 0], [0.5, 0.5, 0, 1]]
    }
    H1 = {
        '00': [[1, 1, 0.5, 0.5], [1, 1, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '01': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 1, 0.5, 0.5], [1, 1, 0.5, 0.5]],
        '02': [[1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '03': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 1, 0.5, 0.5], [0, 0, 0.5, 0.5]],
        '10': [[0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '11': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 1], [0.5, 0.5, 1, 1]],
        '12': [[0.5, 0.5, 1, 1], [0.5, 0.5, 0, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '13': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 1], [0.5, 0.5, 0, 0]],
        '20': [[1, 0, 0.5, 0.5], [1, 0, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '21': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 0, 0.5, 0.5], [1, 0, 0.5, 0.5]],
        '22': [[1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '23': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [1, 0, 0.5, 0.5], [0, 1, 0.5, 0.5]],
        '30': [[0.5, 0.5, 1, 0], [0.5, 0.5, 1, 0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '31': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 0], [0.5, 0.5, 1, 0]],
        '32': [[0.5, 0.5, 1, 0], [0.5, 0.5, 0, 1], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],
        '33': [[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1, 0], [0.5, 0.5, 0, 1]]
    }
    H0 = (H0,14)
    H1 = (H1,15)
    return H0, H1

#verification de subdivision
"""
a = np.array([[1, 2, 3, 4, 1, 2], [5, 6, 7, 8, 5, 6], [9, 10, 11, 12, 9, 10],
              [13, 14, 15, 16, 13, 14], [1, 2, 3, 4, 1, 2], [5, 6, 7, 8, 5, 6],
              [9, 10, 11, 12, 9, 10], [13, 14, 15, 16, 13, 14]])
f = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
# print(a)
result = sub_divise(a, 8, 6)
print(result)

"""

"""
v0 = [1]

v1 = [1 / math.sqrt(2), 1 / math.sqrt(2)]
v2 = [1 / math.sqrt(2), -1 / math.sqrt(2)]

v3 = [1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)]
v4 = [-1 / math.sqrt(2), 0, 1 / math.sqrt(2)]
v5 = [-1 / math.sqrt(6), 2 / math.sqrt(6), -1 / math.sqrt(6)]


v6 = [1 / 2, 1 / 2, 1 / 2, 1 / 2]
v7 = [-1 / 2, 1 / 2, 1 / 2, -1 / 2]
v8 = [(math.sqrt(2) - 2) / (2 * math.sqrt(2)), math.sqrt((2 * math.sqrt(2)) - 1) / 2,
          -math.sqrt((2 * math.sqrt(2)) - 1) / 2, (2 - math.sqrt(2)) / (2 * math.sqrt(2))]
v9 = [-math.sqrt((2 * math.sqrt(2)) - 1) / 2, (math.sqrt(2) - 2) / (2 * math.sqrt(2)),
          (2 - math.sqrt(2)) / (2 * math.sqrt(2)), math.sqrt((2 * math.sqrt(2)) - 1) / 2]

pos = position()

d1 = [groupe.groupe(3, [v3, v4, v5], [pos[0], pos[1], pos[2]]), groupe.groupe(2, [v1, v2], [pos[3], pos[4]]),
          groupe.groupe(2, [v1, v2], [pos[5], pos[6]]), groupe.groupe(2, [v1, v2], [pos[7], pos[8]]),
          groupe.groupe(2, [v1, v2], [pos[9], pos[10]]), groupe.groupe(2, [v1, v2], [pos[11], pos[12]]),
          groupe.groupe(3, [v3, v4, v5], [pos[13], pos[14], pos[15]])]

d2 = [groupe.groupe(2, [v1, v2], [pos[0], pos[1]]), groupe.groupe(2, [v1, v2], [pos[2], pos[4]]),
          groupe.groupe(2, [v1, v2], [pos[3], pos[5]]), groupe.groupe(2, [v1, v2], [pos[6], pos[8]]),
          groupe.groupe(2, [v1, v2], [pos[7], pos[9]]), groupe.groupe(2, [v1, v2], [pos[10], pos[12]]),
          groupe.groupe(2, [v1, v2], [pos[11], pos[13]]), groupe.groupe(2, [v1, v2], [pos[14], pos[15]])]

d3 = [groupe.groupe(1, [v0], [pos[0]]), groupe.groupe(2, [v1, v2], [pos[1], pos[4]]),
          groupe.groupe(3, [v3, v4, v5], [pos[2], pos[5], pos[8]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[3], pos[6], pos[9], pos[12]]),
          groupe.groupe(3, [v3, v4, v5], [pos[7], pos[10], pos[13]]),
          groupe.groupe(2, [v1, v2], [pos[11], pos[14]]), groupe.groupe(1, [v0], [pos[15]])]
d4 = [groupe.groupe(2, [v1, v2], [pos[0], pos[4]]), groupe.groupe(2, [v1, v2], [pos[1], pos[8]]),
          groupe.groupe(2, [v1, v2], [pos[2], pos[9]]), groupe.groupe(2, [v1, v2], [pos[3], pos[10]]),
          groupe.groupe(2, [v1, v2], [pos[5], pos[12]]), groupe.groupe(2, [v1, v2], [pos[6], pos[13]]),
          groupe.groupe(2, [v1, v2], [pos[7], pos[14]]), groupe.groupe(2, [v1, v2], [pos[11], pos[15]])]

d5 = [groupe.groupe(3, [v3, v4, v5], [pos[0], pos[4], pos[8]]), groupe.groupe(2, [v1, v2], [pos[1], pos[12]]),
          groupe.groupe(2, [v1, v2], [pos[5], pos[9]]), groupe.groupe(2, [v1, v2], [pos[2], pos[13]]),
          groupe.groupe(2, [v1, v2], [pos[6], pos[10]]), groupe.groupe(2, [v1, v2], [pos[3], pos[14]]),
          groupe.groupe(3, [v3, v4, v5], [pos[7], pos[11], pos[15]])]

d6 = [groupe.groupe(4, [v6, v7, v8, v9], [pos[0], pos[4], pos[8], pos[12]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[1], pos[5], pos[9], pos[13]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[2], pos[6], pos[10], pos[14]]),
          groupe.groupe(4, [v6, v7, v8, v9], [pos[3], pos[7], pos[11], pos[15]])]

d7 = [groupe.groupe(2, [v1, v2], [pos[0], pos[13]]), groupe.groupe(2, [v1, v2], [pos[1], pos[14]]),
          groupe.groupe(2, [v1, v2], [pos[2], pos[15]]), groupe.groupe(3, [v3, v4, v5], [pos[3], pos[7], pos[11]]),
          groupe.groupe(3, [v3, v4, v5], [pos[4], pos[8], pos[12]]), groupe.groupe(2, [v1, v2], [pos[5], pos[9]]),
          groupe.groupe(2, [v1, v2], [pos[6], pos[10]])]

d8 = [groupe.groupe(2, [v1, v2], [pos[0], pos[9]]), groupe.groupe(2, [v1, v2], [pos[1], pos[10]]),
          groupe.groupe(2, [v1, v2], [pos[2], pos[11]]), groupe.groupe(2, [v1, v2], [pos[3], pos[7]]),
          groupe.groupe(2, [v1, v2], [pos[4], pos[13]]), groupe.groupe(2, [v1, v2], [pos[5], pos[14]]),
          groupe.groupe(2, [v1, v2], [pos[6], pos[15]]), groupe.groupe(2, [v1, v2], [pos[8], pos[12]])]

d9 = [groupe.groupe(4, [v6, v7, v8, v9], [pos[0], pos[5], pos[10], pos[15]]),
          groupe.groupe(3, [v3, v4, v5], [pos[1], pos[6], pos[11]]), groupe.groupe(2, [v1, v2], [pos[2], pos[7]]),
          groupe.groupe(1, [v0], [pos[3]]),
          groupe.groupe(3, [v3, v4, v5], [pos[4], pos[9], pos[14]]), groupe.groupe(2, [v1, v2], [pos[8], pos[13]]),
          groupe.groupe(1, [v0], [pos[12]])]

d10 = [groupe.groupe(2, [v1, v2], [pos[0], pos[6]]), groupe.groupe(2, [v1, v2], [pos[1], pos[7]]),
           groupe.groupe(2, [v1, v2], [pos[2], pos[3]]), groupe.groupe(2, [v1, v2], [pos[4], pos[10]]),
           groupe.groupe(2, [v1, v2], [pos[5], pos[11]]), groupe.groupe(2, [v1, v2], [pos[8], pos[14]]),
           groupe.groupe(2, [v1, v2], [pos[9], pos[15]]), groupe.groupe(2, [v1, v2], [pos[12], pos[13]])]

d11 = [groupe.groupe(2, [v1, v2], [pos[0], pos[7]]), groupe.groupe(3, [v3, v4, v5], [pos[1], pos[2], pos[3]]),
           groupe.groupe(2, [v1, v2], [pos[4], pos[11]]), groupe.groupe(2, [v1, v2], [pos[5], pos[6]]),
           groupe.groupe(2, [v1, v2], [pos[8], pos[15]]), groupe.groupe(2, [v1, v2], [pos[9], pos[10]]),
           groupe.groupe(3, [v3, v4, v5], [pos[12], pos[13], pos[14]])]

d12 = [groupe.groupe(4, [v6, v7, v8, v9], [pos[0], pos[1], pos[2], pos[3]]),
           groupe.groupe(4, [v6, v7, v8, v9], [pos[4], pos[5], pos[6], pos[7]]),
           groupe.groupe(4, [v6, v7, v8, v9], [pos[8], pos[9], pos[10], pos[11]]),
           groupe.groupe(4, [v6, v7, v8, v9], [pos[12], pos[13], pos[14], pos[15]])]
# creation d'une base
base = create_base(d12)

"""

"""print(base["00"])
print(base["01"])
print(base["02"])
print(base["03"])
print(base["10"])
print(base["11"])
print(base["12"])
print(base["13"])
print(base["20"])
print(base["21"])
print(base["22"])
print(base["23"])
print(base["30"])
print(base["31"])
print(base["32"])
print(base["33"])"""

i=0
cmp = 1
fig =  plt.figure(figsize=(20, 20))
mat = np.zeros((16, 16))
"""while i < 4 :
    j = 0
    while j < 4 :
       b= base[str(i)+str(j)]
       #print(b * 128 + 127)
       k = 0
       while k < 4 :
           l = 0
           while l < 4 :
               b[k][l] = b[k][l]*128 + 127
               if b[k][l] == -1 :
                   b[k][l] = 0
               l = l + 1
           k = k + 1
       ax = fig.add_subplot(4, 4,cmp)
       ax.imshow(b,cmap=plt.cm.gray)
       ax.set_xticks([])
       ax.set_yticks([])
       cmp = cmp + 1

       j = j +1
    i = i + 1 
plt.show()"""


# test projection et projection inverse
"""
f = [[1,1,1,1],[1,1,1,1],[0,0,0,0],[1,1,1,1]]
print("******************************  le block 4*4 ********************************** \n")
print(f)
print("\n")
fb = fct(f,base)
print("******************************  l'ecreteur en base b ********************************** \n")
print(fb)
print("\n")
result = projection_inverse(fb,base)
print("******************************  resultat ********************************** \n")
print(result)
"""
"""
#test des bases complementaire
f = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]

h0,h1 = base_H()
(base,i) = h0


fb = fct(f,base)
result = projection_inverse(fb,base)
print("******************************  resultat ********************************** \n")
print(result)

"""
"""
#test base dct
f = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
dct,i = base_DCT()
fb = fct(f,dct)
print(fb)
print(dct["00"])
print(dct["01"])
print(dct["02"])
print(dct["03"])
print(dct["10"])
print(dct["11"])
print(dct["12"])
print(dct["13"])
print(dct["20"])
print(dct["21"])
print(dct["22"])
print(dct["23"])
print(dct["30"])
print(dct["31"])
print(dct["32"])
print(dct["33"])
result = projection_inverse(fb,dct)
print("******************************  resultat ********************************** \n")
#print(result)
while i < 4 :
    j = 0
    while j < 4 :
       b= dct[str(i)+str(j)]
       #print(b * 128 + 127)
       k = 0
       while k < 4 :
           l = 0
           while l < 4 :
               b[k][l] = (b[k][l]+1) * 127
               l = l + 1
           k = k + 1
           
       ax = fig.add_subplot(4, 4,cmp)
       ax.imshow(b,cmap=plt.cm.gray)
       ax.set_xticks([])
       ax.set_yticks([])
       cmp = cmp + 1

       j = j +1
    i = i + 1
plt.show()
"""

""""#test de quantification
f = [[19,200,42,66],[87,54,69,77],[45,78,68,60],[158,180,0,20]]
fq = quantif(f,20)
print(fq)
d = distorsion(f,fq)
print(d)
"""
