import threading
import numpy as np
import math

class MonThread (threading.Thread):

    def __init__(self,dis0,dis1,dis2,coefH,coefV,coefD,base,q,shape):
        threading.Thread.__init__(self)  # ne pas oublier cette ligne
        # (appel au constructeur de la classe mère)
        self.dis0 = dis0
        self.dis1 = dis1
        self.dis2 = dis2
        self.coefH = coefH
        self.coefV = coefV
        self.coefD = coefD
        self.base = base
        self.q = q
        self.shape = shape
        self._return = None # donnée supplémentaire ajoutée à la classe

    def run(self):
        L_h = []
        L_v = []
        L_d = []
        λ0 = 6.5
        # multiplicateur de Lagrange
        λ = (3 / (4 * λ0)) * math.pow(self.q, 2)
        for i, b in enumerate(zip(self.coefH, self.coefV, self.coefD)):# taille de liste
            rc1, rc2, rc3 = self.Rc(b[0], b[1], b[2], self.coefH, self.coefV, self.coefD, self.shape)
            rb = self.Rb(self.base)
            L_h.append(self.dis0[i] + (λ * (rc1 + rb)))

            L_v.append(self.dis1[i] + (λ * (rc2 + rb)))

            L_d.append(self.dis2[i] + (λ * (rc3 + rb)))

            # L_h.append(dis0[i] + λ * (Rc(b[0],coefH,shape) + Rb(base)))
            # L_v.append(dis1[i] + λ * (Rc(b[1], coefV, shape) + Rb(base)))
            # L_d.append(dis2[i] + λ * (Rc(b[2], coefD, shape) + Rb(base)))

        self._return = (L_h, L_v, L_d)

    def join(self):
        threading.Thread.join(self)
        return self._return

    def Rb(self,i):
        if i == 16:
            Prb = 1 / 2
        else:
            Prb = 0.5 / 16

        rb = - math.log2(Prb)

        return rb

    def Rc(self,b1, b2, b3, sous_band1, sous_band2, sous_band3, shape):
        rc1 = 0
        rc2 = 0
        rc3 = 0
        i = 0
        while i < 4:
            j = 0
            while j < 4:
                pr1 = (np.sum(sous_band1 == b1[i][j])) / (shape[0] * shape[1])
                rc1 = rc1 - (math.log2(pr1))

                pr2 = (np.sum(sous_band2 == b2[i][j])) / (shape[0] * shape[1])
                rc2 = rc2 - (math.log2(pr2))

                pr3 = (np.sum(sous_band3 == b3[i][j])) / (shape[0] * shape[1])
                rc3 = rc3 - (math.log2(pr3))
                j = j + 1
            i = i + 1

        return rc1, rc2, rc3


