import numpy as np
import numbaQC
import numpyQC

class qsim():
    def __init__(self, backend='numba'):
        self.backend = backend
        if backend=='numba':
            self.nQC = numbaQC
        elif backend=='numpy':
            self.nQC = numpyQC

    def run(self, nq, names, qargs, parms, vec):
        self.nQC(nq, names, qargs, parms, vec)

    @staticmethod
    def get_circ_data(circ, gate_dict):
        lut = circ.qubits
        nqb = len(lut)
        ngates = len(circ)
        try:
            nugate = gate_dict['u']
        except:
            nugate = 0

        names = np.repeat(['aa'],ngates)
        qargs = np.zeros((ngates,2),dtype=int)-1
        parms = np.empty((nugate,2,2),dtype=complex)
        idx = 0
        idx_p = 0
        for dat in circ:
            if dat[0].name in ['cz', 'h', 'u', 'rz']:
                names[idx] = dat[0].name
                for qarg,qb in enumerate(dat[1]):
                    qargs[idx,qarg] = lut.index(qb)

                idx += 1

                if dat[0].name=='u':
                    parms[idx_p] = dat[0].to_matrix()
                    idx_p += 1
                if dat[0].name=='rz':
                    parms[idx_p,0,0] = dat[0].params[0]
                    idx_p += 1

        return names, qargs, parms

    @staticmethod
    def get_circ_stat(circ):
        gates = {}
        for dat in circ:
            name = dat[0].name
            try:
                gates[name] += 1
            except:
                gates.update({name:1})

        return gates