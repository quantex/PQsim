import numpy as np
from . import numbaQC
from . import numpyQC

class qsim():
    def __init__(self, backend='numba'):
        '''
        Initialize the convenience class.
        '''
        self.backend = backend
        if backend=='numba':
            self.nQC = numbaQC
        elif backend=='numpy':
            self.nQC = numpyQC

    def run(self, nq, names, qargs, parms, vec=[]):
        '''
        Initiate run of a circuit. If no statevector
        is provided, initialize a fresh statevector in
        the zero state.
        '''
        returnstate = False

        if len(vec)!=2**nq:
            vec = np.zeros(2**nq, dtype=complex)
            vec[0] = 1.
            returnstate = True

        self.nQC.do_circ(nq, names, qargs, parms, vec)

        if returnstate:
            return vec

    @staticmethod
    def get_circ_data(circ, gate_dict):
        '''
        This is a compatibility layer for QISkit.
        Given a QISkit circuit, return the lists of instructions
        and parameters necessary to call do_circ().
        Args:
            circ:      A QISkit circuit object.
            gate_dict: A dictionary enumerating all types / count of gates in circ.
        '''
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
        '''
        Given a QISkit circuit, count up the number of each type of gate.
        '''
        gates = {}
        for dat in circ:
            name = dat[0].name
            try:
                gates[name] += 1
            except:
                gates.update({name:1})

        return gates