import numpy as np
from numba import njit, prange

@njit
def do_circ(nq, names, qargs, parms, vec):
    '''
    Runner function. Calls appropriate numerical routines below
    to effect gates as listed in names, qargs, and parms.

    Args:
        names: List of strings indicating which gate to perform.
        qargs: List of integer 2-tuples, indicating which qubit(s) the gate is acting on.
        parms: List of parameter(s) where required, i.e. when a gate is parameterized.
        vec:   An input statevector.
    '''
    idx_parm = 0
    for idx, n in enumerate(names):
        if n=='cz':
            cz(nq, qargs[idx][0], qargs[idx][1], vec)
        elif n=='h':
            h(nq, qargs[idx][0], vec)
        elif n=='u':
            op = parms[idx_parm]
            idx_parm += 1
            apply_1qb(nq, op, qargs[idx][0], vec)
        elif n=='mod2qb':
            mod = parms[idx_parm].reshape(4)
            idx_parm += 1
            modulate_2qb(nq, qargs[idx][0], qargs[idx][1], mod, vec)

@njit(parallel=True)
def cz(n, qb0, qb1, vec):
    '''
    Does a controlled-Z operation between qb0 and qb1,
    on vec, an n-qubit statevector.
    '''
    q0 = min(qb0,qb1)
    q1 = max(qb0,qb1)

    stripe_1 = 2**(q1)
    stride_1 = 2*stripe_1
    stripe_0 = 2**(q0)
    stride_0 = 2*stripe_0
    count_01 = 2**(q1-q0-1)
    
    for i in prange(2**(n-2)):
        quot0 = i//stripe_0

        j = quot0 % count_01 # Which small stripe
        quot1 = quot0//count_01 # Which big stripe
        
        k = i % stripe_0 # Which local addr.

        # Build full address
        l = (quot1*stride_1 + stripe_1) + (j*stride_0 + stripe_0) + k
        
        vec[l] = -vec[l]

@njit(parallel=True)
def h(n, qb0, vec):
    '''
    Does a Hadamard operation targeting qb0,
    on vec, an n-qubit statevector.
    '''
    count = 2**(n-1)
    stripe = 2**qb0
    stride = 2*stripe
    sq2d = 1/np.sqrt(2.)
    
    for i in prange(count):
        quot = i//stripe
        j = i%stripe + quot*stride
        k = j+stripe
        temp0 = (vec[j] + vec[k])*sq2d
        vec[k] = (vec[j] - vec[k])*sq2d
        vec[j] = temp0

@njit(parallel=True)
def apply_1qb(n, op, qb0, vec):
    '''
    Applies a 1-qubit operator "op" targeting qb0,
    on vec, an n-qubit statevector.
    '''
    count = 2**(n-1)
    stripe = 2**qb0
    stride = 2*stripe
    
    for i in prange(count):
        quot = i//stripe
        j = i%stripe + quot*stride
        k = j+stripe
        temp0 = (op[0,0]*vec[j] + op[0,1]*vec[k])
        vec[k] = (op[1,0]*vec[j] + op[1,1]*vec[k])
        vec[j] = temp0

@njit(parallel=True)
def modulate_2qb(n, qb0, qb1, modulator, vec):
    '''
    Modulates vec, an n-qubit statevector,
    with a 2-qubit modulator pattern.

    This is equivalent to treating the modulator as
    a diagonal 2-qubit operator acting on qb0 and qb1.
    '''
    stripe_1 = 2**(qb1)
    stripe_0 = 2**(qb0)
    
    for i in prange(2**n):
        i0 = i%(2**qb0) # qb0 value
        i1 = i%(2**qb1) # qb1 value

        # Modulate
        vec[i] = modulator[2*i1 + i0]*vec[i]