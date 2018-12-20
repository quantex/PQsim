import numpy as np
from numba import njit

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

@njit
def cz(n, qb0, qb1, vec):
    '''
    Does a controlled-Z operation between qb0 and qb1,
    on vec, an n-qubit statevector.
    '''
    q0, q1 = (max(qb0,qb1), min(qb0,qb1))

    count_large = 2**(n-q0-1)
    count_small = 2**(q0-q1-1)

    vec.reshape(count_large,2,count_small,2,-1)[:,1,:,1,:] = \
        -vec.reshape(count_large,2,count_small,2,-1)[:,1,:,1,:]

@njit
def z(n, qb0, vec):
    count = 2**(n-qb0-1)
    count_r = 2**(qb0)

    vin = vec.reshape(count,2,count_r)

    vin[:,1,:] = -vin[:,1,:]

    return vin.reshape(2**n)

@njit
def h(n, qb0, vec):
    '''
    Does a Hadamard operation targeting qb0,
    on vec, an n-qubit statevector.
    '''
    count = 2**(n-qb0-1)
    count_r = 2**(qb0)

    sq2d= 1/np.sqrt(2)
    sq2 = np.sqrt(2)

    vec.reshape(count,2,count_r)[:,0,:] =\
        (vec.reshape(count,2,count_r)[:,0,:] + vec.reshape(count,2,count_r)[:,1,:])*sq2d
    vec.reshape(count,2,count_r)[:,1,:] = \
        vec.reshape(count,2,count_r)[:,0,:] - sq2*vec.reshape(count,2,count_r)[:,1,:]

@njit
def apply_1qb(n, op, qb0, vec):
    '''
    Applies a 1-qubit operator "op" targeting qb0,
    on vec, an n-qubit statevector.
    '''
    count = 2**(n-qb0-1)
    count_r = 2**(qb0)

    temp0 = op[0,0]*vec.reshape(count,2,count_r)[:,0,:] + op[0,1]*vec.reshape(count,2,count_r)[:,1,:]
    
    vec.reshape(count,2,count_r)[:,1,:] = \
        op[1,0]*vec.reshape(count,2,count_r)[:,0,:] + op[1,1]*vec.reshape(count,2,count_r)[:,1,:]

    vec.reshape(count,2,count_r)[:,0,:] = temp0

@njit
def modulate_2qb(n, qb0, qb1, modulator, vec):
    '''
    Modulates vec, an n-qubit statevector,
    with a 2-qubit modulator pattern.

    This is equivalent to treating the modulator as
    a diagonal 2-qubit operator acting on qb0 and qb1.
    '''
    q0, q1 = (max(qb0,qb1), min(qb0,qb1))

    if qb1>qb0:
        temp = modulator[2]
        modulator[2] = modulator[1]
        modulator[1] = temp

    count_large = 2**(n-q0-1)
    count_small = 2**(q0-q1-1)

    vec.reshape(count_large,2,count_small,2,-1)[:,0,:,0,:] = \
        modulator[0]*vec.reshape(count_large,2,count_small,2,-1)[:,0,:,0,:]

    vec.reshape(count_large,2,count_small,2,-1)[:,1,:,0,:] = \
        modulator[1]*vec.reshape(count_large,2,count_small,2,-1)[:,1,:,0,:]
        
    vec.reshape(count_large,2,count_small,2,-1)[:,0,:,1,:] = \
        modulator[2]*vec.reshape(count_large,2,count_small,2,-1)[:,0,:,1,:]
    
    vec.reshape(count_large,2,count_small,2,-1)[:,1,:,1,:] = \
        modulator[3]*vec.reshape(count_large,2,count_small,2,-1)[:,1,:,1,:]