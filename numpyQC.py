import numpy as np
from numba import njit

@njit
def cz(n, qb0, qb1, vec):
    '''
    Does a n-qubit cz on vec, between qb0 and qb1
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
    count = 2**(n-qb0-1)
    count_r = 2**(qb0)

    temp0 = op[0,0]*vec.reshape(count,2,count_r)[:,0,:] + op[0,1]*vec.reshape(count,2,count_r)[:,1,:]
    
    vec.reshape(count,2,count_r)[:,1,:] = \
        op[1,0]*vec.reshape(count,2,count_r)[:,0,:] + op[1,1]*vec.reshape(count,2,count_r)[:,1,:]

    vec.reshape(count,2,count_r)[:,0,:] = temp0

@njit
def modulate_2qb(n, qb0, qb1, modulator, vec):
    '''
    Modulator is arranged as qb1,qb2 = [00, 10, 01, 11].
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