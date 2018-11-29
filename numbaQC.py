from numba import njit, prange

@njit(parallel=True)
def cz(n, qb0, qb1, vec):
    '''
    Does a n-qubit cz on vec, between qb0 and qb1
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
    stripe_1 = 2**(qb1)
    stripe_0 = 2**(qb0)
    
    for i in prange(2**n):
        i0 = i%(2**qb0) # qb0 value
        i1 = i%(2**qb1) # qb1 value

        # Modulate
        vec[i] = modulator[2*i1 + i0]*vec[l]