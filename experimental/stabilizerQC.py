def pauli_merge(p1, p2):
    '''
    Compose two Pauli operators encoded as integers.
    We interpret integers p1 and p2 as 2*(z-bit) + (x-bit).
    '''
    zbit_1 = p1//2
    xbit_1 = p1 %2
    zbit_2 = p2//2
    xbit_2 = p2 %2

    zbit = (zbit_1 + zbit_2)%2
    xbit = (xbit_1 + xbit_2)%2

    return xbit + 2*zbit

def pauli_demux(nqb, p1):
    '''Given an integer encoding a multi-qubit Pauli operator,
    return a list of 1-qb Pauli operators in Little-Endian mode.'''
    plist = np.zeros(nqb, dtype=np.int16)
    
    for idx in range(nqb):
        quot, rem = divmod(p1, 4)
        plist[idx] = rem
        p1 = quot
    
    return plist