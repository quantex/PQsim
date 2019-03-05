# gatelut = ['h','cx','cz','s']

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

def pauli_commute(pvec, op, qargs):
    '''Ruleset for propagation of Pauli operators through Clifford gates.
    Integer "op" numbers correspond to Clifford gates as listed in gatelut.'''
    if op==0:
        qb = qargs[0]
        if pvec[qb]==1:
            pvec[qb] = 2
        elif pvec[qb]==2:
            pvec[qb] = 1
    
    elif op==1:
        src = qargs[0]
        trg = qargs[1]

        if pvec[src]%2==1: # If there is an X Pauli at control qb
            newtrg = pauli_merge(pvec[trg], 1)
        else:
            newtrg = pvec[trg]
        
        if pvec[trg]//2==1: # If there is a Z Pauli at target qb
            newsrc = pauli_merge(pvec[src], 2)
        else:
            newsrc = pvec[src]
        
        pvec[src] = newsrc
        pvec[trg] = newtrg
    
    elif op==2:
        src = qargs[0]
        trg = qargs[1]

        if pvec[src]%2==1:
            newtrg = pauli_merge(pvec[trg], 2)
        else:
            newtrg = pvec[trg]
        
        if pvec[trg]%2==1:
            newsrc = pauli_merge(pvec[src], 2)
        else:
            newsrc = pvec[src]

        pvec[src] = newsrc
        pvec[trg] = newtrg

    elif op==3: 
        zbit = pvec[qargs[0]]//2
        xbit = pvec[qargs[0]]%2
        zbit = (zbit + xbit)%2
        pvec[qargs[0]] = 2*zbit + xbit