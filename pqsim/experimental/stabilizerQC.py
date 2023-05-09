# gatelut = ['h','cx','cz','x','y','z','s','sdg']
from numba import njit, jit, prange
import numpy as np

@njit
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

@njit
def pauli_demux(nqb, p1):
    '''Given an integer encoding a multi-qubit Pauli operator,
    return a list of 1-qb Pauli operators in Little-Endian mode.'''
    plist = np.zeros(nqb, dtype=np.int16)
    
    for idx in range(nqb):
        quot, rem = divmod(p1, 4)
        plist[idx] = rem
        p1 = quot
    
    return plist

@njit
def get_phase_1qb(pterm, op):
    '''Compute global phase incurred while commuting 
    Pauli term 'pterm' through Clifford 'op'.'''
    if op==0:
        if pterm==3:
            return -1.
        else:
            return 1.
    elif op==3: # If op is X
        if pterm==1 or pterm==0:
            return 1.
        else:
            return -1.
    elif op==4: # If op is Y
        if pterm==3 or pterm==0:
            return 1.
        else:
            return -1.
    elif op==5: # If op is Z
        if pterm==2 or pterm==0:
            return 1.
        else:
            return -1.
    elif op==6: # If op is S
        xbit = pterm % 2
        if xbit==1:
            return -1.j
        else:
            return 1.
    elif op==7: # If op is Sdg
        xbit = pterm % 2
        if xbit==1:
            return 1.j
        else:
            return 1.
    else:
        return 1.

@njit
def pauli_commute(pvec, op, qargs, paulimode=0):
    '''Ruleset for propagation of Pauli operators through Clifford gates.
    Integer "op" numbers correspond to Clifford gates as listed in gatelut.
    
    When paulimode==0, when op is a Pauli operator, it is treated like any
    other Clifford op, s.t. a global phase is computed and commutation is otherwise
    trivial. But when paulimode==1, Pauli op is checked against entries in pvec,
    and cancelled when it matches op. This latter mode is useful for fed-forward
    clasically-controlled Pauli ops, such as when doing QEC correction rounds.'''
    phase = 1.
    if op==0:
        qb = qargs[0]
        phase = get_phase_1qb(pvec[qb], op)
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

        if (pvec[src]%2==1) and (pvec[trg]%2==1):
            phase = -1.
        pvec[src] = newsrc
        pvec[trg] = newtrg
    
    elif op==3:
        if paulimode==0:
            phase = get_phase_1qb(pvec[qargs[0]], op)
        else:
            pvec[qargs[0]] = pauli_merge(pvec[qargs[0]], 1)
    
    elif op==4:
        if paulimode==0:
            phase = get_phase_1qb(pvec[qargs[0]], op)
        else:
            pvec[qargs[0]] = pauli_merge(pvec[qargs[0]], 3)
    
    elif op==5:
        if paulimode==0:
            phase = get_phase_1qb(pvec[qargs[0]], op)
        else:
            pvec[qargs[0]] = pauli_merge(pvec[qargs[0]], 2)

    elif op==6 or op==7:
        phase = get_phase_1qb(pvec[qargs[0]], op)
        zbit = pvec[qargs[0]]//2
        xbit = pvec[qargs[0]]%2
        zbit = (zbit + xbit)%2
        pvec[qargs[0]] = 2*zbit + xbit

    return phase

@njit
def check_condition(cvals, condval, contype, conbits):
    '''Checks if a classically-controlled conditional gate is to be applied.
    Args:
        cvals   : Classical-register that controls the gate.
        condval : Value that determines if a gate is applied.
        contype : Integer determining how condval is interpreted;
                If 0, ignore;
                If 1, apply gate when classical bit in cvals is equal to condval;
                If 2, apply gate when classical bit in cvals is NOT condval.
        conbits : List of integer indices specifying which entries in cvals apply.'''
    val = 0
    for idx,cb in enumerate(conbits):
        val = val + 2**idx * cvals[cb]
    
    goflag = False
    if contype==1:
        goflag = (val==condval)
    elif contype==2:
        goflag = (val!=condval)
    elif contype==0:
        goflag = True
    
    return goflag

@njit
def propagate(pvec, carr, noisearray, opnames, noiseid, condval, contype, conbits, opqargs, opcargs, paulimode=0):
    '''Given a starting list of Pauli operators 'pvec', and list of classical bits 'carr', commute
    pvec through Clifford circuit defined by:
        noisearray  : An integer array encoding Pauli errors afflicting each Clifford gate.
        opnames     : Gate names in the circuit (must be member of gatelut)
        noiseid     : A Boolean array indicating whether a gate is noisy.
        condval     : Parameter for classical-control of a gate (see check_condition())
        contype     : Parameter for classical-control of a gate (see check_condition())
        conbits     : Parameter for classical-control of a gate (see check_condition())
        opqargs     : Qubits acted upon by gate.
        opcargs     : Classical bits used by or that determines action of gate.'''
    idx_noise = 0
    phase = 1.

    for idx, name in enumerate(opnames):
        if name<9:
            do_gate = check_condition(carr, condval[idx], contype[idx], conbits[idx])
            if do_gate:
                phase *= pauli_commute(pvec, name, opqargs[idx], paulimode=paulimode)
        
        elif name==9:
            carr[opcargs[idx][0]] = pvec[opqargs[idx][0]]%2
            pvec[opqargs[idx][0]] = 0

        elif name==10:
            for bit in conbits[idx]: # Fail if pvec term is one of conbits
                if pvec[opqargs[idx][0]]==bit:
                    raise Exception("Does not commute!")

        if noiseid[idx]:
            noisedim = len(opqargs[idx])
            pchar = pauli_demux(noisedim, noisearray[idx_noise])
            idx_noise = idx_noise + 1

            for idx_char,pc in enumerate(pchar):
                pvec[opqargs[idx,idx_char]] = pauli_merge(pvec[opqargs[idx,idx_char]], pc)

    return phase

@njit(parallel=True)
def propagate_all_samples(qlen, clen, noisearrays, idx_qlog, syndromes,\
    logicalops, opnames, noiseid, condval, contype, conbits, opqargs, opcargs):
    '''Like propagate() above, only this acts over many possible instances of Pauli errors.
    So, instead of a single 'noisearray' as in propagate() above, pass in noisearrays as a
    LIST of noisearray.'''
    pvecs = np.zeros((len(noisearrays),qlen), dtype=np.int16)
    carrs = np.zeros((len(noisearrays),clen), dtype=np.int16)

    checks = np.zeros(len(noisearrays), dtype=np.int8)

    for idx in prange(len(noisearrays)):
        pvec = np.zeros(qlen, dtype=np.int16)
        carr = np.zeros(clen, dtype=np.int16)
        propagate(pvec, carr, noisearrays[idx], opnames, noiseid,\
            condval, contype, conbits, opqargs, opcargs)
        
        pvecs[idx] = pvec
        carrs[idx] = carr

    return checks, pvecs, carrs
