import jax.numpy as jnp
from jax import jit
import numpy as np

def do_circ(nq, names, qargs, parms, vec):
    idx_parm = 0
    idx_arr = jnp.arange(2**nq, dtype=int)
    for idx, n in enumerate(names):
        if n=='cz':
            q0 = qargs[idx] % nq
            q1 = ((qargs[idx] - q0)//nq) % nq
            vec = cz(vec, idx_arr, q0, q1)
        elif n=='h':
            vec = h(vec, idx_arr, qargs[idx])
        elif n=='u':
            op = parms[idx_parm]
            idx_parm += 1
            vec = u(vec, idx_arr, op, qargs[idx])

    return vec

@jit
def cz(state, idxState, qb1, qb2):
    '''
    Given an index, figure out where a CNOT will permute it to.
    '''
    strip1 = 2**qb1
    stride1 = 2*strip1
    strip2 = 2**qb2
    stride2 = 2*strip2

    is11 = jnp.floor_divide(jnp.mod(idxState, stride1), strip1)*\
        jnp.floor_divide(jnp.mod(idxState, stride2), strip2)
    
    sign_array = 1-2*is11

    return state*sign_array

@jit
def h(state, idxState, qb):
    strip = 2**qb
    stride = 2*strip

    # Array of 0's and 1's, indicating the state of the target qb
    is1 = jnp.floor_divide(jnp.mod(idxState, stride), strip)
    is0 = 1-is1

    idx0 = jnp.mod(idxState, strip) + jnp.floor_divide(idxState, stride)*stride
    sgn = 1-2*is1

    return (jnp.take(state, idx0) + jnp.take(state, idx0+strip)*sgn)/jnp.sqrt(2)

@jit
def u(state, idxState, op, qb):
    strip = 2**qb
    stride = 2*strip

    # Array of 0's and 1's, indicating the state of the target qb
    targState = jnp.floor_divide(jnp.mod(idxState, stride), strip)

    idx0 = jnp.mod(idxState, strip) + jnp.floor_divide(idxState, stride)*stride

    return jnp.take(state, idx0)*op[targState,0] + jnp.take(state, idx0+strip)*op[targState,1]

@jit
def modulate_1qb(state, idxState, modulator, qb):
    strip = 2**qb
    stride = 2*strip

    # Array of 0's and 1's, indicating the state of the target qb
    targState = jnp.floor_divide(jnp.mod(idxState, stride), strip)

    return state*modulator[targState]

@jit
def modulate_2qb(state, idxState, modulator, qb1, qb2):
    '''
    Modulator is arranged as qb1,qb2 = [00, 10, 01, 11].
    '''
    strip1 = 2**qb1
    stride1 = 2*strip1
    strip2 = 2**qb2
    stride2 = 2*strip2

    is_a_1 = jnp.floor_divide(jnp.mod(idxState, stride1), strip1)
    is_b_1 = jnp.floor_divide(jnp.mod(idxState, stride2), strip2)
    idx_modulator = 2*is_b_1 + is_a_1

    return state*modulator[idx_modulator]