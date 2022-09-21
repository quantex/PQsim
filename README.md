# Pedagogical Quantum Simulator

## Introduction

This project implements a simple quantum circuit simulator using the statevector formalism.
Despite being very compact, it supports the simulation of a universal set of operators and can in-principle simulate arbitrary quantum circuits.
The compactness of the implementation is by design, and serves a primary pedagogical goal: to make it easy for anyone who can read basic Python / NumPy code to peer into the machinery of a (virtual) quantum computer.
If you can understand basic matrix and vector multiplication, then seeing how this project simulates a quantum computer removes much of the mystery surrounding quantum computing!

Core numerical routines use plain Python and NumPy syntax.
The user can select between executing the simulation either on a pure Python or a NumPy "backend".
To keep things simple, only a core set of "gates" are supported: (i) the Hadamard gate, (ii) an arbitrary complex 2x2 unitary matrix, and (iii) the so-called controlled-Z or CZ gate (represented as a 4x4 unitary matrix).
Jointly, these gates are universal -- that is, they are sufficient for representing any quantum logic circuit.

Not a lot is done in the way of heavy-handed manual code optimization which can sometimes obscure the logic of the code, thereby diminishing its pedagogical value.
However, in order to maintain respectable simulation performance, the Numba just-in-time (JIT) compiler is used.
This can have a tremendous effect at speeding up code, particularly when using the pure Python backend.
The user can always turn off this JIT compilation by removing the "@njit" decorators wherever they exist.

Experimentation with this codebase is kinda the point!
Despite its simplicity, the implementation here is flexible enough to be used for non-trivial calculations.
The author had used it in the past as a research tool, wherever quick-and-dirty calculations and the flexibility of being able to manipulate the statevector directly were needed; larger software packages sometimes just get in the way of figuring out what's going on!
Even without careful hand optimization and despite its humble Pythonic roots, the Numba-accelerated pure Python backend can routinely handle circuits in excess of 20 qubits and containing many hundreds of gates, on a machine with a Skylake Core i5-8500 processor and 16GB of DDR4 2666MHz memory.

## Installation

Clone this repository (say to folder `pqsim`).
Then, in a Python environment (it must have `pip`), simply invoke: `pip install pqsim`.
Pre-requisites (Numba and NumPy) should be installed automatically.
If you wish to experiment with the simulator code directly (again, this is encouraged), invoke `pip install -e pqsim` instead.
This will allow any changes you make to propagate to the installed package that you'll import into other projects later.

An example of how the simulator may be used in practice is contained within the folder `examples`.
