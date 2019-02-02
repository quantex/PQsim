# Import this package.
import pqsim as sim

import numpy as numpy
import qiskit.circuit as qc

# Instantiate this simulator
mysimulator = sim.qsim(backend='numba')

# Define a simple quantum circuit. This one produces an entangled Bell state.
nq = 2 # Number of qubits in total
gates = np.array(['h', 'h', 'cz', 'h']) # Name of gates to apply
qargs = np.array([[0,-1], [1,-1], [0,1], [1,-1]], dtype=int) # Qubits to target. "-1" is a placeholder
parms = np.zeros((0,2,2), dtype=complex) # Empty gate parameters. H and CZ are un-parameterized

# Run and get a statevector
result0 = mysimulator.run(2, gates, qargs, params)

# Define an identical circuit using QISkit
testqc = qc.QuantumCircuit(nq)
testqc.h(0)
testqc.h(1)
testqc.cz(0,1)
testqc.h(1)

# Convert to PQsim lists
stats = sim.qsim.get_circ_stat(testqc)
gates_qk, qargs_qk, parms_qk = sim.qsim.get_circ_data(testqc, stats)

# Run and get statevector
result1 = mysimulator.run(nq, gates_qk, qargs_qk, parms_qk)

# Compare
print("Inner product: ", np.dot(result0.conjugate(), result1))