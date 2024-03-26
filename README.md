# Statistical-Quantum-Phase-Estimation
Two python classes and demo code which can be used to simulate the classical-quantum variational algoirthm described in A. J. Moore et al 2021 New J. Phys. 23 113027. https://doi.org/10.1088/1367-2630/ac320d

***

# Statistical Approach to Quantum Phase Estimation example code

## Background
### Quantum phase estimation
Quantum phase estimation is a (quantum) algorithm which uses $n$ "control" qubits (or qudits) and a series of controlled-unitary operators to ultimately determine the phase of one of the eigenvalues (i.e. an *eigen*phase) of the unknown unitary matrix. The quantum circuit will return an estimate of the eigenphase to a precision of $2^{-n}$ in the case of qubit controls or to a precision of $d^{-n}$ in the case of qudit controls.

By eigenphase we mean this: the eigenvalues of a unitary matrix will always have magnitude one. Therefore, aside from +1 and -1, all eigenvalues are complex. Then we can represent an eigenvalue as 

$eigenvalue = e^{j \cdot eigenphase}$

where *j* is the imagionary number and *eigenphase* is any real number (in radians). That is, the eigenvalue +1 has a an eigenphase of 0 radians. The eigenvalue -1 has an eigenphase of pi/2 radians. Frequently $eigenphase \in \[0, 2 pi\)$ is used, but $eigenphase \in \[-pi, pi\)$ is also valid.

Generally, in quantum phase estimation, the circuit returns a random eigenphase after each run. Nominally, the user cannot prevent the circuit from returning a previously determined eigenphase.

### Statistical Approach to Quantum Phase Estimation
The 2021 article [Statistical approach to quantum phase estimation](https://doi.org/10.1088/1367-2630/ac320d) by A.J. Moore, Y. Wang, Z. Hu, S. Kais, and A.M. Weiner used a modified version of the quantum phase estimation algorithm (PEA) in addition to a classical controller. This variation algorithm uses a quantum circuit to return the "cost" function which the classical controller tries to drive to one. The classical controller controls the quantum input "target" state to the modified PEA as well as the phase applied to the PEA rotation gate. This amounts to the classical controller guessing at the value of an eigenvector-eigenphase pair and receiving the cost function from the quantum circuit as feedback.

## This code
This code simulates (under ideal conditions) the variational classical-quantum optimizer illustrated in Figure 4 of [Moore et al](https://doi.org/10.1088/1367-2630/ac320d) with the classical controller behaving as described in Section 3 of that paper (the "Standard Approach" to the classical optimizer is used). The **quantum_circuit_class** simulates the response of the quantum circuit and the **classical_controller_class** implements (a single optimization loop) of the classical controller's behavior. Additional methods and functions help a user analyze and interpret the results.

The code can simulate quantum circuits with control qubits or qudits of any dimension. The code can simulate this code acting on an arbitrary unitary of any dimension. The **main.py** file provides two demonstrations for the use of the classical and quantum classes.
- 	Demo 1 illustrates using the classical and quantum classes to return a single eigenvalue-eigenphase pair an unknown unitary
- 	Demo 2 illustrates using the classical and quantum classes iteratively to determine all eigenvalue-eigenphase pairs of an unknown unitary. That is, demo 2 performs a full spectral decomposition on the unknown unitary.

## Code origin
This code was written by the first author of [Statistical approach to quantum phase estimation](https://doi.org/10.1088/1367-2630/ac320d), Alexandria J Moore, by request in March 2024. This is not the original (matlab) code used to generate all (non-qiskit) results in the original paper. However, brief testing suggests that this code performs similarly, if not slightly better, than the original project code. As this was written after publication, the comments of this code make explicit reference to equations, sections, and figures in the 2021 paper.

## Use and distribution
This code can be distributed, updated, modified by anyone for any purpose.
