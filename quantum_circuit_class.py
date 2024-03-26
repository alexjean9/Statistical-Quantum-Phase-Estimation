### Written by Alexandria Moore, Purdue University in March 2024. Contact: moore428@purdue.edu ###
import numpy as np

class Quantum_Circuit:
    def __init__(self, control_level, unitary):
        self.set_control_level(control_level)
        self.set_unitary(unitary)
        
    def set_control_level(self, control_level):
        if (control_level.is_integer() == False) or control_level < 2:
            raise ValueError('the control level should be an integer greater or equal to 2')
        self.control_level = control_level
        print('quantum circuit control-level: ', self.control_level)
        
    def set_unitary(self, unitary):
        # check that the matrix is unitary within precision 
        
        # is it square
        n, m = unitary.shape
        if n != m:
            raise ValueError('A unitary matrix should be square')
        
        # remember unitary dimension
        self.unitary_dimension = n
        
        # is it a unitary matrix?
        eye = np.matmul(unitary, unitary.conj().transpose())
        test = np.absolute(eye - np.identity(n))
        if test.sum() < 10**(-8):
            # matrix is a unitary. Remember this and find the eigen-decomposition to speed up computation in quantum_circuit_response
            self.unitary = unitary
            evalues, evectors = np.linalg.eig(unitary)
            
            # convert eigenvalues to eigenphases. value = exp(2*pi*phase *j)
            self.eigenphases = []
            self.eigenvectors = []
            for ii in range(0, n):
                # (  eigen-values in radians: [-pi, pi)  )
                self.eigenphases.append(    np.angle(evalues[ii])   )
                # store eigen-vectors as bra (row vectors). Need to take conj of the vector for this. Make explicitly 2-dimensional
                self.eigenvectors.append(   (np.array([evectors[:,ii]])).conj())
        else:
            raise ValueError('This is not a unitary matrix (to desired precision)')
            
    def quantum_circuit_response(self, eigen_vector_guess, theta):
        # Simulates a PEA circuit with 
        # (1) a rotation gate with the value -1*theta   (theta is in radians, nominally in -pi to pi)
        # (2) an input state of eigen_vector_guess (ket = column vector ("2" dimensional numpy array))
        # and (3) with controlled unitary equal to self.unitary.
        # Returns the probability that the control qudit (or qubit) will be measured in the |0> state.
        # This calculation is ideal. It does not try to model any noise present in the quantum circuit.
        
        # i.e. this function implements Equation 4 in  Moore et al's 2021 "Statistical approach to quantum phase estimation" paper
        
        ## this code does allow for theta to be a one-dimensional array. This is equivalent to running the quantum circuit
         # multiple times with different settings of theta_R. The dimension of the output is equivalent to the dimension of theta.
        
        
        ## INPUT VALIDATION
        # ensure eigen_vector_guess is a n by 1, normalized ket. Explicitly (2-dimensional numpy array)
        m = eigen_vector_guess.shape
        norm = np.linalg.norm(eigen_vector_guess)
        if (len(m) != 2) or (m[1] != 1) or (m[0] != self.unitary_dimension) or (np.absolute(1 - norm) > 10**-8):
            raise ValueError('The input to the target register of the PEA should be a normalized ket (d rows, 1 column)')
        
        # ensure theta is a float, int or an 1D array of such values
        if (isinstance(theta, int)) or (isinstance(theta, float)):
            # convert to a numpy array
            theta = np.array([theta])
            
        # ensure theta is one dimensional
        if theta.ndim != 1:
            raise ValueError('theta should be a one-dimensional numpy array')
        
        # ensure theta is real
        if np.any(np.isreal(theta) == False):
            raise ValueError('theta must be real')
        
        # final check for proper theta
        m = theta.shape
        if (theta.ndim != 1) or (isinstance(theta, np.ndarray) == False):
            raise ValueError('Theta should be a numpy row vector')
            
        ## VALIDATION FINISHED, BEGIN FUNCTION
        C = np.zeros((theta.shape)[0])
        type(C)
        for kk in range(0, self.unitary_dimension):
            # get absolute value of the inner product of the eigen_vector guess and eigenvector kk
            p1 = np.matmul(self.eigenvectors[kk], eigen_vector_guess)
            p2 = np.matmul(p1, p1.conj())
            p2 = p2[0][0]
          
            # get |sum|^2 of the exponential term for each value of theta with respect to eigenvalue kk
            EXP = np.exp(1j * ( self.eigenphases[kk] - theta ))
            exponentials = (np.array([np.arange(self.control_level)])).transpose()
            EXP = np.power(EXP, exponentials)
            # |sum over all control levels|^2 for each value of theta
            SEXP = (np.sum(EXP, axis = 0))
            SEXP = np.multiply(SEXP, SEXP.conj())
            
            # multiply the inner product term by the exponential sum term (for each value of theta)
            # note that both values are purely real, can return as non-complex numpy type
            # sum with current value of C
            C += np.real(p2 * SEXP)

        # finished with all eigenvalues/vectors, return value divided by d_c^2
        return (C/(self.control_level**2))
        
    def nearest_true_eigenvector(self, final_eigenket):
        # find the eigenvector nearest to our ket by taken inner product of (bra) eigenvector and our ket
        inner_products = []
        for ii in range(0, len(self.eigenvectors)):
            foo = np.abs(np.matmul(self.eigenvectors[ii], final_eigenket))
            inner_products.append(foo.copy())
        inner_prod_max = max(inner_products)
        index_max      = inner_products.index(inner_prod_max)
        e_phase_value  = self.eigenphases[index_max]
        
        return inner_prod_max, e_phase_value, index_max
    