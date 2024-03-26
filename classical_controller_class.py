### Written by Alexandria Moore, Purdue University in March 2024. Contact: moore428@purdue.edu ###
import numpy as np

# implements Standard method form Moore et al Statistical Approach to Quantum Phase Estimation paper
class Classical_Controller:
    def __init__(self, unitary_dimension, **kwargs):  #kwargs:   ini_ket   exclude_dir
        # set basics
        self.theta = np.linspace(-1*np.pi, np.pi, num=30, endpoint=False)
        self.fine_theta = False
        self.iterations = 0
        self.consecutive_fail_count = 0
        
        # set unitary dimension
        if (type(unitary_dimension) != int) or unitary_dimension < 2:
            raise ValueError('The unitary dimension should be an integer greater or equal to 2')
        else:
            self.unitary_dimension = unitary_dimension
  
        # directions we do not allow the ket to search along
        self.set_exclude_dir(kwargs.get('exclude_dir', None))
      
        # set initial ket (i.e. step (a))
        self.set_ini_ket(kwargs.get('ini_ket', None))
        
        # keep current best eigenphase in records, default to none
        self.current_ephase = None
        self.current_qcircuit_return = None
        
    def set_exclude_dir(self, exclude_dir):
        # initialize to none
        self.exclude_dir = None
        
        # unless a valid exclude_dir is provided
        if type(exclude_dir) == np.ndarray:
            n = exclude_dir.shape
            if (n[0] == self.unitary_dimension) and (n[1] < (self.unitary_dimension)) and (len(n)==2):
                # n must be the correct dimension and m must be LESS than the dimension (otherwise the dimension is trivial)
                # finally, check for orthonormality
                expected_identity = np.matmul( (exclude_dir.conj()).transpose(), exclude_dir)
                if np.sum(np.abs(expected_identity - np.identity(n[1]) )) < 10**-8:
                    self.exclude_dir = exclude_dir
                    print('excluded directions set successfully')
                else:
                    print('Error: excluded directions not applied')
            else:
                print('Error: excluded directions not applied')
            

    def set_ini_ket(self, ini_ket):
        # initialize random ket, avoid excluded dimensions
        B = self.orthog_basis(self.exclude_dir)
        n, m = B.shape
        r1 = np.random.rand(m)
        r1 = r1 / np.sqrt(np.sum(np.square(r1)))
        r2 = np.exp(1j * 2 * np.pi * np.random.rand(m))
        
        v = (0 + 0j)* np.zeros((n,1))
        for ii in range(0,m):
            v += r1[ii]*r2[ii] * np.transpose(np.array([B[:,ii]]))
        self.current_ket = v
        
        # overwrite & initialize to user-give ket if of the right type and form
        if type(ini_ket) == np.ndarray:
            if (ini_ket.ndim == 2):
                n, m = ini_ket.shape
                if (m == 1) and (n == self.unitary_dimension):
                    self.current_ket = ini_ket

    def run_single_optimization_loop(self, quantum_circuit):    
        # step (b), construct an orthonormal set of directions to test in
        if self.exclude_dir is None:
            B = self.orthog_basis(self.current_ket)
        else:
            B = self.orthog_basis(np.append(self.current_ket, self.exclude_dir , axis = 1))        
        
        # step (c)   [standard method]
        C = quantum_circuit.quantum_circuit_response(self.current_ket , self.theta)
        C_max = np.max(C)
        index_max = np.argmax(C)
        
        # increase theta resolution once high C value is reached
        if self.fine_theta == False:
            self.update_theta(C_max, index_max, quantum_circuit.control_level)
        
        # step (d)
        update_flag = False
        abort_flag = False
        n,m = B.shape
        a_mult = 2
        while update_flag == False and abort_flag == False:
            if (a_mult < 10**-12):
                abort_flag = True
            a_mult = a_mult/2
            for zz in [1, 1j]:
                for ii in range(0, m):
                    new_ket = self.current_ket + (zz * a_mult * (1 - C_max) * np.transpose(np.array([B[:,ii]])))
                    new_ket = new_ket / np.sqrt( np.matmul( (new_ket.transpose()).conj(), new_ket ))
                    C_star = quantum_circuit.quantum_circuit_response(new_ket , self.theta)
                    C_star_max = np.max(C_star)
                    if C_star_max > C_max:
                        self.current_ket = new_ket.copy()
                        C_max = C_star_max.copy()
                        update_flag = True
                        self.consecutive_fail_count = 0
        
        if update_flag == False:
            self.consecutive_fail_count += 1
        self.current_qcircuit_return = C_max
        self.iterations += 1
        return C_max
        
        
    def update_theta(self, C_max, index_max, d_c):
        if C_max > (.96 - (d_c-2)/65  ) and (self.fine_theta == False):
            self.fine_theta = True
            theta_max = self.theta[index_max];
            theta_r = np.linspace(theta_max-(np.pi/(d_c*2)), theta_max+(np.pi/(d_c*2)), 160)
            theta_r = ( theta_r + np.pi) % (2 * np.pi ) - np.pi
            self.theta = np.sort(np.append(self.theta, theta_r))
    
    def extra_fine_phase(self, quantum_circuit):
        for ii in range(0,3):
            if ii == 0:
                theta = np.linspace(-1*np.pi, np.pi, num=200, endpoint=False)
                C = quantum_circuit.quantum_circuit_response(self.current_ket , theta)
                C_max = np.max(C)
                index_max = np.argmax(C)
                theta_max = theta[index_max]

            if index_max >0 and index_max < (len(theta)-1):
                second_der = theta[index_max + 1] - theta[index_max -1]
            elif index_max == 0:
                second_der = 2*(theta[1] - theta[0])
            else:
                second_der = 2*(theta[index_max] - theta[index_max-1])
            theta_r2 = np.linspace(theta_max-second_der, theta_max+second_der, 50)
            theta = np.sort(( theta_r2 + np.pi) % (2 * np.pi ) - np.pi)
            C = quantum_circuit.quantum_circuit_response(self.current_ket , theta)
            C_max = np.max(C)
            index_max = np.argmax(C)
            theta_max = theta[index_max]

        self.current_ephase = theta_max
        self.current_qcircuit_return = C_max
        return theta_max, C_max

    def orthog_basis(self, A):
        # generates a set of complex orthogonal vectors which, when combined with the vectors in A, form a complete orthonormal basis
        
        ## Special behavior is A is None
        if A is None:
            return Classical_Controller.random_unitary(self.unitary_dimension)
       
        ## VALIDATE INPUT A
        if A.ndim != 2:
            raise ValueError('A should be a two-dimensional numpy array')      
        n, m = A.shape
        if n <= m:
            raise ValueError('A should be comprised of orthogonal column vectors. There should be fewer vectors than the dimension of the vectors')
            
        for nn in range(0, m):
            # check for orthogonal
            for mm in range(0, nn):
                inner_product = np.dot(A[:,nn].conj(), A[:,mm])
                if np.abs(inner_product) > 10**-8:
                    # not orthogonal
                    raise ValueError('the column vectors in A must be orthogonal')
            # check for normal
                inner_product = np.dot(A[:,nn].conj(), A[:,nn])
                if np.abs(inner_product - 1) > 10**-8:
                    # not normal
                    raise ValueError('the column vectors in A must be normalized')
                    
        ## INPUT VALIDATION FINISHED
        # generate a random unitary, thereby generating a random orthonormal basis B
        B = Classical_Controller.random_unitary(n)
        # for the following procedure to work, no vector in A can be identical to the latter (m-n) vectors in B
        test = True
        while test == True:
            test = False
            # test against relevant vectors
            for nn in range(0, m):
                for mm in range(m, n):
                    inner_product = np.dot(A[:,nn].conj(), B[:,mm])
                    if np.abs(inner_product) > (1- (10**-6)):
                        # choose another basis and restart
                        B = Classical_Controller.random_unitary(n)
                        test = True
        # remove first m rows from B
        #B = np.delete(B, range(0, m), 1)
        
        
        # construct the rest of A using B columns m through finish
        for nn in range(m, n):
            u = (B[:, nn]).copy()
            q,r = A.shape
            s = (0+0j)*np.zeros(n)
            for mm in range(0,r):
                s += (np.dot(A[:, mm].conj(), u)) * A[:, mm]
            new_A = u - s
            new_A = new_A / np.sqrt(np.dot(new_A.conj(), new_A))
            A = np.append(A,  (np.array([new_A])).transpose(), axis = 1)


        # remove first m columns from A and return 
        return np.delete(A, range(0, m), 1)

    
    @staticmethod
    def random_unitary(n):
        # generates random n-by-n dimensional unitary matrix
        x = (np.random.rand(n,n) - np.random.rand(n,n)) + 1j * (np.random.rand(n,n) - np.random.rand(n,n))
        Q, R = np.linalg.qr(x)
        r = np.diag(R)
        L = np.diag(  np.divide(  r,  np.absolute(r)   ))
        return np.matmul(Q, L)
    @staticmethod
    def random_ket(n):
        # generates random n-by-1 complex vector (ket), normalized
        x = (np.random.rand(n,1) - np.random.rand(n,1)) + 1j * (np.random.rand(n,1) - np.random.rand(n,1))
        norm = np.matmul( (x.conj()).transpose(), x)
        return x / np.sqrt(norm)
