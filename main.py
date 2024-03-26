### Written by Alexandria Moore, Purdue University in March 2024. Contact: moore428@purdue.edu ###
import numpy as np
import time

# my files
from quantum_circuit_class import Quantum_Circuit
from classical_controller_class import Classical_Controller
from report_results import report_results, report_spectral_decomposition_results


################## user settings ##############################
## number of control_level (i.e. d_c) in quantum circuit
control_levels = 4

## "unknown" unitary matrix of specified dimension (i.e. d)
unitary_dimension = 6
unitary =  Classical_Controller.random_unitary(unitary_dimension)

## classical controller settings
max_iterations = 500
max_consequative_failures = 20
C_target       = 1 - 10**-5
###############################################################



##############################################################################################################
################# DEMO 1: WHERE ONE EIGENVECTOR-EIGENPHASE PAIR SHOULD BE SOLVED FOR #########################
print('-- demo example 1 --')
print('-- single eigenvector-eigenvalue pair is retrieved from a random unitary')
# initialize the quantum circuit simulator
qCircuit = Quantum_Circuit(control_levels, unitary) 

# initialize the classical controller
cController = Classical_Controller(qCircuit.unitary_dimension)  # ini_ket and exclude_dir NOT SET

# run classically-controlled optimization
print('beginning variational optimization... ')

print('iteration # ...', 'C_max ...', 'elapsed time (seconds)')
C_max = 0
this_time = time.time()
while (cController.iterations < max_iterations) and (C_max < C_target) and (cController.consecutive_fail_count < max_consequative_failures):
    C_max = cController.run_single_optimization_loop(qCircuit)
    # status updates
    foo = cController.iterations / 25
    if int(foo) == foo:
        last_time = this_time
        this_time = time.time()
        print('  ', cController.iterations, ' ... ', C_max, ' ... ', this_time - last_time)
print('OPTIMIZATION CONCLUDED')
word = 'exceeded' if C_max >= C_target else 'did not meet'
print('  C_max =', C_max, word, 'C_target= ', C_target)
print('  total iterations = ', cController.iterations, '. A maximum of', max_iterations, 'were allowed')
print('  consecutive fail count =', cController.consecutive_fail_count, '. A maximum of', max_consequative_failures, 'were allowed.')
 
# optimization concluded, get best theta value for analysis (this will usually also give a small increase to C_max in the formal results)
cController.extra_fine_phase(qCircuit)

# report results to user and save results in dictionary
results_dict = report_results(cController, qCircuit)
print('-- end of demo 1 --')

print('Press enter to continue to demo 2')
x = input()

#########################################################################################
########## DEMO 2:  WHERE ALL EIGENVECTOR-EIGENPHASE PAIRS ARE FOUND ####################
# i.e. here we perform a full spectral decomposition
print(' ')
print(' ')
print('-- demo example 2 --')
print('-- full spectral decomposition')
print('   ... this is actually a continuation of demo 1, where we proceed to find all remaining eigenvector-eigenphases of the current unitary')

all_results = []
all_results.append(results_dict.copy())

# for unitary of dimension d, the optimization must be run (d-1) more times. The last eigenvector will be estimated by whatever is left
# Note that when performing a full reconstruction in this way, a very high C_target is unlikely to be reached for the latter eigenvectors
# of large matrices. You can overcome this by being more generous with the excluded directions than I was in the classical controller algorithm.
# It is  you choice if you'd like to develop something more sophisticated here. For example, I would recommend excluding previous eigenvectors early
# on in the classically-controlled optimization, then reverting to an optimization with no exclusions once a high value of C has been achieved.
for ii in range(1, qCircuit.unitary_dimension):
    print(' ')
    print('===========================================================')
    print('=== BEGIN search for eigenvector number', ii+1, 'out of', qCircuit.unitary_dimension, '===')
    # the quantum circuit simulator is already initialized

    # we now forbid searching in the direction of the estimated eigenvectors
    excluded_directions = all_results[0]["estimated_eigenvector"]
    for jj in range(1,len(all_results)):
        excluded_directions = np.append(excluded_directions, all_results[jj]["estimated_eigenvector"], axis=1)

    # re-initialize classical controller, now with excluded directions
    cController = Classical_Controller(qCircuit.unitary_dimension, exclude_dir = excluded_directions)
    
    # if this is the last eigenvector, we don not need to run any optimization as we take the only remaining direction
    if ii < (qCircuit.unitary_dimension - 1):
        # run classically-controlled optimization
        print('beginning variational optimization... ')
        print('iteration # ...', 'C_max ...', 'elapsed time (seconds)')
        C_max = 0
        this_time = time.time()
        while (cController.iterations < max_iterations) and (C_max < C_target) and (cController.consecutive_fail_count < max_consequative_failures):
            C_max = cController.run_single_optimization_loop(qCircuit)
            
            foo = cController.iterations / 25
            if int(foo) == foo:
                last_time = this_time
                this_time = time.time()
                print('  ', cController.iterations, ' ... ', C_max, ' ... ', this_time - last_time)
        print('OPTIMIZATION CONCLUDED')
        word = 'exceeded' if C_max >= C_target else 'did not meet'
        print('  C_max =', C_max, word, 'C_target= ', C_target)
        print('  total iterations = ', cController.iterations, '. A maximum of', max_iterations, 'were allowed')
        print('  consecutive fail count =', cController.consecutive_fail_count, '. A maximum of', max_consequative_failures, 'were allowed.')
    else:
        print('no classical optimization run for final eigenvector')
        
    # optimization concluded (/not run), get best theta value for analysis
    cController.extra_fine_phase(qCircuit)
    
    # report results to user and save results in dictionary
    results_dict = report_results(cController, qCircuit)

    # append results and continue to next loop
    all_results.append(results_dict.copy())
    print('=== Search for eigenvector number', ii+1, 'out of', qCircuit.unitary_dimension, 'CONCLUDED ===')
    print(' ')


print('===========================================================')
print('===========================================================')
# report overall results
fidelity, average_phase_error = report_spectral_decomposition_results(all_results, qCircuit.unitary)
 
print(' ')
print('-- end of demo 2 --')