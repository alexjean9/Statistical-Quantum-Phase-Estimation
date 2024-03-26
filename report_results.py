### Written by Alexandria Moore, Purdue University in March 2024. Contact: moore428@purdue.edu ###
import numpy as np

def report_results(cController, qCircuit):
    print(' ')
    print('-- OPTIMIZATION REPORT --')
    print(' C value:')
    print('   ', cController.current_qcircuit_return, '      [ideal value is 1]')
    
    print(' Estimated eigenphase in radians:')
    print('  ', cController.current_ephase)

    print(' Estimated eigenvector (ket transposed to a row):')
    print('  ', np.transpose(cController.current_ket))

    
    inner_prod, true_phase, e_index = qCircuit.nearest_true_eigenvector(cController.current_ket)
    print(' Absolute inner product of estimated eigenvector with nearest true eigenvector:')
    print('  ', inner_prod[0][0], '      [ideal value is 1]')
    
    ephase_error = np.abs(true_phase - cController.current_ephase)
    if ephase_error > np.pi:
        ephase_error = 2*np.pi - ephase_error
    print(' Phase error: absolute value of (true eigenphase - estimated eigenphase) in radians:')
    print('  ', ephase_error, '      [ideal value is 0]')

    print(' The optimizer converged to estimate of the eigenvector we have (arbitrarily) labeled #', e_index)

    result_dictionary = {
      "eigen_index"                  : e_index,
      "estimated_eigenvector"        : cController.current_ket,
      "estimated_eigenphase"         : cController.current_ephase,
      "C_final"                      : cController.current_qcircuit_return,
      "est_actual_eigenvec_product"  : inner_prod,
      "eigen_phase_absolute_error"   : ephase_error
    }
    return result_dictionary
    
def report_spectral_decomposition_results(all_results, qUnitary):
    print(' ')
    print('-- Individual eigenvector-eigenphase summary--')
    print('Search # ... eigenvector # ... inner product ... eigenphase error ....  C_max')
    for ii in range(0, len(all_results)):
        print('  ', \
            ii+1, ' ...       ', \
            all_results[ii]["eigen_index"], ' ...    ', \
            all_results[ii]["est_actual_eigenvec_product"][0][0], ' ... ', \
            all_results[ii]["eigen_phase_absolute_error"], ' ... ', \
            all_results[ii]["C_final"])

    print(' ')
    print('-- FULL SPECTRAL DECOMPOSITION REPORT --')
    # assume that all results is the same length as the dimensions of the unitary (it should be!!)
    d_u = len(all_results)
    # construct unitary_retrieved (Equation 6)
    Unitary_Retrieved = (0+0j)*np.zeros((d_u, d_u))
    for ii in range(0, d_u):
        ket = all_results[ii]["estimated_eigenvector"]
        bra = np.conj(np.transpose(ket))
        Unitary_Retrieved += np.matmul(ket, bra) * np.exp(1j * all_results[ii]["estimated_eigenphase"])

    M = np.matmul(np.conj(np.transpose(qUnitary)),  Unitary_Retrieved)
    fidelity = (np.trace(np.matmul(M, np.conj(np.transpose(M)))) + (np.abs(np.trace(M)))**2) / (d_u * (d_u +1))

    average_phase_error = 0
    for ii in range(0, d_u):
        average_phase_error += all_results[ii]["eigen_phase_absolute_error"]
    average_phase_error = average_phase_error / d_u

    print(' fidelity (Eq. 7): ')
    print('   ', np.abs(fidelity), '      [ideal value is 1')
    print(' average phase error (Eq. 8): ')
    print('   ', average_phase_error, '      [ideal value is 0]')
    return fidelity, average_phase_error