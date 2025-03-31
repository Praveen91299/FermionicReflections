import numpy as np

def get_hf_occ(n_electrons, n_orbitals, spin_ord = 'udud', remove_qubit_loc = []):
    '''
    List slater determinant of HF
    '''
    hf = [1]*n_electrons + [0]*(2*n_orbitals - n_electrons)
    if spin_ord == 'uudd':
        hf = hf[::2] + hf[1::2]
    
    hf_f = []
    for i, a in enumerate(hf):
        if i not in remove_qubit_loc:
            hf_f.append(a)
    return hf_f

def get_hf_wfn(occ):
    wfn = [1.0]
    for i in occ:
        if i == 1:
            wfn = np.kron(wfn, [0, 1])
        else:
            wfn = np.kron(wfn, [1, 0])
    return wfn