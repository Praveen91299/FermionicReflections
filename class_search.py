### Finding unique non-equivalent 3 body Cartan reflections

import numpy as np

def get_tuple(a):
    '''
    Returns tuple form represented by integer. Eg: 13 = '1101' = (0, 2, 3)
    '''
    i = 0
    t = []
    while a > 0:
        if a % 2 == 1:
            t = [i] + t
        a = a>>1
        i +=1
    
    return tuple(t)

def get_int_pm_array(i, n):
    t = get_tuple(i)

    a = np.ones(n)
    for idx in t:
        a[idx] = -1
    return a

def get_perm_matrix(perm):
    """
    Returns permutation matrix corresponding to the defined tuple
    """

    n=len(perm)
    M = np.zeros((n, n))

    for i, p in enumerate(perm):
        M[i, p] = 1
    return M

def construct_3_perm_matrices():
    """
    Returns all 3 variate permutation matrices

    """
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (2, 0, 1), (1, 2, 0), (2, 1, 0)]
    perm_M = [get_perm_matrix(perm) for perm in perms]

    P_array = []
    for M in perm_M:
        P = np.zeros((8, 8))
        P[0, 0] = 1
        P[7, 7] = 1
        P[1:4, 1:4] = M
        P[4:7, 4:7] = M
        P_array.append(P)
    return P_array

def check_equiv(a, b):
    #permutations
    P_list = construct_3_perm_matrices()

    b_perms = [P@b for P in P_list]

    for bp in b_perms:
        if all(a == bp):
            return True

    return False

N = 3
vals = [0, 1]
A = []

#create case matrix
for n1 in vals:
    for n2 in vals:
        for n3 in vals:
            A.append([1, n1, n2, n3, n2*n3, n1*n3, n1*n2, n1*n2*n3])

A = np.array(A)
A_inv = np.linalg.inv(A)

n_eqn = len(A)
n_cases = 2**n_eqn
bs = [get_int_pm_array(i, n_eqn) for i in range(n_cases)]

coeffs = [A_inv @ b for b in bs]
coeffs = [a[0]*a for a in coeffs]

#get unique cases #
unique_coeffs = []

for c in coeffs:
    flag = True
    for uc in unique_coeffs:
        if check_equiv(c, uc):
            flag = False
    if flag:
        unique_coeffs.append(c)

traces = []
tb_uc = []
for uc in unique_coeffs:
    if uc[-1] == 0:
        trace = np.sum(A@uc)
        print(uc, trace)
        traces.append(trace)
        tb_uc.append(uc)

print(len(tb_uc))

from openfermion import FermionOperator
#create fermionic operators and find trace
n = lambda i: FermionOperator('{}^ {}'.format(i, i), 1.0)
op_basis = [1.0, n(0), n(1), n(2), n(1)*n(2), n(2)*n(0), n(0)*n(1), n(0)*n(1)*n(2)]
ops = []

for _, tb in enumerate(tb_uc):
    o = sum([op_basis[i] * tb[i] for i in range(len(op_basis))])
    ops.append(o)
# todo frane a problem of 1 norm reduction here!

def get_trace_vec(op, n_qubits):
    """
    Get traces of representations of diagonal reflections

    Makes use of the fact that the irreps are invariant with qubit relabelling.

    """
    diag = np.diag(get_sparse_operator(op, n_qubits).toarray())

    v = np.zeros(n_qubits + 1)

    for i in range(0, 1<<n_qubits):
        t = get_tuple(i)
        hamming_wt = len(t)

        v[hamming_wt] += diag[i]

    return v

trace_vectors = []
for op in ops:
    v = get_trace_vec(op, 3)
    print(v)

    trace_vectors.append(v)