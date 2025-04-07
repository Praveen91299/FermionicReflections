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

from openfermion import FermionOperator, get_sparse_operator
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

from ansatz import *
from openfermion import FermionOperator, get_sparse_operator
import scipy

def list_number_operator(n_list):
    op = FermionOperator('', 0.0)
    for n in n_list:
        op += FermionOperator('{}^ {}'.format(n, n), 1.0)
    return op

def number_projector(e_val, n_qubits, n_list = [], sparse = True):
    """
    e_val: target number
    n_list: positions to consider
    n_qubits: total number of qubits
    """
    if n_list == []:
        n_list = list(range(n_qubits))
    
    ev_list = list(range(len(n_list)+1))
    
    if e_val not in ev_list:
        print('Warning: Number operator eigen value unphysical.')
    n = list_number_operator(n_list)#sp(sum([n(i) for i in range(n_qubit)]))
    #if sparse: n = get_sparse_operator(n, n_qubit)
    return lagrange_proj(n, e_val, ev_list, n_qubits, sparse)

def lagrange_proj(op, ev, ev_list, n_qubit, sparse):
    def sp(op):
        return get_sparse_operator(op, n_qubit) if sparse == True else op
    if ev not in ev_list:
        print('Warning: Ev not in ev list!')
    else:
        ev_list.remove(ev)
        
    f = 1.0
    p = sp(FermionOperator('', 1.0))
    if sparse:
        iden = scipy.sparse.csc_matrix(np.identity(2**n_qubit))
        op = sp(op)
    else:
        iden = 1.0
    for e in ev_list:
        f *= (ev - e)
        p *= (op - e*iden)
    return p/f


kinds = ['p11', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29']
indices = [
    [0],
    [0, 1], [0, 1], [0, 1], [0, 1],
    [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
    [0, 1, 2, 3]
           ]

n_qubits = 4

def get_FixedBody_trace(op, n_qubits):
    """
    Fixed body operator parts' traces
    
    """
    op_list = [0]*(n_qubits+1)
    for k, v in zip(op.terms.keys(), op.terms.values()):
        op_list[len(k)//2] += FermionOperator(k, v)

    tr_list = []

    for o in op_list:
        if o is 0:
            tr_list.append(0)
        if o is not 0:
            tr_list.append(np.int(get_sparse_operator(o, n_qubits).trace()))
    return tr_list

def get_irrep_trace(op, n_qubits):
    """
    Project onto Fock spaces and get traces

    """

    traces = []
    sparse_op = get_sparse_operator(op, n_qubits)
    for i in range(n_qubits + 1):
        proj = number_projector(i, n_qubits, sparse=True)

        sparse_proj = sparse_op * proj

        trace = sparse_proj.trace()
        traces.append(int(trace))

    return traces

#trace
for i, kind in enumerate(kinds):
    p = get_poly(kind, indices[i])
    sparse_p = get_sparse_operator(p, n_qubits)
    trace = int(sparse_p.trace())

    print('\n\n', kind)
    print(trace)
    print(get_FixedBody_trace(p, n_qubits))
    print(get_irrep_trace(p, n_qubits))
