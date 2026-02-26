### bch closure tests

import numpy as np
from openfermion import FermionOperator, commutator, normal_ordered, jordan_wigner, QubitOperator, count_qubits, get_sparse_operator

tol = 1e-10
tol_dec = 10

from utils.mat_utils import is_hermitian

is_proj = lambda p: np.isclose(np.linalg.norm(p @ p - p, 'fro'), 0, atol=tol)
def proj(v):
    """
    Construct projector with given vector v

    """
    p  = (np.array([v], complex).T @ np.array([v], complex).conjugate()) / (np.linalg.norm(v, 2))
    assert is_proj(p), f"Not a projector!"

    return p

def construct_projectors(G, n_qubits, tol = 1e-10):
    """
    Construct projectors to the distinct eigenspaces of hermitian G
    
    """

    tol_dec = - int(np.log10(tol))

    assert is_hermitian(get_sparse_operator(G, n_qubits)), "Operator not hermitian! Hermiticity required for constructing orthogonal projectors."

    eigenvalues, eigenvectors = np.linalg.eigh(get_sparse_operator(G, n_qubits).toarray())

    projectors = []
    eig_unique = list(set(np.around(eigenvalues, tol_dec)))
    e_vecs = []

    for i, e_val in enumerate(eig_unique):
        e_vec = []
        projector = np.zeros((1<<n_qubits, 1<<n_qubits), complex)

        for j, u in enumerate(eigenvalues):
            if np.isclose(e_val, u, atol=tol):
                p = proj(eigenvectors[:, j])
                projector += p
                e_vec.append(eigenvectors[:, j])
        
        assert is_proj(projector), f"Not a projector: e_val = {e_val}"
        e_vecs.append(e_vec)
        projectors.append(projector)
    
    ### verifying completeness of projectors
    assert np.isclose(np.linalg.norm(sum(projectors) - np.identity(1<<n_qubits), 'fro'), 0, atol=tol), f"{np.linalg.norm(sum(projectors) - np.identity(1<<n_qubits))}"
    
    for i, p in enumerate(projectors):
        assert (np.isclose(np.linalg.norm(get_sparse_operator(G, n_qubits) @ p - eig_unique[i]*p, 'fro'), 0, atol=tol)), "Projector yielding incorrect magnitude of eigen value."

    return projectors, eig_unique, e_vecs

def get_block_norms(H, projectors, silent = True, tol=1e-10):
    """
    Prints block frobenius norms and returns index pairs of non-zero blocks

    """
    tol_dec = -int(np.log10(tol))

    n_blocks = len(projectors)
    blocks = []
    block_norms = np.zeros((n_blocks, n_blocks))

    for i, pi in enumerate(projectors):
        for j, pj in enumerate(projectors):
            #print("e1: {}, e2: {}, diff {}".format(e_vals[i], e_vals[j], e_vals[i] - e_vals[j]))

            b = pi @ H @ pj
            blocks.append(b)

            fro_norm = np.round(np.linalg.norm(b, 'fro'), tol_dec)
            block_norms[i, j] = fro_norm

            if not silent: print(f"Block ({i}, {j}): {fro_norm}")
    
    return blocks, block_norms

def get_commutators(G, h, deg = 0):
    """
    Get upto deg commutators

    """
    commutators = [h]
    for i in range(deg):
        commutators.append(commutator(G, commutators[-1]))
    
    return commutators

from itertools import combinations

def get_poly_coeff(roots):
    """
    Get binomial coeffs of polynomial with given roots
    
    """

    d = len(roots)
    coeffs = np.zeros(d+1, complex)
    
    for i in range(d+1):
        root_combs = combinations(roots, i)

        coeffs[i] = ((-1)**i) *  np.sum([np.prod(c, dtype=complex) for c in root_combs])
    return reversed(coeffs)

Kia     = lambda i, a: FermionOperator(f'{a}^ {i}', 1.0) - FermionOperator(f'{i}^ {a}', 1.0)
Kjaib   = lambda j, a, i, b: FermionOperator(f'{a}^ {b}^ {i} {j}', 1.0) - FermionOperator(f'{j}^ {i}^ {b} {a}', 1.0)

Tia     = lambda i, a: Kia(2*i, 2*a) + Kia(2*i + 1, 2*a + 1)
Tiiaa   = lambda i, a: Kjaib(2*i, 2*a, 2*i+1, 2*a+1)
Tiiab   = lambda i, a, b: Kjaib(2*i, 2*a, 2*i+1, 2*b+1) + Kjaib(2*i, 2*a + 1, 2*i+1, 2*b)
Tijaa   = lambda i, j, a: Kjaib(2*i, 2*a, 2*j+1, 2*a+1) + Kjaib(2*i+1, 2*a, 2*j, 2*a+1)

n_qubits = 6
G = 1.j * (Tiiab(0, 1, 2))
print("G: ", G)
G = jordan_wigner(G)

projectors, e_vals, e_vecs = construct_projectors(G, n_qubits, tol=tol)

H = jordan_wigner(FermionOperator('2^ 3^ 1 0', 1.0))# + jordan_wigner(FermionOperator('2^ 5', 1.0)) + jordan_wigner(FermionOperator('3^ 1', 1.0) + FermionOperator('0^ 2', 1.0) + FermionOperator('1^ 3', 1.0))

Hs   =   get_sparse_operator(H, n_qubits).toarray()
bl, bl_norms = get_block_norms(Hs, projectors)

print("Block norms:\n", bl_norms)

eval_diff = np.zeros(np.shape(bl_norms), complex)

nz_block_eig_pos = []
all_differences = []
for i, v1 in enumerate(e_vals):
    for j, v2 in enumerate(e_vals):
        eval_diff[i, j] = v1 - v2
        all_differences.append(v1 - v2)

        if bl_norms[i, j] > 1e-5:
            nz_block_eig_pos.append((i, j))
print("Eigen values:\n", np.around(e_vals, 5))
print("Eigen value differences:\n", np.around(eval_diff, tol_dec))

### # of eigen value differences in non-zero blocks
nz_eig = []
for pos in nz_block_eig_pos:
    nz_eig.append(eval_diff[pos])

nz_eig = list(set(nz_eig))
max_eig_diff_count = len(set(np.around(all_differences, tol_dec)))
print("Found {} unique eigenvalue differences over non-vanishing blocks:\n(maximum possible differences: {}) \n{}".format(len(nz_eig), max_eig_diff_count, nz_eig))

d = len(nz_eig)

terms = get_commutators(G, H, d)
coeff = get_poly_coeff(nz_eig)

fh = np.sum([c*t for c, t in zip(coeff, terms)])
print(f'f(H) = {fh}')