from openfermion import FermionOperator, jordan_wigner, get_sparse_operator
import numpy as np
from scipy.sparse import csc_matrix, issparse
from scipy.sparse.linalg import expm as sparse_expm
from opt_einsum import contract

def chem_ferm_to_chem_tbt(op: FermionOperator, n_qubits, tol = 1e-5):
    tbt = np.zeros((n_qubits, n_qubits, n_qubits, n_qubits), complex)

    constant =  op.constant
    op = op - constant

    for key, coeff in zip(op.terms.keys(), op.terms.values()):

        if abs(coeff) < tol:
            continue
        
        assert len(key) == 4, "Not a two body operator"
        assert key[0][1] == 1 and key[1][1] == 0 and key[2][1] == 1 and key[3][1] == 0, "Operator not in chem two body ordering"

        tbt[key[0][0], key[1][0], key[2][0], key[3][0]] = coeff

    return tbt, constant

def rotate_chem_tbt(tbt, U):

    x, y = np.shape(U)
    U_conj = np.conjugate(U)

    a, b, c, d = np.shape(tbt)
    tbt_new = np.zeros(np.shape(tbt), complex)

    assert y == a and y == b and y == c and y == d, "Incompatible rotation matrix."

    tbt_new = contract('pi,qj,ijkl,rk,sl->pqrs', U, U_conj, tbt, U, U_conj)

    return tbt_new

def chem_tbt_to_chem_ferm(tbt):

    op = FermionOperator()
    a, b, c, d = np.shape(tbt)

    for i in range(a):
        for j in range(b):
            for k in range(c):
                for l in range(d):
                    op += FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l), tbt[i, j, k, l])

    return op

def promote_cartan_twobody(op):
    """
    Check if cartan operator, and promote 1 body to 2 body terms

    """
    op_new = FermionOperator()
    op_new += op.constant
    for key, coeff in zip(op.terms.keys(), op.terms.values()):
        
        #check tbt
        if len(key) == 2:
            assert key[0][1] == 1 and key[1][1] == 0, "Operator not in Chem one body"
            assert key[0][0] == key[1][0], "Operator not diagonal"

            op_new += FermionOperator('{}^ {} {}^ {}'.format(key[0][0], key[0][0], key[0][0], key[0][0]), coeff)

        elif len(key) == 4:
            #check chem
            assert key[0][1] == 1 and key[1][1] == 0 and key[2][1] == 1 and key[3][1] == 0, "Operator not in chem two body ordering"
            #check cartan
            assert key[0][0] == key[1][0] and key[2][0] == key[3][0], "Operator not diagonal"

            op_new += FermionOperator('{}^ {} {}^ {}'.format(key[0][0], key[0][0], key[2][0], key[2][0]), coeff)
    
    return op_new

def build_sparse_basis(n_qubits):
    """
    Builds dictionary of sparse versions of ferm op a_i^ a_j a_k^ a_l

    Uses Jordan-Wigner transform by default
    
    """
    basis_dict = {}

    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):
                    basis_dict[(i, j, k, l)] = get_sparse_operator(jordan_wigner(FermionOperator('{}^ {} {}^ {}'.format(i, j, k, l), 1.0)), n_qubits)
    
    print("...Built sparse basis for {} qubits".format(n_qubits))
    return basis_dict

def get_sparse_fermop(tbt, basis_dict):
    """
    Construct sparse operator represented by chemist ordered tensor, tbt using premade sparse operator tensor

    """

    n_qubits = len(tbt)
    op = csc_matrix((2**n_qubits, 2**n_qubits))

    for i in range(n_qubits):
        for j in range(n_qubits):
            for k in range(n_qubits):
                for l in range(n_qubits):
                    op += basis_dict[(i, j, k, l)]*tbt[i, j, k, l]
    
    return op

def return_sparse(op, n_qubits):
    """
    Returns sparse operator, does nothing if already sparse

    """
    if issparse(op):
        return op
    else:
        get_sparse_operator(op, n_qubits)

Epq = lambda p, q: FermionOperator('{}^ {}'.format(p, q), 1.0)
g_pq_real = lambda p, q: Epq(p, q) - Epq(q, p)
g_pq_imag = lambda p, q: 1.j*(Epq(p, q) + Epq(q, p))

def get_U(mat, n_qubits):
    """
    Get the 2^N x 2^N unitary corresponding to the N x N matrix representation of the U(N) algebra

    """

    assert np.shape(mat) == (n_qubits, n_qubits)

    op = FermionOperator('', 0)
    for i in range(n_qubits):
        for j in range(n_qubits):
            op += mat[i, j]*Epq(i, j)
    
    s_op = get_sparse_operator(op, n_qubits)
    return sparse_expm(s_op)