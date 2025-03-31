from openfermion import FermionOperator
import numpy as np

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

    tbt_new = np.einsum('ijkl,pi,qj,rk,sl->pqrs', tbt, U, U_conj, U, U_conj)

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