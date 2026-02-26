import numpy as np
import scipy.sparse as spr
from openfermion import QubitOperator

def is_close_to_identity(A, tol=1e-6):
    if not spr.issparse(A):
        raise ValueError("Input matrix must be sparse.")

    identity = spr.eye(A.shape[0], format=A.format)  # Create sparse identity matrix
    diff = A - identity  # Compute difference
    max_diff = np.abs(diff).max()  # Maximum absolute entry

    return max_diff < tol  # Check if within tolerance

def is_hermitian(A, tol=1e-10):
    """
    Checks if a sparse matrix A is Hermitian (A = A.H).
    
    Parameters:
        A (scipy.sparse matrix): Input sparse matrix.
        tol (float): Tolerance for numerical errors.
        
    Returns:
        bool: True if A is Hermitian, False otherwise.
    """
    if not spr.issparse(A):
        #print("Hermiticity check: Matrix not sparse, converting to sparse to check hermiticity.")
        A = spr.csc_matrix(A, dtype=complex)
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        print("Hermiticity check: Matrix not square.")
        return False  # Non-square matrices cannot be Hermitian
    
    # Compute difference between A and its conjugate transpose
    diff = A - A.getH()  # A.getH() is equivalent to A.conj().T for sparse matrices
    
    # Check if the maximum absolute entry in diff is within tolerance
    return np.abs(diff).max() < tol

def is_antihermitian(A, tol=1e-10):
    if not spr.issparse(A):
        #print("Anti-Hermiticity check: Matrix not sparse, converting to sparse to check anti-hermiticity.")
        A = spr.csc_matrix(A, dtype=complex)
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        print("Anti hermiticity check: Matrix not square.")
        return False  # Non-square matrices cannot be Hermitian

    return is_hermitian(1.j*A)

def is_unitary(U, tol=1e-10):
    """
    Checks if a sparse matrix is unitary (U @ U.getH() = I = U.getH() @ U)

    """
    if not spr.issparse(U):
        #print("Unitary check: Matrix is not sparse, converting to sparse to check unitarity.")
        U = spr.csc_matrix(U)
    
    if U.shape[0] != U.shape[1]:
        print("Unitary check: Matrix not square.")
        return False
    
    U_dag = U.getH()
    return is_close_to_identity(U @ U_dag, tol) and is_close_to_identity(U_dag @ U, tol)

def is_reflection(R, tol=1e-10):
    """
    Checks if a sparse matrix is a reflection. Equivalent to being unitary AND hermitian
    
    """
    if not spr.issparse(R):
        #print("Relfection check: Matrix is not sparse, converting to sparse to check unitarity.")
        R = spr.csc_matrix(R)
    
    if R.shape[0] != R.shape[1]:
        print("Reflection check: Matrix not square.")
        return False
    
    R2 = R@R
    return is_close_to_identity(R2, tol) and is_hermitian(R)

def pad_2d_to_square(arr, n):
    """
    Pad matrix with extra rows and columns of zeros
    
    """
    rows, cols = arr.shape
    pad_rows = n - rows
    pad_cols = n - cols
    return np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)

def truncate_pauli_hamiltonian(HQ: QubitOperator, tol = 1e-5, n_terms = None, verbose=True):
    """
    Truncate a Pauli Hamiltonian upto tolerance

    """

    HQ_new = QubitOperator()
    if verbose: print("Original term count: {}".format(len(list(HQ.terms.keys()))))
    if n_terms is not None:
        #Truncate to number of terms

        paired_list = zip(HQ.terms.keys(), HQ.terms.values())
        #sort terms
        paired_list_sorted = sorted(paired_list, key=lambda x: abs(x[1]), reverse=True)

        n = min([len(paired_list_sorted), n_terms])
        for pair in paired_list_sorted[:n]:
            HQ_new += QubitOperator(pair[0], pair[1])

    else:
        #truncate to coefficient
        for key, coeff in zip(HQ.terms.keys(), HQ.terms.values()):
            if abs(coeff) >= tol:
                HQ_new += QubitOperator(key, coeff)
    
    if verbose: print("Truncated term count: {}".format(len(list(HQ_new.terms.keys()))))
    return HQ_new

def ensure_real(HQ, tol=1e-5):
    """
    Ensures the coefficients of Pauli products in HQ are real valued (for hermiticity)
    if magnitude of imaginary part >=tol, prints a warning

    """

    HQ_new = QubitOperator()
    for term, coeff in zip(HQ.terms.keys(), HQ.terms.values()):
        if abs(np.imag(coeff)) >= tol: print("Warning (ensure_real): Truncating significant imaginary term in operator.")
        HQ_new += QubitOperator(term, np.real(coeff))
    
    return HQ_new