import numpy as np
import scipy.sparse as sp

def is_close_to_identity(A, tol=1e-6):
    if not sp.issparse(A):
        raise ValueError("Input matrix must be sparse.")

    identity = sp.eye(A.shape[0], format=A.format)  # Create sparse identity matrix
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
    if not sp.issparse(A):
        raise ValueError("Input matrix must be a sparse matrix.")
    
    # Check if square
    if A.shape[0] != A.shape[1]:
        return False  # Non-square matrices cannot be Hermitian
    
    # Compute difference between A and its conjugate transpose
    diff = A - A.getH()  # A.getH() is equivalent to A.conj().T for sparse matrices
    
    # Check if the maximum absolute entry in diff is within tolerance
    return np.abs(diff).max() < tol