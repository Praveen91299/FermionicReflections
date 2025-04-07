###############################################################################
# FULL UPDATED CODE: Reflection Transformation with Pre-Transformation Spectrum
###############################################################################

from utils.pickle_utils import get_pkl_object
from ansatz import FermionicReflection, ReflectionAnsatz, get_poly
from openfermion import (
    count_qubits,
    get_sparse_operator,
    FermionOperator,
    normal_ordered
)
from openfermion.utils import commutator
from utils.hf_utils import get_hf_occ, get_hf_wfn
import numpy as np
import pickle
from scipy.sparse.linalg import eigsh


def reflection_transform(H, R, tau):
    """
    Apply the transformation:
      H_tilde = H 
                - i * (sin(tau)/2) * [H, R]
                + (1 - cos(tau))/2 * (R H R - H).
    
    Args:
        H   (FermionOperator): Original Hamiltonian
        R   (FermionOperator): Reflection operator
        tau (float):          Reflection angle/parameter

    Returns:
        FermionOperator: The transformed Hamiltonian H_tilde.
    """
    sin_t = np.sin(tau)
    cos_t = np.cos(tau)
    
    # Commutator [H, R] = H*R - R*H
    comm_HR = commutator(H, R)

    # R*H*R
    RHR = R * H * R

    # Combine terms
    H_tilde = (
        H
        - 1j * (sin_t / 2.0) * comm_HR
        + ((1.0 - cos_t) / 2.0) * (RHR - H)
    )

    # Normal order
    H_tilde = normal_ordered(H_tilde)
    return H_tilde


###############################################################################
# 1) Load your Hamiltonian H
###############################################################################
mol = 'h4'
spin_ord = 'udud'
filename = f'{mol}{spin_ord}_fer.bin'
directory = './saved/hamiltonians/'

# Load the fermionic Hamiltonian H from disk
H = get_pkl_object(filename=filename, directory=directory)

n_qubits = count_qubits(H)
n_electrons = n_qubits // 2

# Convert to sparse for quick checks
Hs = get_sparse_operator(H, n_qubits)

# Build HF reference wavefunction
occ = get_hf_occ(n_electrons, n_qubits // 2, spin_ord)
hf_wfn = get_hf_wfn(occ)

print(f"Number of qubits = {n_qubits}")
print(f"Number of electrons = {n_electrons}")
print("\nFermionic Hamiltonian H loaded.\n")




###############################################################################
# 3) Build Reflection Operators
###############################################################################
qubit_pairs = []
for i in [3, 4, 5, 6, 7]:
    qubit_pairs.append((0, i, "imag"))
    qubit_pairs.append((1, i, "imag"))
    qubit_pairs.append((2, i, "imag"))
    qubit_pairs.append((3, i, "imag"))

kinds = ['p11', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29']
indices = [
    [0],
    [0, 1], [0, 1], [0, 1], [0, 1],
    [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
    [0, 1, 2, 3]
]
R_list = []

# Build the polynomials, reflection operators, and optimize them
for i, kind in enumerate(kinds):
    p = get_poly(kind, indices[i])
    print('\nPolynomial kind =', kind)
    fr = FermionicReflection(n_qubits, p, V_type='restricted', qubit_pairs=qubit_pairs)
    fr.optimize_grad(Hs, hf_wfn, init_param_type='random')
    R_list.append(fr)

# Choose the reflection operator with the largest gradient
best_fr = max(R_list, key=lambda fr: fr.max_grad)
print("\nSelected generator with max gradient = {:.6f}".format(best_fr.max_grad))
###############################################################################
# 4) Create a ReflectionAnsatz and optimize
###############################################################################
# Example: use only reflection #4 in the list [4:5]
RA = ReflectionAnsatz(n_qubits=n_qubits, ref_state=hf_wfn, reflections=R_list[4:5])

# Print initial energy
init_energy = RA.energy(Hs)
print(f"\nInitial energy from ReflectionAnsatz with reflection #4: {init_energy:.6f}\n")

# Optimize reflection ansatz
init_params = np.random.rand(len(R_list[4:5]))
RA.optimize_energy(Hs, init_params)

final_energy = RA.energy(Hs)
print(f"Final energy after optimization: {final_energy:.6f}")
print("Final taus:", RA.taus)

# This reflection operator
R_op = RA.reflections[0].get_R(sparse=False)
tau_value = RA.taus[0]
##print("\nReflection Operator R_op:\n", R_op)
print(f"\nAssociated tau_value = {tau_value:.6f}")

###############################################################################
# 5) Transform the Hamiltonian using the reflection_transform(...)
###############################################################################
H_tilde = reflection_transform(H, R_op, tau_value)
##print("\n--- Transformed Hamiltonian (H_tilde) ---")
##print(H_tilde)

# (Optional) Save H_tilde to a file
with open('hamiltonian_reflection_transformed.pkl', 'wb') as f:
    pickle.dump(H_tilde, f)
    
###############################################################################
# 2) Print the spectrum (lowest eigenvalues) of the original Hamiltonian H
###############################################################################
n_eigs_h = min(32, Hs.shape[0] - 2)
eigvals_H, _ = eigsh(Hs, k=n_eigs_h, which='SA')
eigvals_H = np.sort(eigvals_H)

print("\n=== Eigenvalues of the Original Hamiltonian (H) ===")
for i, val in enumerate(eigvals_H):
    print(f"  Eigenvalue {i+1}: {val:.6f} Hartree")

###############################################################################
# 6) Diagonalize H_tilde and print eigenvalues
###############################################################################
H_tilde_sparse = get_sparse_operator(H_tilde, n_qubits)
n_eigs_tilde = min(32, H_tilde_sparse.shape[0] - 2)

eigvals_tilde, _ = eigsh(H_tilde_sparse, k=n_eigs_tilde, which='SA')
eigvals_tilde = np.sort(eigvals_tilde)

print("\n=== Eigenvalues of the Transformed Hamiltonian (H_tilde) ===")
for i, val in enumerate(eigvals_tilde):
    print(f"  Eigenvalue {i+1}: {val:.6f} Hartree")

