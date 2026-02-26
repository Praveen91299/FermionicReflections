# h4 minimal generators for chemical accuracy, for Reza

import numpy as np

from Refl.orbital_rotation import FullOrbitalRotation, RestrictedOrbitalRotation
from Refl.reflection_generator import FermionicReflection, get_poly
from Refl.reflection_ansatz import ReflectionAnsatz
import numpy as np

from utils.hf_utils import *
from utils.ferm_utils import build_sparse_basis
from openfermionpyscf import run_pyscf
from openfermion import get_fermion_operator, count_qubits, get_sparse_operator, MolecularData

#prep
n_hydrogens = 4
bond_length = 1
basis = 'sto-3g'
multiplicity = 1
spin_ord = 'udud'

molecule = MolecularData(
    geometry=[('H', (0, 0, i * bond_length)) for i in range(n_hydrogens)],
    charge=0,
    basis=basis,
    multiplicity=multiplicity,
    description=f"linear_r-{bond_length}")

molecule = run_pyscf(molecule, run_scf=True, run_cisd=True, run_fci=True)
print("PyScf Calculation complete. Hartree-Fock energy:", molecule.hf_energy, 
    "\nFCI energy:", molecule.fci_energy)

mh = molecule.get_molecular_hamiltonian()
H = get_fermion_operator(mh)


n_qubits = count_qubits(H)
Hs = get_sparse_operator(H, n_qubits)
n_electrons = n_qubits//2

Hs = get_sparse_operator(H, n_qubits)
occ = get_hf_occ(n_electrons, n_qubits//2, spin_ord)
hf_wfn = get_hf_wfn(occ)

basis_dict = build_sparse_basis(n_qubits)

# make ansatz
V1 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(3, 4, "imag")])
V2 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(2, 5, "real")])

V3 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(1, 5, "imag")])
V4 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(0, 4, "real")])

V5 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(3, 6, "imag")])
V6 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(2, 7, "real")])

V7 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(0, 7, "imag")])
V8 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(3, 4, "real")])

V9 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(1, 6, "imag")])
V10 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(2, 5, "real")])

V11 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(0, 7, "imag")])
V12 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(1, 6, "real")])

V13 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(0, 5, "imag")])
V14 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(3, 6, "real")])

V15 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(1, 7, "imag")])
V16 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(2, 4, "real")])

V17 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(0, 6, "imag")])
V18 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(2, 4, "real")])

V19 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(1, 7, "imag")])
V20 = RestrictedOrbitalRotation(n_qubits, params = [np.pi/4], qubit_pairs = [(3, 5, "real")])

FR = FermionicReflection(n_qubits, get_poly('p24', [2, 3]), [V1, V2], basis_dict=basis_dict)
FR2 = FermionicReflection(n_qubits, get_poly('p24', [0, 1]), [V3, V4], basis_dict=basis_dict)
FR3 = FermionicReflection(n_qubits, get_poly('p24', [2, 3]), [V5, V6], basis_dict=basis_dict)
FR4 = FermionicReflection(n_qubits, get_poly('p24', [0, 3]), [V7, V8], basis_dict=basis_dict)
FR5 = FermionicReflection(n_qubits, get_poly('p24', [1, 2]), [V9, V10], basis_dict=basis_dict)
FR6 = FermionicReflection(n_qubits, get_poly('p24', [0, 1]), [V11, V12], basis_dict=basis_dict)
FR7 = FermionicReflection(n_qubits, get_poly('p24', [0, 3]), [V13, V14], basis_dict=basis_dict)
FR8 = FermionicReflection(n_qubits, get_poly('p24', [1, 2]), [V15, V16], basis_dict=basis_dict)
FR9 = FermionicReflection(n_qubits, get_poly('p24', [0, 2]), [V17, V18], basis_dict=basis_dict)
FR10 = FermionicReflection(n_qubits, get_poly('p24', [1, 3]), [V19, V20], basis_dict=basis_dict)

FR.optimize_grad(Hs, hf_wfn, 1)
FR2.optimize_grad(Hs, hf_wfn, 1)
FR3.optimize_grad(Hs, hf_wfn, 1)
FR4.optimize_grad(Hs, hf_wfn, 1)
FR5.optimize_grad(Hs, hf_wfn, 1)
FR6.optimize_grad(Hs, hf_wfn, 1)
FR7.optimize_grad(Hs, hf_wfn, 1)
FR8.optimize_grad(Hs, hf_wfn, 1)
FR9.optimize_grad(Hs, hf_wfn, 1)
FR10.optimize_grad(Hs, hf_wfn, 1)

# optimize reflection taus
Refl = ReflectionAnsatz(n_qubits, hf_wfn, [FR, FR2, FR3, FR4, FR5, FR6, FR7, FR8, FR9, FR10])
Refl.optimize_energy(Hs, n_random=1)

### dressing Pauli counts

FRs = [FR, FR2, FR3, FR4, FR5, FR6, FR7, FR8, FR9, FR10]
pauli_counts = []

for i in range(len(FRs)):
    FR_l = FRs[:i+1]

    Refl = ReflectionAnsatz(n_qubits, hf_wfn, FR_l)
    Refl.optimize_energy(Hs, True, 1)
    Hd = Refl.dress_hamiltonian(H, operator_type="QubitOperator")
    n_paulis = len(Hd.terms.keys())

    pauli_counts.append(n_paulis)
    print("\n# Reflections: {}, # Paulis: {}\n".format(i+1, n_paulis))