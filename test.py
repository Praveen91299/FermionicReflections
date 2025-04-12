### select generators
from openfermion import commutator, expectation, double_commutator
from scipy.sparse import issparse
import scipy
from utils.pickle_utils import *
from ansatz import *
from openfermion import count_qubits, get_sparse_operator, FermionOperator, get_fermion_operator, MolecularData
from utils.hf_utils import *
from utils.ferm_utils import *
from utils.misc_utils import line
from optimparallel import minimize_parallel
from openfermionpyscf import run_pyscf

if __name__ == '__main__':

    mol = 'h4'
    spin_ord = 'udud'

    n_hydrogens = 4
    bond_length = 1
    basis = 'sto-3g'
    multiplicity = 1

    molecule = MolecularData(
        geometry=[('H', (0, 0, i * bond_length)) for i in range(n_hydrogens)],
        charge=0,
        basis=basis,
        multiplicity=multiplicity,
        description=f"linear_r-{bond_length}")

    line(40)
    print("Molecule: {}\nBond length: {}\nBasis: {}\nMultiplicity: {}".format(mol, bond_length, basis, multiplicity))
    molecule = run_pyscf(molecule, run_scf=True, run_cisd=True, run_fci=True)
    print("PyScf Calculation complete. Hartree-Fock energy:", molecule.hf_energy, 
        "\nCISD energy:", molecule.fci_energy)

    mh = molecule.get_molecular_hamiltonian()
    H = get_fermion_operator(mh)

    n_qubits = count_qubits(H)
    n_electrons = n_qubits//2

    Hs = get_sparse_operator(H, n_qubits)
    occ = get_hf_occ(n_electrons, n_qubits//2, spin_ord)
    hf_wfn = get_hf_wfn(occ)
    basis_dict = build_sparse_basis(n_qubits)

    p = get_poly('p29', [3, 4, 5, 6])
    qubit_pairs, params = iterative_V_construction(8, H, p, 10, hf_wfn, basis_dict=basis_dict, n_random=5, optimization="global", tol=1e-3)

    R = FermionicReflection(n_qubits, p, "iterative", params_init=params, qubit_pairs = qubit_pairs, basis_dict = basis_dict)
    # R = FermionicReflection.build_R_iterative(params, 8, p, qubit_pairs, False, basis_dict)
    fr = ReflectionAnsatz(n_qubits, hf_wfn, [R])

    fr.optimize_energy(Hs)

    fr.dress_hamiltonian()
