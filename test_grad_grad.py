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
import pandas as pd

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

    
    poly_types = ['p11', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29']
    indices = [
        [3],
        [3, 4], [3, 4], [3, 4], [3, 4],
        [2, 3, 4], [2, 3, 4], [2, 3, 4], [2, 3, 4],
        [2, 3, 4, 5]
            ]
    n_l = 100

    energies = []
    gradients = []
    n_generators = [] 

    Hs_new = deepcopy(Hs)
    Refl_list = []

    for i, poly_type in enumerate(poly_types):
        p = get_poly(poly_type, indices[i])
        print('\n\nReflection {} of {} over {} indices.'.format(i+1, poly_type, indices[i]))

        #find Refl/orbital rotation with respect to new dressed Hamiltonian
        qubit_pairs, params = iterative_V_construction(n_qubits, Hs_new, p, n_l, hf_wfn, basis_dict=basis_dict, n_random=1, optimization="global", tol=1e-3)

        R = FermionicReflection(n_qubits, p, "iterative", params_init=params, qubit_pairs = qubit_pairs, basis_dict = basis_dict)
        Refl_list.append(R)

        #optimize energy with all reflections so far.
        fr = ReflectionAnsatz(n_qubits, hf_wfn, [R])

        fr.optimize_energy(Hs)
        e = fr.energy(Hs)
        g = R.get_gradient(Hs, hf_wfn)

        #Hs_new = fr.dress_hamiltonian(Hs)
        line(50)
        print("Grad: {}\nE: {}\n#gen: {}".format(g, e, len(qubit_pairs)))

        #write to csv
        energies.append(e)
        gradients.append(g)
        n_generators.append(len(qubit_pairs))

        d = {'poly': poly_types[:i+1], 'indices': indices[:i+1], 'energies': energies, 'gradients': gradients, 'n_gens': n_generators}
        print(d)
        df = pd.DataFrame(d)
        df.to_csv("./saved/data/grad_grad.csv")
