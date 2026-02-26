### Feb 22, 2026 p24 full V adapt version with truncated Hamiltonians
import numpy as np

from Refl.orbital_rotation import FullOrbitalRotation, RestrictedOrbitalRotation
from Refl.reflection_generator import FermionicReflection, get_poly
from Refl.reflection_ansatz import ReflectionAnsatz
import numpy as np

from utils.hf_utils import *
from utils.mat_utils import ensure_real, truncate_pauli_hamiltonian
from utils.ferm_utils import build_sparse_basis
from openfermionpyscf import run_pyscf
from openfermion import get_fermion_operator, count_qubits, get_sparse_operator, MolecularData
import pandas as pd
from openfermion import QubitOperator, jordan_wigner
import pickle as pkl
from copy import deepcopy

#prep

if __name__ == "__main__":
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
    HQ = jordan_wigner(H)

    n_qubits = count_qubits(H)
    Hs = get_sparse_operator(H, n_qubits)
    n_electrons = n_qubits//2

    Hs = get_sparse_operator(H, n_qubits)
    occ = get_hf_occ(n_electrons, n_qubits//2, spin_ord)
    hf_wfn = get_hf_wfn(occ)

    basis_dict = build_sparse_basis(n_qubits)

    n_gens = 10
    conv_tol = 1.5e-3
    trunc_tol = 1e-5

    HQ_dressed = deepcopy(HQ)
    Hs_dressed = deepcopy(Hs)

    energies = []
    gradients = []
    Reflections = []

    poly_list = ['p2{}'.format(i) for i in range(1, 10)]
    poly_dict = {'p21': get_poly('p21', [0, 1]),
                 'p22': get_poly('p22', [0, 1]),
                 'p23': get_poly('p23', [0, 1]),
                 'p24': get_poly('p24', [0, 1]),
                 'p25': get_poly('p25', [0, 1, 2]),
                 'p26': get_poly('p26', [0, 1, 2]),
                 'p27': get_poly('p27', [0, 1, 2]),
                 'p28': get_poly('p28', [0, 1, 2]),
                 'p29': get_poly('p29', [0, 1, 2, 3]),
                 }

    for n_iter in range(n_gens):
        print("\nGenerator ", n_iter+1)

        V = FullOrbitalRotation(n_qubits, params = [np.random.rand() for _ in range(FullOrbitalRotation.num_params(n_qubits))])

        #random start required only for first generator
        if n_iter == 0:
            n_rand_grad = 1
        else:
            n_rand_grad = 0
        
        grads = []
        opt_FRs = []
        for p, poly in poly_dict.items():
            print("Optimizing gradient for class ", p)
            FR = FermionicReflection(n_qubits, poly, [V], basis_dict=basis_dict)
            
            FR.optimize_grad(Hs_dressed, hf_wfn, n_rand_grad)
            grads.append(np.abs(FR.get_gradient(Hs_dressed, hf_wfn)))
            opt_FRs.append(FR)
        
        max_p_idx = grads.index(max(grads))
        print("Maximum gradient {} for class {} ".format(grads[max_p_idx], poly_list[max_p_idx]))
        FR = opt_FRs[max_p_idx]
        gradients.append(grads[max_p_idx])

        Refl = ReflectionAnsatz(n_qubits, hf_wfn, [FR])
        Refl.optimize_energy(Hs_dressed, n_random=1)
        energies.append(Refl.energy(Hs_dressed))
        print(energies[-1])
        
        HQ_dressed = Refl.dress_hamiltonian(HQ_dressed, "QubitOperator")
        HQ_dressed = truncate_pauli_hamiltonian(ensure_real(HQ_dressed), trunc_tol)
        Hs_dressed = get_sparse_operator(HQ_dressed)

        Reflections.append(Refl)

    ### final optimization
    energies = []
    for i in range(1, n_gens+1):
        ansatz = ReflectionAnsatz(n_qubits, hf_wfn, reflections = [RA.reflections[0] for RA in Reflections[:i]], taus = [RA.taus[0] for RA in Reflections[:i]])
        
        print("\nBeginning energy optimization with {} reflections. Energy at: {}".format(len(ansatz.reflections), ansatz.energy(Hs)))
        ansatz.optimize_energy(Hs, n_random=1)
        energies.append(np.float(ansatz.energy(Hs)))

        #save energies and gradients
        df = pd.DataFrame({'energy': energies, 'gradients': gradients[:len(energies)]})
        df.to_csv('./saved/data/H4_1.0_energies_all_refl_1e-5.csv')
    # save all data

    filename='./saved/data/H4_1.0_all_refl_1e-5.pkl'
    print("Saving ansatz with reflections to file: ", filename)

    with open(filename, 'wb') as file:
        pkl.dump(ansatz, file)