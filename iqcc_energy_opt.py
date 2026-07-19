from tequila.apps.qcc.qcc import IterativeQCC
from openfermion import jordan_wigner, expectation
import pandas as pd

import numpy as np

from Refl.orbital_rotation import FullOrbitalRotation, RestrictedOrbitalRotation
from Refl.reflection_generator import FermionicReflection, get_poly
from Refl.reflection_ansatz import ReflectionAnsatz
import numpy as np

from utils.hf_utils import *
from utils.ferm_utils import build_sparse_basis
from openfermionpyscf import run_pyscf
from openfermion import get_fermion_operator, count_qubits, get_sparse_operator, MolecularData
import pandas as pd
from openfermion import QubitOperator, jordan_wigner

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

def combine_iqcc_generator_list(generators):
    idx = 0
    gen_list_new = {}
    for gen_dict in generators:
        for k, v in zip(gen_dict.keys(), gen_dict.values()):
            gen_list_new['g{}'.format(idx)] = v
            idx += 1

    return gen_list_new

iqcc = IterativeQCC.init_from_hamiltonian(HQ, occ)
# iqcc.do_qcc(n_gen=10)

energies_qcc = [np.real(expectation(Hs, hf_wfn))]
terms_qcc = [iqcc.n_terms[0]]
trunc_tol=1e-5

iqcc = IterativeQCC.init_from_hamiltonian(HQ, occ)
n_gens = 10 #maximum generators

#iqcc to check
for iter in range(n_gens):
    iqcc.do_iteration(n_gen=1, compression_threshold=trunc_tol)

print(iqcc.generators)

#qcc like re-optimization
for i in range(1, n_gens+1):    
    #use generators to re-optimize QCC
    qcc = IterativeQCC.init_from_hamiltonian(HQ, occ)
    qcc.generators = [combine_iqcc_generator_list(iqcc.generators[:i])]
    qcc.get_qcc_unitary()
    qcc.optimize()
    #qcc.update(compression_threshold=trunc_tol)

    print(qcc.energies[-1])

    energies_qcc.append(qcc.energies[-1])
    print("QCC energy {} at {} generators found using iqcc.".format(energies_qcc[-1], i))
    #terms_qcc.append(qcc.n_terms[-1])

    df = pd.DataFrame({'energy': energies_qcc})#, 'terms': terms_qcc})
    df.to_csv('./saved/data/H4_energies_qcc_Feb23.csv')

#save ansatz for chemical accuracy