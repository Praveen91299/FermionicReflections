from utils.pickle_utils import *
from ansatz import *
from openfermion import count_qubits, get_sparse_operator, FermionOperator
from utils.hf_utils import *

mol = 'h2'
spin_ord = 'udud'
filename = '{}{}_fer.bin'.format(mol, spin_ord)
directory = './saved/hamiltonians/'

H = get_pkl_object(filename=filename, directory=directory)

n_qubits = count_qubits(H)
n_electrons = n_qubits//2

Hs = get_sparse_operator(H, n_qubits)
occ = get_hf_occ(n_electrons, n_qubits//2, spin_ord)
hf_wfn = get_hf_wfn(occ)

o = FermionOperator('0^ 0 1^ 1', 2.0) - 1

qubit_pairs = []
for i in [1, 2, 3]:
    qubit_pairs.append((0, i, "imag"))
    qubit_pairs.append((1, i, "imag"))

kinds = ['p11', 'p21', 'p22', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p29']
indices = [
    [0],
    [0, 1], [0, 1], [0, 1], [0, 1],
    [0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2],
    [0, 1, 2, 3]
           ]
R_list = []

for i, kind in enumerate(kinds):
    p = get_poly(kind, indices[i])
    print('\n\nPolynomial = ', kind)
    fr = FermionicReflection(n_qubits, p, V_type='restricted', qubit_pairs=qubit_pairs)
    fr.optimize_grad(Hs, hf_wfn, init_param_type='random')
    R_list.append(fr)

## one particular class as generator

RA = ReflectionAnsatz(n_qubits=n_qubits, ref_state=hf_wfn, reflections=R_list[4:5])
print(RA.energy(Hs))

RA.optimize_energy(Hs, np.random.rand(len(R_list[4:5])))