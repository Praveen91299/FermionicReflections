from Refl.orbital_rotation import combine_orbital_rotations, OrbitalRotation
from utils.mat_utils import is_unitary, is_hermitian
from utils.ferm_utils import *
from utils.misc_utils import *

import scipy.sparse as spr
from scipy.optimize import minimize
from optimparallel import minimize_parallel
import numpy as np
from copy import deepcopy
from openfermion import get_sparse_operator, expectation, jordan_wigner, FermionOperator, hermitian_conjugated, commutator


### reflection generator constructions
def get_poly(kind: str, qubit_idx: list):

    ni = lambda i: FermionOperator('{}^ {}'.format(i, i), 1.0)
    p11 = lambda i: 1 - 2*ni(i)
    p21 = lambda i, j: 1 - 2*ni(i)*ni(j)
    p22 = lambda i, j: 1 - 2*ni(i) + 2*ni(i)*ni(j)
    p23 = lambda i, j: 1 - 2*ni(i) - 2*ni(j) + 2*ni(i)*ni(j)
    p24 = lambda i, j: 1 - 2*ni(i) - 2*ni(j) + 4*ni(i)*ni(j)
    p25 = lambda i, j, k: 1 - 2*ni(i) - 2*ni(j)*ni(k) + 2*ni(i)*ni(k)
    p26 = lambda i, j, k: 1 - 2*ni(i) - 2*ni(j)*ni(k) + 2*ni(i)*ni(k) + 2*ni(i)*ni(j)
    p27 = lambda i, j, k: 1 - 2*ni(i) - 2*ni(j) + 2*ni(i)*ni(k) + 2*ni(i)*ni(j)
    p28 = lambda i, j, k: 1 - 2*ni(i) - 2*ni(j) - 2*ni(k) + 2*ni(j)*ni(k) + 2*ni(i)*ni(k) + 2*ni(i)*ni(j)
    p29 = lambda i, j, k, l: 1 - 2*ni(i) - 2*ni(j) - 2*ni(k)*ni(l) + 2*ni(i)*ni(k) + 2*ni(j)*ni(l) + 2*ni(i)*ni(j)
    

    if kind == 'p11':
        assert len(qubit_idx) == 1
        return p11(qubit_idx[0])
    elif kind == 'p21':
        assert len(qubit_idx) == 2
        return p21(*qubit_idx)
    elif kind == 'p22':
        assert len(qubit_idx) == 2
        return p22(*qubit_idx)
    elif kind == 'p23':
        assert len(qubit_idx) == 2
        return p23(*qubit_idx)
    elif kind == 'p24':
        assert len(qubit_idx) == 2
        return p24(*qubit_idx)
    elif kind == 'p25':
        assert len(qubit_idx) == 3
        return p25(*qubit_idx)
    elif kind == 'p26':
        assert len(qubit_idx) == 3
        return p26(*qubit_idx)
    elif kind == 'p27':
        assert len(qubit_idx) == 3
        return p27(*qubit_idx)
    elif kind == 'p28':
        assert len(qubit_idx) == 3
        return p28(*qubit_idx)
    elif kind == 'p29':
        assert len(qubit_idx) == 4
        return p29(*qubit_idx) 

    raise Exception('Poly type {} not recognized'.format(kind))

def grad(params, n_qubits, poly, orbital_rotation_list: list[OrbitalRotation], Hs, ref_wfn, basis_dict):
    """
    Reflection gradient

    pass params as None to use orbital_rotation_list directly
    
    """

    ### split params, build orbital rotations

    if params is not None:
        OR_list_local = deepcopy(orbital_rotation_list)
        num_param_list = [OR.get_num_params() for OR in orbital_rotation_list]

        assert len(params) == sum(num_param_list), "Number of parameters passed do not match!"

        st_idx = 0
        for i, OR in enumerate(OR_list_local):
            OR.params = params[st_idx: st_idx + num_param_list[i]]
            st_idx += num_param_list[i]
        
        Rs = FermionicReflection.build_R(n_qubits, poly, OR_list_local, basis_dict, sparse=True)
    else:
        Rs = FermionicReflection.build_R(n_qubits, poly, orbital_rotation_list, basis_dict, sparse=True)
    
    com = Hs @ Rs - Rs @ Hs
    g =  - np.abs(np.imag(expectation(com, ref_wfn)))
    return g

class FermionicReflection:
    def __init__(self, n_qubits, polynomial, orbital_rotations = [], basis_dict = None):
        self.n_qubits = n_qubits
        self.poly = polynomial
        self.orbital_rotations = orbital_rotations

        #store FQ transformed operators
        self.basis_dict = basis_dict
    
    def init_basis_dict(self):
        """
        Creates and stores sparse basis_dict

        """
        if self.basis_dict is None:
            print("Initializing basis dictionary...")
            self.basis_dict = build_sparse_basis(self.n_qubits)
            print("Basis Dictionary initialized.\nInitializing gradient optimization...")
    
    def get_num_params(self):
        """
        Gets number of Parameters (defining all orbital rotations)
        
        """
        return sum([OR.get_num_params() for OR in self.orbital_rotations])
    
    def set_params(self, params):
        """
        Sets orbital rotation params
        
        """
        assert len(params) == self.get_num_params(), 'len(params): {}, while {} params expected.'.format(len(params), self.get_num_params())

        idx = 0
        for OR in self.orbital_rotations:
            OR.params = params[idx: idx + OR.get_num_params()]
            idx += OR.get_num_params()

    @classmethod
    def build_R(cls, n_qubits, poly, orbital_rotation_list, basis_dict=None, sparse=False):

        #combine orbital rotation list
        combined_rotation = combine_orbital_rotations(orbital_rotation_list=orbital_rotation_list)
        U = combined_rotation.get_mat_rep()

        assert is_unitary(U), "Combined orbital rotation not unitary."

        poly_tbt, const = chem_ferm_to_chem_tbt(promote_cartan_twobody(poly), n_qubits) #ignoring constant term as it is a global phase.
        R_tbt = rotate_chem_tbt(poly_tbt, U)

        if sparse:

            if basis_dict is None:
                op = chem_tbt_to_chem_ferm(R_tbt) + const
                return get_sparse_operator(jordan_wigner(op), n_qubits)
            
            return get_sparse_fermop(R_tbt, basis_dict=basis_dict) + const*spr.eye(2**n_qubits)
        else:
            op = chem_tbt_to_chem_ferm(R_tbt) + const
            return op

    def get_R(self, sparse=True):
        return self.build_R(n_qubits = self.n_qubits, poly = self.poly, orbital_rotation_list=self.orbital_rotations, basis_dict=self.basis_dict, sparse=sparse)
    
    def get_gradient(self, Hs, ref_wfn):
        """
        Return gradient at current orbital parameter value
        
        """

        return grad(None, self.n_qubits, self.poly, self.orbital_rotations, Hs, ref_wfn, basis_dict=self.basis_dict)

    def optimize_grad(self, Hs, ref_wfn, n_random=10, parallel = True):
        """
        Optimize orbital rotation parameters to maximize gradient wrt (sparse) Hamiltonian Hs

        """

        n_params = self.get_num_params()
        self.init_basis_dict()

        params_list = [np.zeros(n_params)]
        for _ in range(n_random):
            params_list.append(np.random.rand(n_params))
        
        max_gradient = -1
        params_at_max_gradient = []

        print("Entering gradient optimization...")
        for i, params in enumerate(params_list):
            print("Trial {}, Initial params: {}".format(i+1, params))

            if parallel:
                result = minimize_parallel(grad, params, args=(self.n_qubits, self.poly, self.orbital_rotations, Hs, ref_wfn, self.basis_dict))
            else:
                result = minimize(grad, params, args=(self.n_qubits, self.poly, self.orbital_rotations, Hs, ref_wfn, self.basis_dict))
            
            print("Completed optimization, gradient = {}".format(-result.fun))

            if abs(result.fun) > max_gradient:
                max_gradient = abs(result.fun)
                params_at_max_gradient = result.x

        print("Maximum gradient: {}".format(max_gradient))
        self.set_params(params_at_max_gradient)

