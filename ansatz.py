### stuff

### TODO
"""
A code base to construct and test reflection ansatz
select generators of maximum gradient (requires optimizing orbital rotation)
    Have orbital gate reductions
Optimize energy expectation value
Check dressing approach reduction

Have a class that holds ansatz description and performs dressing
Class for generator object.

"""

import numpy as np
from copy import deepcopy
from openfermion import get_sparse_operator, expectation, jordan_wigner, FermionOperator, hermitian_conjugated
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from scipy.linalg import expm
from utils.ferm_utils import *
from utils.mat_utils import *

def reflection_transform(H, R, tau, sparse = False):
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
    sin_t2 = np.sin(tau/2)
    cos_t2 = np.cos(tau/2)
    
    if sparse:
        HR = H@R
        RHR = R@HR
        comm_HR = HR - HR.getH()
    else:
        # Commutator [H, R] = H*R - R*H
        HR = H*R
        RHR = R * HR
        comm_HR = HR - hermitian_conjugated(HR)

    # Combine terms
    H_tilde = (
        (cos_t2**2) * H 
        - 1j * (sin_t2 * cos_t2) * comm_HR
        + (sin_t2**2) * RHR
    )  
    
    return H_tilde

def grad(params, n_qubits, V_type, poly, Hs, ref_wfn, qubit_pairs, basis_dict):
    #n_qubits, V_type, poly, Hs, ref_wfn = args

    #build R
    Rs = FermionicReflection.build_R(params, n_qubits, V_type, poly, sparse=True, basis_dict=basis_dict, qubit_pairs = qubit_pairs)

    #commutator
    com = Hs @ Rs - Rs @ Hs
    
    g =  - np.abs(np.imag(expectation(com, ref_wfn))) #commutator is anti hermitian, thus expectation values are imaginary, and we maximize absolute value.
    return g


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

class ReflectionAnsatz:
    def __init__(self, n_qubits, ref_state, reflections = [], taus = []):
        """
        Class for Reflection VQE ansatz

        Uses exp(-i * R * theta/2) = cos(theta/2) - i * sin(theta/2) * R convention
        """
        
        self.n_qubits = n_qubits
        self.ref_state = ref_state
        self.reflections = reflections

        if taus == []:    
            self.taus = np.zeros(len(reflections))

        return
    
    def add_ref(self, ref, tau):

        self.reflections.append(ref)
        self.taus.append(tau)

        return
    
    def energy(self, Hs):

        state = deepcopy(self.ref_state)
        
        for refl, tau in zip(self.reflections, self.taus):
            R = refl.get_R(True)
            state = np.cos(tau/2) * state - (np.sin(tau/2) * 1.j) * R@state

        return expectation(Hs, state)
    
    def optimize_energy(self, Hs, tau_init = None):
        """
        Optimize Taus
        """

        def energy_at(taus, reflections, ref_state, Hs):

            state = deepcopy(ref_state)
            for refl, tau in zip(reflections, taus):
                state = np.cos(tau/2) * state - (np.sin(tau/2) * 1.j) * refl@state

            return np.real(expectation(Hs, state))
        
        #ensure hermitian
        assert is_hermitian(Hs), "Hamiltonian not hermitian, cannot optimize energy"
        
        refs = [refl.get_R(True) for refl in self.reflections]
        if tau_init is None:
            tau_init = self.taus
        
        result = minimize(energy_at, tau_init, args=(refs, self.ref_state, Hs))
        self.taus = result.x

        print("\nOptimization terminated successfully, energy at {}".format(self.energy(Hs)))
        return
    
    def dress_hamiltonian(self, H, sparse = True):
        """
        Returns dressed Hamiltonain

        H : FermionOperator or csc_matrix

        if sparse == True, expects sparse H
        
        """
        if sparse:
            assert sp.issparse(H), "Hamiltonian not sparse..."

        H_new = deepcopy(H)
        
        n_ref = len(self.reflections)
        for k in reversed(range(0, n_ref)):
            R = self.reflections[k].get_R(sparse)
            tau = self.taus[k]

            H_new = reflection_transform(H_new, R, tau=tau, sparse=sparse)
        
        return H_new

class FermionicReflection:
    def __init__(self, n_qubits, poly, V_type, params_init = None, **kwargs):
        """
        Build a reflective fermionic operator by utilizing the structure

        V * p * V^\dagger
            V ~ mean-field rotation
            p ~ reflective polynomial of occupations
        
        """

        self.n_qubits = n_qubits

        if "basis_dict" in kwargs:
            self.basis_dict = kwargs["basis_dict"]
        else:
            self.basis_dict = None

        self.V_type = V_type
        if V_type == 'restricted':
            self.qubit_pairs = kwargs["qubit_pairs"]
        else:
            self.qubit_pairs = None
        
        if params_init == None:
            self.params = np.random.rand(FermionicReflection.get_num_params(n_qubits, V_type, **kwargs))

        # polynomial
        self.poly = poly
        poly_s = get_sparse_operator(poly)
        assert is_close_to_identity(poly_s@poly_s), "Polynomial not reflective"

    @classmethod
    def get_num_params(cls, n_qubits, V_type, **kwargs):
        if V_type == 'all':
            return n_qubits*(n_qubits-1)
        elif V_type == 'real' or V_type == 'imag':
            return n_qubits*(n_qubits-1)//2
        elif V_type == 'restricted':

            qubit_pairs = kwargs["qubit_pairs"]
            return len(qubit_pairs)
    
    @classmethod
    def build_param_mat(cls, params, n_qubits, V_type, **kwargs):

        param_mat = np.zeros((n_qubits, n_qubits), complex)
        if V_type == 'all':
            """
            Full U_mf, N(N-1) parameters
            """

            ## get anti hermitian matrix, transform polynomial, and convert to Sparse

            N = FermionicReflection.get_num_params(n_qubits, V_type)
            assert len(params) == N, "Number of parameters provided don't match!"
            phi = params[:N//2]
            theta = params[N//2:]

            idx = 0
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    param_mat[i, j] =   theta[idx] + 1.j * phi[idx] 
                    param_mat[j, i] = - theta[idx] + 1.j * phi[idx]

                    idx += 1
        
        if V_type == 'real':
            """
            Real orbital rotations, N(N-1)/2 parameters
            """
            N = FermionicReflection.get_num_params(n_qubits, V_type)
            assert len(params) == N, "Number of parameters provided don't match!"
            theta = params

            idx = 0
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    param_mat[i, j] =   theta[idx]
                    param_mat[j, i] = - theta[idx]

                    idx += 1

        
        if V_type == 'imag':
            """
            Imaginary rotations, N(N-1)/2 parameters 
            """
            N = FermionicReflection.get_num_params(n_qubits, V_type)
            assert len(params) == N, "Number of parameters provided don't match!"
            phi = params

            idx = 0
            for i in range(n_qubits):
                for j in range(i+1, n_qubits):
                    param_mat[i, j] = 1.j * phi[idx] 
                    param_mat[j, i] = 1.j * phi[idx]
                    
                    idx += 1
        
        if V_type == 'restricted':
            """
            Excitations restricted to subset of qubit pairs provided (i, j, r/i)
            
            """

            qubit_pairs = kwargs["qubit_pairs"]
            assert len(params) == len(qubit_pairs), "Number of parameters provided don't match!"

            idx = 0
            for t in qubit_pairs:
                i, j, kind = t

                if kind == "real":
                    param_mat[i, j] += params[idx]
                    param_mat[j, i] += -params[idx]
                
                if kind == "imag":
                    param_mat[i, j] += 1.j * params[idx]
                    param_mat[j, i] += 1.j * params[idx]
                
                idx += 1

        return param_mat
    
    def init_basis_dict(self):
        """
        Creates and stores sparse basis_dict

        """
        if self.basis_dict is None:
            self.basis_dict = build_sparse_basis(self.n_qubits)
    
    def get_param_mat(self):
        return FermionicReflection.build_param_mat(self.params, self.n_qubits, self.V_type, qubit_pairs = self.qubit_pairs)
    
    @classmethod
    def build_R(cls, params, n_qubits, V_type, poly, sparse=False, basis_dict=None, **kwargs):
        
        param_mat = cls.build_param_mat(params, n_qubits, V_type, **kwargs)
        U = expm(param_mat)

        #dress poly # TODO move poly tensor conversion outside
        poly_tbt, const = chem_ferm_to_chem_tbt(promote_cartan_twobody(poly), n_qubits) #ignoring constant term as it is a global phase.
        R_tbt = rotate_chem_tbt(poly_tbt, U)

        if sparse:

            if basis_dict is None:
                op = chem_tbt_to_chem_ferm(R_tbt) + const
                return get_sparse_operator(jordan_wigner(op), n_qubits)
            
            return get_sparse_fermop(R_tbt, basis_dict=basis_dict) + const*sp.eye(2**n_qubits)
        else:
            op = chem_tbt_to_chem_ferm(R_tbt) + const
            return op

    def get_R(self, sparse=False):
        return FermionicReflection.build_R(self.params, self.n_qubits, self.V_type, self.poly, sparse=sparse, basis_dict=self.basis_dict, qubit_pairs=self.qubit_pairs)

    def get_gradient(self, Hs, ref_wfn):
        """
        Return gradient at current parameter value

        """
        return grad(self.params, self.n_qubits, self.V_type, self.poly, Hs, ref_wfn, self.qubit_pairs, self.basis_dict)

    def optimize_grad(self, Hs, ref_wfn, init_param_type = 'zero', parallel = True):
        """
        Optimize params to maximize generator gradient wrt to Hamiltonian H

        Since generators are hermitian, gradient is imaginary
        """
        
        n_params = FermionicReflection.get_num_params(self.n_qubits, self.V_type, qubit_pairs=self.qubit_pairs)
        print("Initializing basis dictionary...")
        self.init_basis_dict()
        
        print("Basis Dictionary initialized.\n\n\nInitializing gradient optimization of polynomial {} ...".format(self.poly))

        if init_param_type == 'zero':
            param_init = np.zeros(n_params)
        elif init_param_type == 'random':
            param_init = np.random.rand(n_params)
        
        print("Initial parameters: {}".format(param_init))
        
        if parallel:
            result = minimize_parallel(grad, param_init, args=(self.n_qubits, self.V_type, self.poly, Hs, ref_wfn, self.qubit_pairs, self.basis_dict))
        else:
            result = minimize(grad, param_init, args=(self.n_qubits, self.V_type, self.poly, Hs, ref_wfn, self.qubit_pairs, self.basis_dict))
        self.params = result.x

        print("\n\nCompleted gradient optimization, max gradient = {}".format(-result.fun))
        return result.x