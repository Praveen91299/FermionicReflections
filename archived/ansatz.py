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
from openfermion import get_sparse_operator, expectation, jordan_wigner, FermionOperator, hermitian_conjugated, commutator
import scipy.linalg
from scipy.optimize import minimize
from optimparallel import minimize_parallel
import scipy
from utils.ferm_utils import *
from utils.mat_utils import *
from utils.misc_utils import line

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

def grad(params, n_qubits, excitation_type, poly, Hs, ref_wfn, qubit_pairs, basis_dict):
    """
    Reflection gradient, built from full matrix

    """
    #n_qubits, V_type, poly, Hs, ref_wfn = args

    #build R
    Rs = FermionicReflection.build_R(params, n_qubits, excitation_type, poly, sparse=True, basis_dict=basis_dict, qubit_pairs = qubit_pairs)

    #commutator
    com = Hs @ Rs - Rs @ Hs
    
    g =  - np.abs(np.imag(expectation(com, ref_wfn))) #commutator is anti hermitian, thus expectation values are imaginary, and we maximize absolute value.
    return g

def grad_iterative(params, n_qubits, qubit_pairs, tbt, Hs, basis_dict, ref_state):
    """
    Returns gradient of iteratively constructed R
    Dresses tbt by new orbital rotation determined by qubit_pairs, params

    """
    
    m = FermionicReflection.build_param_mat_iterative(params, n_qubits, qubit_pairs)
    U = scipy.linalg.expm(m)

    tbt_new = rotate_chem_tbt(tbt, U)
    Rs = get_sparse_fermop(tbt_new, basis_dict)

    #find gradient
    HR = Hs@Rs
    RH = HR.getH()

    return - np.abs(np.imag(expectation(HR- RH, ref_state)))

def select_generators(n_qubits, refl_op, H, state, n_generators):
    """
    Return generators/qubit pairs that have the highest contribution to energy lowering (determined based on gradient) of reflection ansatz


    """
    def get_excitation_gradient(generator, refl_op_sparse, H, state):
        return i*expectation(commutator(commutator(generator, refl_op_sparse), H), state)

    g_pq_real = lambda p, q: FermionOperator('{}^ {}'.format(p, q), 1.0) - FermionOperator('{}^ {}'.format(q, p), 1.0)
    g_pq_imag = lambda p, q: 1.j*(FermionOperator('{}^ {}'.format(p, q), 1.0) + FermionOperator('{}^ {}'.format(q, p), 1.0))

    ### prepare list of generators
    Hs = return_sparse(H, n_qubits)
    refl_op_sparse = return_sparse(refl_op, n_qubits)

    generators = []
    qubit_pairs = []
    for i in range(n_qubits):
        for j in range(n_qubits):
            if i != j:
                generator_real = get_sparse_operator(g_pq_real(i, j), n_qubits)
                qubit_pairs.append([i, j, "real"])
                generator_imag = get_sparse_operator(g_pq_imag(i, j), n_qubits)
                qubit_pairs.append([i, j, "imag"])
                generators.append(generator_real)
                generators.append(generator_imag)

    ### get gradients
    print("\nDetermining gradients of generators...")
    gradients = []
    for i, generator in enumerate(generators):
        gradients.append((qubit_pairs[i], get_excitation_gradient(generator, refl_op_sparse, Hs, state)))
    
    ### choose top gradients:
    gradients_sorted = sorted(gradients, key=lambda x: np.abs(x[1]), reverse=True)
    n = min(len(gradients_sorted), n_generators)
    chosen = gradients_sorted[:n]

    print("\nChoosing {} generators from {} generators:".format(n_generators, len(generators)))
    for c in chosen:
        print("Qubit pair: {}, generator gradient: {}".format(c[0], np.abs(c[1])))

    return [generator[0] for generator in chosen] ### return only qubit pairs

def iterative_V_construction(n_qubits, H, poly, n_generators, ref_state, basis_dict, optimization = "single", tol = 1e-3, n_random = 1):
    """
    Iteratively determine best generators for the orbital rotation defining the reflection ansatz for energy lowering/gradient maximization.

    optimization (Bool): Determines optimization routine for angles after each step
        Options:
        "single" - at each step only optimize new chosen generator angles
        "global" - optimize every angle parameters in each iteration, with previously found angles as starting point for old generators and random angles for new generators.

    tol (float): If the change in gradient is below tol, the search terminates

    n_random (int >0): nummber of randomized starting guesses for randomized optimization steps.
    
    """

    Hs = return_sparse(H, n_qubits)
    poly_tbt, const = chem_ferm_to_chem_tbt(promote_cartan_twobody(poly), n_qubits) #ignoring constant term as it is a global phase.

    qubit_pairs = []
    params = []
    gradients = [0]

    line(20)
    print("Starting iterative construction of V with {} generator optimization.\nPolynomial: {}".format(optimization, poly))

    for i in range(n_generators):
        print("\nIteration {}".format(i+1))

        #transform center by collected generators so far
        R = FermionicReflection.build_R_iterative(params, n_qubits, poly, qubit_pairs, False, basis_dict)
        Rs = get_sparse_operator(R, n_qubits)

        qubit_pair = select_generators(n_qubits, Rs, Hs, ref_state, n_generators=1)[0]

        #dress R_tbt by qubit_pair and maximize gradient
        if optimization == "single":
            max_grad = 0
            opt_params = deepcopy(params)
            R_tbt, const = chem_ferm_to_chem_tbt(R)

            for r in range(n_random):
                try:
                    result = minimize_parallel(grad_iterative, x0 = [np.random.rand()], args=(n_qubits, qubit_pair, R_tbt, Hs, basis_dict, ref_state))
                except:
                    print("Parallel minimization error. Terminated!")
                grad = np.abs(result.fun)

                if grad > max_grad:
                    max_grad = grad
                    opt_param = result.x
            
            print("\nMaximum gradient of {} at {} on {} random parameter starts.".format(max_grad, opt_param, n_random))
            opt_params.append(opt_param)
            
        elif optimization == "global":
            max_grad = 0
            opt_params = 0
            qubit_pairs_temp = deepcopy(qubit_pairs)
            qubit_pairs_temp.append(qubit_pair)

            for r in range(n_random):

                try:
                    result = minimize_parallel(fun=grad_iterative, x0=np.random.rand(len(qubit_pairs_temp)), args=(n_qubits, qubit_pairs_temp, poly_tbt, Hs, basis_dict, ref_state))
                except:
                    print("Parallel minimize error, terminated!")
                
                grad = np.abs(result.fun)

                if grad > max_grad:
                    max_grad = grad
                    opt_params = result.x

        if np.abs(gradients[-1] - max_grad) < tol and i > 0:
            line(20)
            print("\nGradient difference lower than tolerance, ending search with {} generators...".format(len(qubit_pairs)))
            return qubit_pairs, params
        else:
            print("\nGenerator {}, {}. Reflection gradient at {}".format(i+1, qubit_pair, max_grad))
            params = opt_params
            qubit_pairs.append(qubit_pair)
            gradients.append(max_grad)

    print("Ending search at {} generators.".format(len(qubit_pairs)))

    return qubit_pairs, params

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
        else:
            self.taus = taus
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
            assert scipy.sparse.issparse(H), "Hamiltonian not sparse..."

        H_new = deepcopy(H)
        
        n_ref = len(self.reflections)
        for k in reversed(range(0, n_ref)):
            R = self.reflections[k].get_R(sparse)
            tau = self.taus[k]

            H_new = reflection_transform(H_new, R, tau=tau, sparse=sparse)
        
        return H_new

class FermionicReflection:
    def __init__(self, n_qubits, poly, V_type = "joint", excitation_type = "all", params_init = None, **kwargs):
        """
        Build a reflective fermionic operator by utilizing the structure

        V * p * V^\dagger
            V ~ mean-field rotation
            p ~ reflective polynomial of occupations
        
        V_type : Type of orbital rotation construction
            'joint' - exp of a sum fo generators
            'iterative' - defined as a product of exp of generators
        
        excitation_type : Type of generators, applicable only for V_type = 'joint'
            'all'
            'real'
            'imag'
            'restricted'

        """

        self.n_qubits = n_qubits

        if "basis_dict" in kwargs:
            self.basis_dict = kwargs["basis_dict"]
        else:
            self.basis_dict = None

        self.V_type = V_type
        self.excitation_type = excitation_type

        if V_type == 'joint':
            if excitation_type == 'restricted':
                self.qubit_pairs = kwargs["qubit_pairs"]
            else:
                self.qubit_pairs = None
        elif V_type == 'iterative':
            self.qubit_pairs = kwargs["qubit_pairs"]
        
        if params_init is None:
            self.params = np.random.rand(FermionicReflection.get_num_params(n_qubits, V_type, excitation_type, **kwargs))
        else:
            self.params = params_init

        # polynomial
        self.poly = poly
        poly_s = get_sparse_operator(poly)
        assert is_close_to_identity(poly_s@poly_s), "Polynomial not reflective"

    @classmethod
    def get_num_params(cls, n_qubits, V_type, excitation_type, **kwargs):
        if excitation_type == 'restricted' or V_type == 'iterative':
            qubit_pairs = kwargs["qubit_pairs"]
            return len(qubit_pairs)
        
        if V_type == "joint":
            if excitation_type == 'all':
                return n_qubits*(n_qubits-1)
            elif excitation_type == 'real' or excitation_type == 'imag':
                return n_qubits*(n_qubits-1)//2
            
    
    @classmethod
    def build_param_mat(cls, params, n_qubits, excitation_type, **kwargs):

        param_mat = np.zeros((n_qubits, n_qubits), complex)
        if excitation_type == 'all':
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
        
        if excitation_type == 'real':
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

        
        if excitation_type == 'imag':
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
        
        if excitation_type == 'restricted':
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

    @classmethod
    def build_param_mat_iterative(cls, params, n_qubits, qubit_pairs):
        """
        Combines a train of qubit pair generators into matrix representation of exp of sum.

        U = ... U_3 U_2 U_1 U_0 order

        Note: Since there is no complete set or fixed ordering, passing qubit_pairs is mandatory

        """

        product = np.eye(n_qubits)

        for angle, qubit_pair in zip(params, qubit_pairs):
            assert qubit_pair[0] < n_qubits and qubit_pair[1] < n_qubits, "Qubit index out of range"

            mat = cls.build_param_mat([angle], n_qubits, "restricted", qubit_pairs = [qubit_pair])
            exp_mat = scipy.linalg.expm(mat)
            product = exp_mat @ product
        
        return scipy.linalg.logm(product)
    
    def init_basis_dict(self):
        """
        Creates and stores sparse basis_dict

        """
        if self.basis_dict is None:
            self.basis_dict = build_sparse_basis(self.n_qubits)
    
    def get_param_mat(self):

        if self.V_type == 'joint':
            return FermionicReflection.build_param_mat(self.params, self.n_qubits, self.excitation_type, qubit_pairs = self.qubit_pairs)
        elif self.V_type == 'iterative':
            return FermionicReflection.build_param_mat_iterative(self.params, self.n_qubits, qubit_pairs = self.qubit_pairs)
    
    @classmethod
    def build_R(cls, params, n_qubits, excitation_type, poly, sparse=False, basis_dict=None, **kwargs):
        
        param_mat = cls.build_param_mat(params, n_qubits, excitation_type, **kwargs)
        U = scipy.linalg.expm(param_mat)

        #dress poly # TODO move poly tensor conversion outside
        poly_tbt, const = chem_ferm_to_chem_tbt(promote_cartan_twobody(poly), n_qubits) #ignoring constant term as it is a global phase.
        R_tbt = rotate_chem_tbt(poly_tbt, U)

        if sparse:

            if basis_dict is None:
                op = chem_tbt_to_chem_ferm(R_tbt) + const
                return get_sparse_operator(jordan_wigner(op), n_qubits)
            
            return get_sparse_fermop(R_tbt, basis_dict=basis_dict) + const*scipy.sparse.eye(2**n_qubits)
        else:
            op = chem_tbt_to_chem_ferm(R_tbt) + const
            return op
    
    @classmethod
    def build_R_iterative(cls, params, n_qubits, poly, qubit_pairs, sparse=False, basis_dict=None):
        """
        Iteratively build Reflection based on poly, with excitations defined by qubit_pairs

        Note: Will return different matrix as in FermionicReflections.build_R()

        """

        param_mat = cls.build_param_mat_iterative(params, n_qubits, qubit_pairs)
        U = scipy.linalg.expm(param_mat)

        poly_tbt, const = chem_ferm_to_chem_tbt(promote_cartan_twobody(poly), n_qubits)
        R_tbt = rotate_chem_tbt(poly_tbt, U)

        if sparse:

            if basis_dict is None:
                op = chem_tbt_to_chem_ferm(R_tbt) + const
                return get_sparse_operator(jordan_wigner(op), n_qubits)
            
            return get_sparse_fermop(R_tbt, basis_dict=basis_dict) + const*scipy.sparse.eye(2**n_qubits)
        else:
            op = chem_tbt_to_chem_ferm(R_tbt) + const
            return op

    def get_R(self, sparse=False):

        if self.V_type == "joint":
            return FermionicReflection.build_R(self.params, self.n_qubits, self.excitation_type, self.poly, sparse=sparse, basis_dict=self.basis_dict, qubit_pairs=self.qubit_pairs)
        elif self.V_type == "iterative":
            return FermionicReflection.build_R_iterative(self.params, self.n_qubits, self.poly, self.qubit_pairs, sparse=sparse, basis_dict=self.basis_dict)

    def get_gradient(self, Hs, ref_wfn):
        """
        Return gradient at current parameter value

        """
        if self.V_type == "joint":

            return grad(self.params, self.n_qubits, self.excitation_type, self.poly, Hs, ref_wfn, self.qubit_pairs, self.basis_dict)
        elif self.V_type == "iterative":

            tbt, const = chem_ferm_to_chem_tbt(promote_cartan_twobody(self.poly), self.n_qubits)
            return grad_iterative(self.params, self.n_qubits, self.qubit_pairs, tbt, Hs, self.basis_dict, ref_wfn)

    def optimize_grad(self, Hs, ref_wfn, init_param_type = 'zero', parallel = True):
        """
        Optimize params to maximize generator gradient wrt to Hamiltonian H

        Since generators are hermitian, gradient is imaginary
        """
        
        n_params = FermionicReflection.get_num_params(self.n_qubits, self.V_type, self.excitation_type, qubit_pairs=self.qubit_pairs)
        print("Initializing basis dictionary...")
        self.init_basis_dict()
        
        print("Basis Dictionary initialized.\n\n\nInitializing gradient optimization of polynomial\n{} ...".format(self.poly))

        if init_param_type == 'zero':
            param_init = np.zeros(n_params)
        elif init_param_type == 'random':
            param_init = np.random.rand(n_params)
        
        print("Initial parameters: \n{}".format(param_init))
        
        if parallel:
            result = minimize_parallel(grad, param_init, args=(self.n_qubits, self.excitation_type, self.poly, Hs, ref_wfn, self.qubit_pairs, self.basis_dict))
        else:
            result = minimize(grad, param_init, args=(self.n_qubits, self.excitation_type, self.poly, Hs, ref_wfn, self.qubit_pairs, self.basis_dict))
        self.params = result.x

        print("\nCompleted gradient optimization, max gradient = {}".format(-result.fun))
        return result.x