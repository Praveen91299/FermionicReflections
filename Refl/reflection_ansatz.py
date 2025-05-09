"""
Objects for reflection ansatz

"""

from copy import deepcopy
from openfermion import expectation, hermitian_conjugated, FermionOperator, QubitOperator
import scipy.sparse as spr
import numpy as np
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from utils.mat_utils import is_hermitian
from utils.ferm_utils import return_qubitop, return_sparse
from Refl.reflection_generator import FermionicReflection

def energy_at(taus, refs, ref_state, Hs):
    """
    Energy at particular taus. 

    refs, Hs are sparse matrices.
    """

    state = deepcopy(ref_state)
    for refl, tau in zip(refs, taus):
        state = np.cos(tau/2) * state - (np.sin(tau/2) * 1.j) * refl@state

    return np.real(expectation(Hs, state))

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

class ReflectionAnsatz:
    def __init__(self, n_qubits, ref_state, reflections: list[FermionicReflection] = [], taus = None):
        """
        Class for Reflection VQE ansatz

        Uses exp(-i * R * theta/2) = cos(theta/2) - i * sin(theta/2) * R convention
        """
        
        self.n_qubits = n_qubits
        self.ref_state = ref_state
        self.reflections = reflections

        if taus is None:    
            taus = np.zeros(self.get_num_params())
        
        self.set_params(taus)
    
    def add_ref(self, ref, tau):

        self.reflections.append(ref)
        self.taus.append(tau)

        return
    
    def get_num_params(self):
        return len(self.reflections)
    
    def set_params(self, params):
        """
        Sets tau values

        """
        assert len(params) == self.get_num_params(), 'len(params): {}, while {} params expected.'.format(len(params), self.get_num_params())

        self.taus = params

    def energy(self, Hs):

        state = deepcopy(self.ref_state)
        
        for refl, tau in zip(self.reflections, self.taus):
            R = refl.get_R(True)
            state = np.cos(tau/2) * state - (np.sin(tau/2) * 1.j) * R@state

        return expectation(Hs, state)
    
    def optimize_energy(self, Hs, parallel=True, n_random = 10):
        """
        Optimize Taus

        Does not return anything, use self.energy(Hs)
        """
        
        #ensure hermitian
        assert is_hermitian(Hs), "Hamiltonian not hermitian, cannot optimize energy"
        
        refs = [refl.get_R(True) for refl in self.reflections]

        n_params = self.get_num_params()
        params_list = [np.zeros(n_params)]
        for _ in range(n_random):
            params_list.append(np.random.rand(n_params))

        min_energy = energy_at(params_list[0], refs, self.ref_state, Hs)
        params_at_min_energy = params_list[0]

        print("Entering energy optimization...")
        for i, params in enumerate(params_list):
            print("Trial {}, Initial Tau: {}".format(i+1, params))

            if parallel:
                result = minimize_parallel(energy_at, params, args=(refs, self.ref_state, Hs))
            else:
                result = minimize(energy_at, params, args=(refs, self.ref_state, Hs))
            
            print("Completed optimization, energy = {}".format(result.fun))

            if result.fun < min_energy:
                min_energy = result.fun
                params_at_min_energy = result.x
        
        print("\nOptimization terminated successfully, energy: {}".format(min_energy))
        self.taus = params_at_min_energy
    
    def dress_hamiltonian(self, H, operator_type = "sparse", transform='jw'):
        """
        Returns dressed Hamiltonain

        H : FermionOperator or csc_matrix

        if sparse == True, expects sparse H
        
        """
        if operator_type == "sparse":
            sparse = True
            H_new = return_sparse(H)
        else:
            sparse = False
        
        if operator_type == "QubitOperator":
            H_new = return_qubitop(H, n_qubits = self.n_qubits, transform = transform)
        elif operator_type == "FermionOperator":
            assert type(H) is FermionOperator, "Hamiltonian not fermion operator"
            H_new = deepcopy(H)
        
        n_ref = len(self.reflections)
        for k in reversed(range(0, n_ref)):
            R = self.reflections[k].get_R(sparse)
            tau = self.taus[k]

            if operator_type == "QubitOperator":
                R = return_qubitop(R, n_qubits=self.n_qubits, transform = transform)
            
            H_new = reflection_transform(H_new, R, tau=tau, sparse=sparse)
        
        return H_new