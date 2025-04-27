"""
Objects for reflection ansatz

"""

from copy import deepcopy
from openfermion import expectation, hermitian_conjugated
import scipy.sparse as spr
import numpy as np
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from utils.mat_utils import is_hermitian
from Refl.reflection_generator import FermionicReflection

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
    def __init__(self, n_qubits, ref_state, reflections: list[FermionicReflection] = [], taus = []):
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
    
    def optimize_energy(self, Hs, tau_init = None, parallel=True):
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
            assert spr.issparse(H), "Hamiltonian not sparse..."

        H_new = deepcopy(H)
        
        n_ref = len(self.reflections)
        for k in reversed(range(0, n_ref)):
            R = self.reflections[k].get_R(sparse)
            tau = self.taus[k]

            H_new = reflection_transform(H_new, R, tau=tau, sparse=sparse)
        
        return H_new