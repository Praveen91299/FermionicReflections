import numpy as np
from scipy.linalg import expm, logm
from utils.ferm_utils import get_U
from utils.mat_utils import is_antihermitian, pad_2d_to_square, is_unitary

### objects

class OrbitalRotation:
    """
    General orbital rotation object defined by generator_mat

    """
    def __init__(self, n_qubits, generator_mat):
        self.n_qubits = n_qubits
        self.generator_mat = generator_mat
    
    @property
    def n_qubits(self):
        return self._n_qubits
    
    @n_qubits.setter
    def n_qubits(self, value):
        assert isinstance(value, int) and value > 0, "Number of qubits {} invalid".format(value)
        self._n_qubits = value

    @property
    def generator_mat(self):
        return self._generator_mat
    
    @generator_mat.setter
    def generator_mat(self, value):

        ### checks square shape of n_qubits x n_qubits, anti hermiticity

        assert (np.shape(value) == (self.n_qubits, self.n_qubits)), "Generator matrix not of correct dimensions!"
        assert is_antihermitian(value), "Generator matrix not antihermitian!"

        self._generator_mat = value
    
    def get_mat_rep(self):
        """
        Returns N x N matrix representation of unitary

        """

        return expm(self.generator_mat)
    
    def get_exp_rep(self):
        """
        Returns 2^N x 2^N (sparse) matrix representation of unitary
        
        """
        return get_U(self.get_mat_rep(), self.n_qubits)
    
    @classmethod
    def num_params(cls, n_qubits):
        return 0

    def get_num_params(self):
        return 0


class ParameterizedOrbitalRotation(OrbitalRotation):
    """
    Parameterized orbital rotation - angles stored in params - cannot be directly used
    
    """
    def __init__(self, params):
        print("DO NOT INITIALIZE PARAMETERIZED ORBITAL ROTATION DIRECTLY")

    @classmethod
    def build_param_mat(self, params, n_qubits):
        pass

    @classmethod
    def num_params(cls, n_qubits):
        pass
    
    def get_num_params(self):
        return self.num_params(self.n_qubits)

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        assert len(value) == self.get_num_params(), "Incorrect number of params passed!"
        self._params = value
    
    @property
    def generator_mat(self):
        return self.build_param_mat(self.params, self.n_qubits)
    
    def freeze_params(self):
        """
        Returns unparameterized version

        """

        return OrbitalRotation(self.n_qubits, self.generator_mat)

class RealOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Real orbital rotation

    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params
    
    @classmethod
    def num_params(cls, n_qubits):
        return n_qubits*(n_qubits-1)//2
    
    @classmethod
    def build_param_mat(cls, params, n_qubits):
        """
        Real orbital rotations, N(N-1)/2 parameters
        """
        N = cls.get_num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        theta = params

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param_mat[i, j] =   theta[idx]
                param_mat[j, i] = - theta[idx]

                idx += 1

        return param_mat


class ImaginaryOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Imaginary orbital rotation
    
    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params

    @classmethod
    def num_params(cls, n_qubits):
        return n_qubits*(n_qubits-1)//2
    
    @classmethod
    def build_param_mat(cls, params, n_qubits):
        """
        Imaginary rotations, N(N-1)/2 parameters 
        """
        N = cls.num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        phi = params

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param_mat[i, j] = 1.j * phi[idx] 
                param_mat[j, i] = 1.j * phi[idx]
                
                idx += 1
        
        return param_mat

class FullOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Full orbital rotation
    
    """
    def __init__(self, n_qubits, params):
        self.n_qubits = n_qubits
        self.params = params

    @classmethod
    def num_params(cls, n_qubits):
        return n_qubits*(n_qubits-1)
    
    @classmethod
    def build_param_mat(cls, params, n_qubits):
        """
        Full U_mf, N(N-1) parameters
        """

        ## get anti hermitian matrix, transform polynomial, and convert to Sparse

        N = cls.num_params(n_qubits)
        assert len(params) == N, "Number of parameters provided don't match!"
        phi = params[:N//2]
        theta = params[N//2:]

        param_mat = np.zeros((n_qubits, n_qubits), complex)

        idx = 0
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                param_mat[i, j] =   theta[idx] + 1.j * phi[idx] 
                param_mat[j, i] = - theta[idx] + 1.j * phi[idx]

                idx += 1
        
        return param_mat

class RestrictedOrbitalRotation(ParameterizedOrbitalRotation):
    """
    Restricted orbital rotation - to qubit pairs

    """
    def __init__(self, n_qubits, params, qubit_pairs):
        self.n_qubits = n_qubits
        self.params = params
        self.qubit_pairs = qubit_pairs
    
    @classmethod
    def num_params(cls, qubit_pairs):
        return len(qubit_pairs)
    
    def get_num_params(self):
        return self.num_params(self.qubit_pairs)

    @classmethod
    def build_param_mat(cls, params, n_qubits, qubit_pairs):
        """
        Excitations restricted to subset of qubit pairs provided (i, j, r/i)
        
        """

        assert len(params) == len(qubit_pairs), "Number of parameters provided don't match!"

        param_mat = np.zeros((n_qubits, n_qubits), complex)

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
    
    @property
    def generator_mat(self):
        return self.build_param_mat(self.params, self.n_qubits, self.qubit_pairs)

### functions

def combine_orbital_rotations(orbital_rotation_list: list[OrbitalRotation]):
    """
    Create single orbital rotation obj from a list of orbital rotations

    Returns OrbitalRotation of largest qubit size of the list

    """

    #combined_u = orbital_rotation_list[0].get_mat_rep()
    n_qubits_max = int(np.max([orb.n_qubits for orb in orbital_rotation_list]))
    combined_u = np.eye(n_qubits_max, dtype=complex)

    for orb in orbital_rotation_list:
        
        gen = pad_2d_to_square(orb.generator_mat, n_qubits_max)
        combined_u = combined_u @ expm(gen)
    
    combined_generator_matrix = logm(combined_u)

    assert is_unitary(combined_u)
    assert is_antihermitian(combined_generator_matrix)

    return OrbitalRotation(n_qubits=n_qubits_max, generator_mat=combined_generator_matrix)