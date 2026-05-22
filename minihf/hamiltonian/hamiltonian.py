import numpy as np
import itertools
import os
from multiprocessing import Pool

from minihf.molecule import Molecule
from minihf.basis.basis_set import BasisSet
from minihf.hamiltonian.integral.overlap import Overlap
from minihf.hamiltonian.integral.kinetic import Kinetic
from minihf.hamiltonian.integral.nuclear_attraction import NuclearAttraction
from minihf.hamiltonian.integral.electron_repulsion import ElectronRepulsion


__all__ = ['Hamiltonian']


ncpu = os.cpu_count()


class Hamiltonian(dict):
    """Chemical Hamiltonian for a Schrodinger equation.

    Methods
    -------
    __init__(self, nspatial)
        Initialize the Hamiltonian.

    assign_operators(self, operator)
        Assigns operator integral to the Hamiltonian.

    ClassMethods
    ------------
    build(cls,molecule,basis_set)
        Build the Chemical Hamiltonian
    """
    def __new__(cls):
        """Generate new molecule object.
        """
        obj = dict().__new__(cls)

        return obj

    @classmethod
    def build(cls,mol,orbs):
        """Build the chemical hamiltonian.

        Parmeters
        ---------
        mol : Molecule
            Instance of Molecule.

        orb : BasisSet
            Instance of BasisSet.
        """
        #Create new object instance.
        ham = cls()
        ham.mol = mol
        ham.orbs = orbs
        
        #Build integral matrices.
        S = ham.build_overlap(orbs)
        T = ham.build_kinetic(orbs)
        V = ham.build_nuclear_attraction(mol, orbs)
        Hcore = T+V
        Eri = ham.build_electron_repulsion(orbs)
        #Eri = ElectronRepulsion.build(orbs)
        
        #Assign operators to Hamiltonian.
        ham['S'] = S
        ham['T'] = T
        ham['V'] = V
        ham['Hcore'] = Hcore
        ham['Eri'] = Eri
        
        return ham
    

    def build_overlap(self,basis_set):
        """Build the overlap integral matrix.

        Parameters
        ----------
        basis_set : BasisSet
            Hatree Fock basis set instance.

        Raises
        ------
        TypeError
            If basis_set parameter is not a BasisSet instance.

        Return
        ------
        S_matrix: np.array
            Return the overlap integral matrix.
        """
        if not isinstance(basis_set, BasisSet):
            raise TypeError("basis_set parameter must be a BasisSet instance.")
        
        S_integral = Overlap()
        norb = len(basis_set)    
        S_matrix = np.zeros((norb,norb))
        
        for i, j in itertools.product(range(norb), repeat=2):
            if i <= j:
                S_matrix[i,j] = S_integral.contracted(basis_set[i],basis_set[j])
                
        S_matrix = S_matrix + S_matrix.T - np.diag(S_matrix.diagonal())
        return S_matrix
    
    def build_kinetic(self,basis_set):
        """Build the kinetic integral matrix.

        Parameters
        ----------
        basis_set : BasisSet
            Hatree Fock basis set instance.

        Raises
        ------
        TypeError
            If basis_set parameter is not a BasisSet instance.

        Return
        ------
        T_matrix: np.array
            Return the kinetic integral matrix.
        """
        if not isinstance(basis_set, BasisSet):
            raise TypeError("basis_set parameter must be a BasisSet instance.")
        
        T_integral = Kinetic()
        norb = len(basis_set)    
        T_matrix = np.zeros((norb,norb))
        
        for i, j in itertools.product(range(norb), repeat=2):
            if i <= j:
                T_matrix[i,j] = T_integral.contracted(basis_set[i],basis_set[j])
        T_matrix = T_matrix + T_matrix.T - np.diag(T_matrix.diagonal())
        return T_matrix 

    def build_nuclear_attraction(cls,molecule,basis_set):
        """Build the nuclear repulsion operator.

        Parameters
        ----------
        basis_set : BasisSet
            Hatree Fock basis set instance.

        Raises
        ------
        TypeError
            If basis_set parameter is not a BasisSet instance.

        Return
        ------
        operator: Kinetic
            Return an operator instance.
        """  
        if not isinstance(molecule, Molecule):
            raise TypeError("molecule parameter must be a Molecule instance")
        if not isinstance(basis_set, BasisSet):
            raise TypeError("basis_set parameter must be a BasisSet instance")
        
        V_integral = NuclearAttraction()
        norb = len(basis_set)
        V_matrix = np.zeros((norb,norb))
        
        for i, j in itertools.product(range(norb), repeat=2):
            if i <= j:
                V_matrix[i,j] = V_integral.contracted(basis_set[i],basis_set[j],molecule)
        V_matrix = V_matrix + V_matrix.T - np.diag(V_matrix.diagonal())
        return V_matrix


    def build_electron_repulsion(cls,basis_set):
        """Build the electron repulsion integral matrix.

        Parameters
        ----------
        basis_set : BasisSet
            Hatree Fock basis set instance.

        Raises
        ------
        TypeError
            If basis_set parameter is not a BasisSet instance.

        Return
        ------
        Eri_matrix: np.array
            Return the electron repulsion integral matrix.
        """
        from minihf.hamiltonian.auxiliary import eint
        if not isinstance(basis_set, BasisSet):
            raise TypeError("basis_set parameter must be a BasisSet instance")

        Eri_integral = ElectronRepulsion()
        norb = len(basis_set)
        integral_list = []    
        Eri_tensor = np.zeros((norb,norb,norb,norb))

        # Calculate integral list with 8-fold permutational symmetry
        indices = []
        bases = []
        n = 0
        for a, b, c, d in itertools.product(range(norb), repeat=4):
            if not (a < b or c < d or a < c or (a == c and b < d)):
                indices.append((a,b,c,d))
                bases.append((basis_set[a],basis_set[b],basis_set[c],basis_set[d]))
        pool = Pool(ncpu)
        integral_list = pool.starmap(Eri_integral.contracted, bases)
        pool.close()

        # Build integral tensor from list 
        for a, b, c, d in itertools.product(range(norb), repeat=4):
            Eri_tensor[a,b,c,d] = integral_list[eint(a,b,c,d)]

        return Eri_tensor
    