from minihf.basis.gauss import ContractedGaussian

import copy
import itertools

__all__ = ['BasisSet']

class BasisSet(list):
    """One electron basis set.
    
    Classmethods
    ------------
    build(cls,molecule,basisfile)
        Build the instance.

    Methods
    -------
    __new__(cls)
        Generate new instance.
    
    __init__(self)
        Initialize the instance.
    
    append(self,basis)
        Apend basis to basis set.
    """
    def __new__(cls):
        """Generate new instance.
        """
        obj = list.__new__(cls)
        return obj

    @classmethod
    def build(cls,molecule,basisfile):
        """Build the basis set.

        Parameters
        ----------
        molecule: Molecule
            Moleucle instance.

        basisfile : str
            Name of the basis set file.

        Return
        ------
        basis_set: BasisSet
            Basis set instance.
        """
        basis_set = BasisSet()

        if not isinstance(basisfile, str):
            raise TypeError("basisfile must be str")
        if basisfile.endswith('.nwchem'):
            from minihf.basis import nwchem
            for atom in molecule:
                old_len = len(basis_set)
                nwchem.load(atom,basis_set,basisfile)
                basisfile.strip('.nwchem')
        else:
            raise ValueError("unknown file format")

        return basis_set

    def append(self,basis):
        """Append basis to basis set.

        Parameters
        ----------
        basis : ContractedGaussian
            ContractedGaussian instance.
        
        Raises
        ------
        TypeError
            If basis is not a ContractedGaussian instance.
        """
        if not isinstance(basis, ContractedGaussian):
            raise TypeError("Basis must be ContractedGaussian instance")
        super().append(basis)