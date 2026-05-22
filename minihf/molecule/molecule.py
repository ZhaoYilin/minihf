from collections.abc import Iterable
import re
import numpy as np
from minihf.molecule.atom import Atom

class Molecule(object):
    """
    Molecule class representing a collection of atoms
    """   
    def __init__(self, atoms=None, charge=0, multiplicity=1):
        """
        Initialize a molecule
        
        Parameters
        ----------
        atoms : list, optional
            List of Atom objects
        charge : int, optional
            Total molecular charge (default: 0)
        multiplicity : int, optional
            Spin multiplicity (default: 1 for singlet)
        """
        self.atoms = []
        self.charge = charge
        self.multiplicity = multiplicity
        
        if atoms:
            for atom in atoms:
                self.add_atom(atom)

    @classmethod
    def build(cls, atoms, charge=0, multiplicity=1):
        """Build a molecule from Atom objects.

        Parameters
        ----------
        atoms : list
            A list of Atom objects.
        charge : int, optional
            Total molecular charge (default: 0)
        multiplicity : int, optional
            Spin multiplicity (default: 1 for singlet)
        """
        mol = cls()
        for atom_info in atoms:
            if isinstance(atom_info, Iterable):
                if isinstance(atom_info, str):
                    atom_info = re.split(r'[ ,;]+', atom_info.strip())
            else:
                raise TypeError("Atom info must be an iterable")
            
            if atom_info[0].isdigit():
                atom_number_label = int(atom_info[0])
            else:
                atom_number_label = atom_info[0]
            atom_coords = list(map(float, atom_info[1:4]))
            atom_obj = Atom(atom_number_label, atom_coords)
            mol.add_atom(atom_obj)
            
        mol.charge = charge
        mol.multiplicity = multiplicity
        return mol

    def add_atom(self, atom):
        """Add an atom to the molecule"""
        if not isinstance(atom, Atom):
            raise TypeError("Can only add Atom objects")
        self.atoms.append(atom)
    
    def remove_atom(self, index):
        """Remove atom at given index"""
        if index < 0 or index >= len(self.atoms):
            raise IndexError(f"Atom index {index} out of range")
        return self.atoms.pop(index)
    
    def get_atom(self, index):
        """Get atom at given index"""
        if index < 0 or index >= len(self.atoms):
            raise IndexError(f"Atom index {index} out of range")
        return self.atoms[index]

    def bond_length(self, i, j):
        """Calculate the distance between atom i and j.

        Parameters
        ----------
        i : int
            Index of first atom.
        
        j : int
            Index of second atom.
            
        Returns
        -------
        float
            Distance between atoms i and j
        """
        A = np.array(self[i].coords)
        B = np.array(self[j].coords)
        return np.linalg.norm(A - B)

    def bond_angle(self, i, j, k):
        """Calculate the bond angle between atoms i,j,k.
           Where atom j is the central atom.

        Parameters
        ----------
        i : int
            Index of first atom.
        
        j : int
            Index of second atom.
            
        k : int
            Index of third atom.

        unit : str, optional
            Unit of the returned angle ('rad' or 'deg', default: 'rad')
        
        Returns
        -------
        float
            Bond angle between atoms i,j,k in specified unit
        """        
        A = np.array(self[i].coords)
        B = np.array(self[j].coords)
        C = np.array(self[k].coords)
        
        # Calculate vectors from central atom
        BA = A - B
        BC = C - B
        
        # Calculate bond angle using dot product formula
        cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
        # Handle numerical precision issues
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        return np.degrees(angle_rad)


    def dihedral_angle(self, i, j, k, l):
        """Calculate dihedral angle for atom connectivity A,B,C,D.

        Parameters
        ----------
        i : int
            Index of first atom (atom A).
        
        j : int
            Index of second atom (atom B).
            
        k : int
            Index of third atom (atom C).

        l : int
            Index of fourth atom (atom D).

        Returns
        -------
        float
            Dihedral angle in degrees (-180 to 180)

        Raises
        ------
        TypeError
            If index of atom is not int.
        ValueError
            If index of atom is not zero or positive number.            
            If index of atom is not smaller than number of atoms.            
        """
        self._validate_atom_indices(i, j, k, l)
        
        A = np.array(self[i].coords)
        B = np.array(self[j].coords)
        C = np.array(self[k].coords)
        D = np.array(self[l].coords)
        
        # Calculate vectors
        AB = B - A
        BC = C - B
        CD = D - C
        
        # Calculate normals
        n1 = np.cross(AB, BC)
        n2 = np.cross(BC, CD)
        
        # Normalize normals
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        
        # Calculate dihedral angle
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # Determine sign using triple product
        sign = np.sign(np.dot(np.cross(n1, n2), BC))
        if sign == 0:
            sign = 1
        
        dihedral = np.degrees(angle_rad) * sign
        return dihedral
    

    @property
    def nelec(self):
        """Number of total electrons in the molecule."""
        return sum(atom.number for atom in self.atoms) - self.charge
    
    @property
    def nalpha(self):
        """Number of alpha electrons in the molecule."""
        return (self.nelec + self.multiplicity - 1) // 2
    
    @property
    def nbeta(self):
        """Number of beta electrons in the molecule."""
        return self.nelec - self.nalpha

    @property
    def energy_nuc(self):
        """Nucelar repulsion energy.

        Return
        ------
        energy: float
            Nuclear repulsion energy.
        """
        energy = 0.
        for i,A in enumerate(self):
            for j,B in enumerate(self):
                if j>i:
                    energy += (A.number*B.number)/self.bond_length(i,j)
        return energy

    @property
    def atom_coords(self):
        """Get all atomic coordinates as Nx3 array"""
        return [atom.coords for atom in self.atoms]

    @property
    def atom_masses(self):
        """Get all atomic masses as Nx1 array"""
        return [atom.mass for atom in self.atoms]
    
    @property
    def atom_numbers(self):
        """Get all atomic numbers as Nx1 array"""
        return [atom.number for atom in self.atoms]

    @property
    def atom_symbols(self):
        """Get all atomic symbols as Nx1 array"""
        return [atom.symbol for atom in self.atoms]
    
    @property
    def mass(self):
        """Get total molecular mass"""
        return sum(atom.mass for atom in self.atoms)

    @property
    def center_of_mass(self):
        """Calculate the center of mass for molecular.
        
        Returns
        -------
        np.ndarray
            [x, y, z] coordinates of center of mass
        """
        if len(self) == 0:
            return np.array([0.0, 0.0, 0.0])
        
        total_mass = self.mass
        if total_mass == 0:
            return np.mean(self.atom_coords, axis=0)
        
        weighted_coords = np.sum([atom.mass * atom.coords for atom in self.atoms], axis=0)
        return weighted_coords / total_mass
    
    def __len__(self):
        return len(self.atoms)
    
    def __getitem__(self, index):
        return self.get_atom(index)
    
    def __iter__(self):
        return iter(self.atoms)

    def _validate_atom_indices(self, i, j, k, l):
        if not all(isinstance(idx, int) for idx in (i, j, k, l)):
            raise TypeError("Atom indices must be integers")
        n_atoms = len(self)
        if any(idx < 0 or idx >= n_atoms for idx in (i, j, k, l)):
            raise IndexError("Atom index out of range")

    def __repr__(self):
        """String representation of molecule"""
        return f"Molecule(atoms={len(self)}, charge={self.charge}, multiplicity={self.multiplicity})"