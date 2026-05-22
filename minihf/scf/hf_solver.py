import numpy as np

__all__ = ['HFSolver']

class HFSolver(object):
    """Plain self-consistent mean field method solver.

    Attributes
    ----------
    ham: Hamiltonian
        Chemical Hamiltonian.
    I_thld: int
        Threshold of iteration number.
    E_thld: float
        Threshold of energy convergence.
    D_thld: float
            Threshold of density matrix convergence.
    guess: str
            Guess method for density matrix.
            Default is 'core_hamiltonian'.
        
    Methods
    -------
    __init__(self,ham,I_thld=100,E_thld=1.0E-8,D_thld=1.0E-8,guess='core_hamiltonian')
        Initialize the solver.
    rhf(self)
        Restricted Hartree Fock solver.
    uhf(self)
        Unrestricted Hartree Fock solver.
    rohf(self)
        Restricted Open-shell Hartree Fock solver.
    ghf(self)
        Generalized Hartree Fock solver.
    """
    def __init__(self,ham,I_thld=1000,E_thld=1.0E-8,D_thld=1.0E-8,guess='core_hamiltonian'):
        """Initialize the solver.

        Parameters
        ----------
        ham: Hamiltonian
            Chemical Hamiltonian.
        I_thld: int
            Threshold of iteration number.
        E_thld: float
            Threshold of energy convergence.
        D_thld: float
            Threshold of density matrix convergence.
        guess: str
            Guess method for density matrix.
            Default is 'core_hamiltonian'.
        """
        self.ham = ham
        self.I_thld = I_thld
        self.E_thld = E_thld
        self.D_thld = D_thld
        self.guess = guess

    def rhf(self, guess='core_hamiltonian'):
        """Restricted Hartree Fock solver.
        
        Parameters
        ----------
        guess: str
            Guess method for density matrix.
            Default is 'core_hamiltonian'.

        Returns
        -------
        results : dict
            Hartree Fock calculation results.
        """
        #Set attributes
        mol = self.ham.mol  
        orbs = self.ham.orbs
        
        energy_nuc = mol.energy_nuc
        nalpha = mol.nalpha
        nbeta = mol.nbeta
        ndocc = nalpha
        
        S = np.array(self.ham['S'])
        Hcore = np.array(self.ham['Hcore'])
        Eri = np.array(self.ham['Eri'])
        
        if not nalpha == nbeta:
            raise ValueError("System must be a restricted close shell.")        

        I_thld = self.I_thld
        E_thld = self.E_thld
        D_thld = self.D_thld
        
        from minihf.scf.rhf import rhf
        results = rhf(S,Hcore,Eri,ndocc,energy_nuc,I_thld,E_thld,D_thld,guess)

        return results

    def uhf(self, guess='core_hamiltonian'):
        """Unrestricted Hartree Fock solver.
        
        Parameters
        ----------
        guess: str
            Guess method for density matrix.
            Default is 'core_hamiltonian'.

        Returns
        -------
        results : dict
            Hartree Fock calculation results.
        """
        #Set attributes
        mol = self.ham.mol
        
        energy_nuc = mol.energy_nuc
        S = np.array(self.ham['S'])
        Hcore = np.array(self.ham['Hcore'])
        Eri = np.array(self.ham['Eri'])

        nalpha = mol.nalpha
        nbeta = mol.nbeta
        
        I_thld = self.I_thld
        E_thld = self.E_thld
        D_thld = self.D_thld
        
        from minihf.scf.uhf import uhf
        results = uhf(S,Hcore,Eri,nalpha,nbeta,energy_nuc,I_thld,E_thld,D_thld,guess)
        
        return results

    def rohf(self, guess='extended_huckel'):
        """Restricted open shell Hartree Fock solver.
        
        Parameters
        ----------
        guess: str
            Guess method for density matrix.
            Default is 'extended_huckel'.

        Returns
        -------
        results : dict
            Hartree Fock calculation results.
        """
        #Set attributes
        mol = self.ham.mol
        
        energy_nuc = mol.energy_nuc
        S = np.array(self.ham['S'])
        Hcore = np.array(self.ham['Hcore'])
        Eri = np.array(self.ham['Eri'])

        nalpha = mol.nalpha
        nbeta = mol.nbeta
        
        I_thld = self.I_thld
        E_thld = self.E_thld
        D_thld = self.D_thld
        
        from minihf.scf.rohf import rohf
        results = rohf(S,Hcore,Eri,nalpha,nbeta,energy_nuc,I_thld,E_thld,D_thld,guess)
        
        return results

    def ghf(self, guess='extended_huckel'):
        """Generalized Hartree Fock solver.

        Parameters
        ----------
        guess: str
            Guess method for density matrix.
            Default is 'extended_huckel'.
        
        Returns
        -------
        results : dict
            Hartree Fock calculation results.
        """
        mol = self.ham.mol

        energy_nuc = mol.energy_nuc
        S = np.array(self.ham['S'])
        Hcore = np.array(self.ham['Hcore'])
        Eri = np.array(self.ham['Eri'])

        nalpha = mol.nalpha
        nbeta = mol.nbeta

        I_thld = self.I_thld
        E_thld = self.E_thld
        D_thld = self.D_thld

        from minihf.scf.ghf import ghf
        results = ghf(S, Hcore, Eri, nalpha, nbeta, energy_nuc, I_thld, E_thld, D_thld, guess)

        return results