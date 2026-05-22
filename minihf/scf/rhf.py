import numpy as np
from minihf.scf.initial_guess import *

def rhf(S,Hcore,Eri,nocc,energy_nuc,I_thld,E_thld,D_thld,guess='core_hamiltonian'):
    """Restricted close-shell Hartree Fock algorithm.

    Parameters
    ----------
    S: np.ndarray
        Overlap matrix.
    Hcore: np.ndarray
        Core Hamiltonian matrix.
    Eri: np.ndarray
        Electron repulsion matrix.
    nocc: int
        Number of occupied orbitals.
    I_thld: int
        Threshold of iteration number.
    E_thld: float
        Threshold of energy convergence.
    D_thld: float
            Threshold of density matrix convergence.

    Returns
    -------
    results : dict
        Hartree Fock calculation results.
    """
    # Build the orthogonalization matrix X = S^(-1/2)
    eigval, eigvec = np.linalg.eigh(S)
    X = eigvec @ np.diag(eigval**(-0.5)) @ eigvec.T
    
    # Initial guess for HF
    if guess=='core_hamiltonian':
        C = core_hamiltonian(Hcore, S)
    elif guess=='extended_huckel':
        C = extended_huckel(Hcore, S)
    else:
        raise ValueError(f"Unknown guess method: {guess}")
    
    D = np.einsum('pi,qi->pq', C[:,:nocc], C[:,:nocc])
    
    # Store initial value of electronic energy and density matrix
    Dold = D.copy()
    Eold = np.einsum('pq,pq->', Hcore, D)

    # SCF Iterations
    for Iter in range(1,I_thld+1):
        # Update Fock matrix
        J = np.einsum('pqrs,rs->pq', Eri, D)
        K = np.einsum('prqs,rs->pq', Eri, D)
        F = Hcore + 2*J - K

        # Update density matrix
        Fp = X @ F @ X
        Emo, Cp = np.linalg.eigh(Fp)
        C = X @ Cp
        D = np.einsum('pi,qi->pq', C[:,:nocc], C[:,:nocc])
        
        # Update electronic energy
        E = np.einsum('pq,pq->', Hcore+F, D)
        
        # Update convergence
        dE = abs(E-Eold)
        dRMS = np.mean((D-Dold)**2)**0.5

        # Store Updated electronic energy and density matrix
        Dold = D
        Eold = E

        # Check convergence
        if (dE < E_thld) and (dRMS < D_thld):
            break
        if Iter == I_thld:
            raise Exception("Maximum number of SCF cycles exceeded.")
    
    D_total = 2*D

    results = {
    "energy_elec": float(E),
    "energy_nuc": float(energy_nuc),
    "energy_total": float(E + energy_nuc),
    "mo_energy": Emo,
    "mo_coeff": C,
    "rdm1": D_total
    }
    
    return results
