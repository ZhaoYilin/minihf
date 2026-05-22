import numpy as np
from minihf.scf.initial_guess import *

def rohf(S, Hcore, Eri, nalpha, nbeta, energy_nuc, I_thld, E_thld, D_thld, guess='extended_huckel'):
    """ Restricted Open-shell Hartree Fock algorithm using Roothaan equations.

    Parameters
    ----------
    S: np.ndarray
        Overlap matrix.
    Hcore: np.ndarray
        Core Hamiltonian matrix.
    Eri: np.ndarray
        Electron repulsion matrix.
    nalpha: int
        Number of spin alpha electrons.
    nbeta: int
        Number of spin beta electrons.
    I_thld: int
        Threshold of iteration number.
    E_thld: float
        Threshold of energy convergence.
    D_thld: float
        Threshold of density matrix convergence.
    guess: str
        Initial guess method. Default is 'extended_huckel'.

    Returns
    -------
    results : dict
        Hartree Fock calculation results.
    """
    # Orbital classification
    nocc = max(nalpha, nbeta)        # Occupied orbitals
    ndocc = min(nalpha, nbeta)       # Doubly occupied orbitals

    # Initial guess: extended Huckel theory
    if guess=='core_hamiltonian':
        C = core_hamiltonian(Hcore, S)
    elif guess=='extended_huckel':
        C = extended_huckel(Hcore, S)
    else:
        raise ValueError(f"Unknown guess method: {guess}")
    
    # Initial alpha and beta density matrices
    Ddocc = np.einsum('pi,qi->pq', C[:, :ndocc], C[:, :ndocc])
    Dsocc = np.einsum('pi,qi->pq', C[:, ndocc:nocc], C[:, ndocc:nocc])
    Da = Ddocc + Dsocc
    Db = Ddocc
    D = Da + Db
    Dold = D.copy()
    
    Ja = np.einsum('pqrs,rs->pq', Eri, Da)
    Jb = np.einsum('pqrs,rs->pq', Eri, Db)
    Ka = np.einsum('prqs,rs->pq', Eri, Da)
    Kb = np.einsum('prqs,rs->pq', Eri, Db)
    Fa = Hcore + Ja + Jb - Ka
    Fb = Hcore + Ja + Jb - Kb

    # Initial guess of electronic energy
    Eold = 0.5 * (np.einsum('pq,pq->', Hcore, Da) + 
                   np.einsum('pq,pq->', Hcore, Db) +
                   np.einsum('pq,pq->', Fa, Da) +
                   np.einsum('pq,pq->', Fb, Db))    
     
    # SCF Iterations
    for Iter in range(I_thld):
        # Update Fock matrices
        Ja = np.einsum('pqrs,rs->pq', Eri, Da)
        Jb = np.einsum('pqrs,rs->pq', Eri, Db)
        Ka = np.einsum('prqs,rs->pq', Eri, Da)
        Kb = np.einsum('prqs,rs->pq', Eri, Db)
        Fa = Hcore + Ja + Jb - Ka
        Fb = Hcore + Ja + Jb - Kb

        # Build Canonical ROHF effective Fock
        Fa_mo = C.T @ Fa @ C
        Fb_mo = C.T @ Fb @ C
        Feff_mo = 0.5 * (Fa_mo + Fb_mo)

        # closed-open block
        Feff_mo[:ndocc, ndocc:nocc] = Fb_mo[:ndocc, ndocc:nocc]
        Feff_mo[ndocc:nocc, :ndocc] = Fb_mo[ndocc:nocc, :ndocc]
        # open-virtual block
        Feff_mo[ndocc:nocc, nocc:] = Fa_mo[ndocc:nocc, nocc:]
        Feff_mo[nocc:, ndocc:nocc] = Fa_mo[nocc:, ndocc:nocc]

        Emo, Cp = np.linalg.eigh(Feff_mo)
        C = C @ Cp

        # New densities
        Ddocc = np.einsum('pi,qi->pq', C[:, :ndocc], C[:, :ndocc])
        Dsocc = np.einsum('pi,qi->pq', C[:, ndocc:nocc], C[:, ndocc:nocc])
        Da = Ddocc + Dsocc
        Db = Ddocc
        D = Da + Db
        
        # Electronic energy
        E = 0.5 * (np.einsum('pq,pq->', Hcore, Da) + 
                   np.einsum('pq,pq->', Hcore, Db) +
                   np.einsum('pq,pq->', Fa, Da) +
                   np.einsum('pq,pq->', Fb, Db))

        # Convergence criteria
        dE = abs(E - Eold)
        dRMS = np.mean((D - Dold) ** 2) ** 0.5

        # Update for next iteration
        Dold = D.copy()
        Eold = E.copy()

        # Check convergence
        if (dE < E_thld) and (dRMS < D_thld):
            break
        if Iter == I_thld:
            raise Exception(f"Maximum number of SCF cycles ({I_thld}) exceeded.")
        
    results = {
        "energy_elec": float(E),
        "energy_nuc": float(energy_nuc),
        "energy_total": float(E + energy_nuc),
        "mo_energy": Emo,
        "mo_coeff": C,
        "rdm1": [Da, Db]
    }

    return results