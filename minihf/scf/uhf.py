import numpy as np
from minihf.scf.initial_guess import *

def uhf(S,Hcore,Eri,nalpha,nbeta,energy_nuc,I_thld=500,E_thld=1e-10,D_thld=1e-8,guess='core_hamiltonian'):
    """Unrestricted open-shell Hartree Fock algorithm.

    Parameters
    ----------
    S: np.ndarray
        Overlap matrix.
    Hcore: np.ndarray
        Core Hamiltonian matrix.
    Eri: np.ndarray
        Electron repulsion integrals (in chemists' notation, aosym='s8').
    nalpha: int
        Number of spin alpha electrons.
    nbeta: int
        Number of spin beta electrons.
    energy_nuc: float
        Nuclear repulsion energy.
    I_thld: int
        Maximum number of iterations.
    E_thld: float
        Threshold for energy convergence.
    D_thld: float
        Threshold for density matrix convergence.
    guess: str
        Initial guess method.
    break_symmetry: bool
        Whether to break symmetry for open-shell systems.

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
            
    # Initial alpha and beta density matrices
    Da = np.einsum('pi,qi->pq', C[:, :nalpha], C[:, :nalpha])
    Db = np.einsum('pi,qi->pq', C[:, :nbeta], C[:, :nbeta])
    
    # Break symmetry for open-shell systems
    np.random.seed(42)
    perturbation = 1e-4 * np.random.randn(*Da.shape)
    perturbation = (perturbation + perturbation.T) / 2
    Da += perturbation
    Db -= perturbation
        
    # Initial Fock matrices
    Ja = np.einsum('pqrs,rs->pq', Eri, Da)
    Jb = np.einsum('pqrs,rs->pq', Eri, Db)
    Ka = np.einsum('prqs,rs->pq', Eri, Da)
    Kb = np.einsum('prqs,rs->pq', Eri, Db)
    Fa = Hcore + Ja + Jb - Ka
    Fb = Hcore + Ja + Jb - Kb

    # Store initial value of electronic energy and density matrix
    D = Da + Db 
    Dold = D.copy()
    Eold = 0.5 * (np.einsum('pq,pq->', Hcore, Da) + 
        np.einsum('pq,pq->', Hcore, Db) +
        np.einsum('pq,pq->', Fa, Da) +
        np.einsum('pq,pq->', Fb, Db))

    # SCF Iterations
    for Iter in range(1, I_thld + 1):
        # Update Fock matrices
        Ja = np.einsum('pqrs,rs->pq', Eri, Da)
        Jb = np.einsum('pqrs,rs->pq', Eri, Db)
        Ka = np.einsum('prqs,rs->pq', Eri, Da)
        Kb = np.einsum('prqs,rs->pq', Eri, Db)

        Fa = Hcore + Ja + Jb - Ka
        Fb = Hcore + Ja + Jb - Kb

        # Transform to orthogonal basis and diagonalize (alpha)
        Fp = X @ Fa @ X
        Emo_a, Cp = np.linalg.eigh(Fp)
        Ca = X @ Cp
        Da = np.einsum('pi,qi->pq', Ca[:, :nalpha], Ca[:, :nalpha])

        # Transform to orthogonal basis and diagonalize (beta)
        Fp = X @ Fb @ X
        Emo_b, Cp = np.linalg.eigh(Fp)
        Cb = X @ Cp
        Db = np.einsum('pi,qi->pq', Cb[:, :nbeta], Cb[:, :nbeta])

        # Total density matrix
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
        Eold = E

        # Check convergence
        if (dE < E_thld) and (dRMS < D_thld):
            break
        if Iter == I_thld:
            raise Exception(f"Maximum number of SCF cycles ({I_thld}) exceeded.")

    # Compute <S^2> and multiplicity for diagnostic purposes
    S2, mult = spin_expectation(Da,Db,S,nalpha,nbeta)
    
    # Prepare results
    Emo = [Emo_a, Emo_b]
    C = [Ca, Cb]
    D = [Da, Db]

    results = {
        "energy_elec": float(E),
        "energy_nuc": float(energy_nuc),
        "energy_total": float(E + energy_nuc),
        "mo_energy": Emo,
        "mo_coeff": C,
        "rdm1": D,
        "s2": float(S2),
        "mult": float(mult)
    }

    return results

def spin_expectation(Da,Db,S,nalpha,nbeta):
    """
    Compute S^2 expectation value for UHF wavefunction.
    
    Parameters
    ----------
    Da : np.ndarray
        Alpha density matrix (nbf, nbf)
    Db : np.ndarray
        Beta density matrix (nbf, nbf)
    S : np.ndarray
        Atomic overlap matrix (nbf, nbf)
    nalpha : int
        Number of alpha electrons
    nbeta : int
        Number of beta electrons
    
    Returns
    -------
    s2 : float
        Expectation value of S^2
    mult : float
        Multiplicity (2S+1)
    """
    # Compute <S^2> for diagnostic purposes
    S_z = (nalpha - nbeta) / 2.0
    S2 = S_z * (S_z + 1) + nbeta - np.trace(Da @ S @ Db @ S)

    # Multiplicity
    mult = 2 * (np.sqrt(S2 + 0.25) - 0.5) + 1
    
    return S2, mult