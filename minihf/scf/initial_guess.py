import numpy as np
from numpy.linalg import eigh

def core_hamiltonian(Hcore, S):
    """ Core Hamiltonian matrix initial guess.

    Parameters
    ----------
    H_core : np.ndarray
        Core Hamiltonian matrix

    S : np.ndarray
        AO overlap matrix

    Returns
    -------
    C : np.ndarray
        Core Hamiltonian guess MO coefficients
    """
    # Build the orthogonalization matrix X = S^(-1/2)
    eigval, eigvec = np.linalg.eigh(S)
    X = eigvec @ np.diag(eigval**(-0.5)) @ eigvec.T
    
    Hp = X.T @ Hcore @ X
    _, Cp = eigh(Hp)
    C = X @ Cp 
    
    return C

def extended_huckel(Hcore, S, K=1.75):
    """ Extended Hückel initial guess.

    Parameters
    ----------
    Hcore : np.ndarray
        Core Hamiltonian matrix

    S : np.ndarray
        AO overlap matrix

    K : float
        Wolfsberg-Helmholtz constant

    Returns
    -------
    C : np.ndarray
        Extended Hückel guess MO coefficients
    """
    # Build the orthogonalization matrix X = S^(-1/2)
    eigval, eigvec = np.linalg.eigh(S)
    X = eigvec @ np.diag(eigval**(-0.5)) @ eigvec.T
    
    Hdiag = np.diag(Hcore)
    F_guess = 0.5 * K * S * (Hdiag[:, None] + Hdiag[None, :])
    np.fill_diagonal(F_guess, Hdiag)
    _, Cp = np.linalg.eigh(F_guess)
    C = X @ Cp
    
    return C