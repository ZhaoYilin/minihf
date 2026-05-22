import numpy as np
from minihf.scf.initial_guess import *

def ghf(S, Hcore, Eri, nalpha, nbeta, energy_nuc, I_thld=500, E_thld=1e-10, D_thld=1e-8, guess='extended_huckel'):
    """Generalized Hartree-Fock (GHF) algorithm for non-collinear spin systems.
    
    GHF allows each molecular orbital to have arbitrary spin character (mix of alpha and beta),
    enabling description of non-collinear spin arrangements and spin-orbit coupling effects.

    Parameters
    ----------
    S: np.ndarray
        Overlap matrix (nao, nao).
    Hcore: np.ndarray
        Core Hamiltonian matrix (nao, nao).
    Eri: np.ndarray
        Electron repulsion integrals in chemists' notation (nao, nao, nao, nao), aosym='s8'.
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
        Initial guess method: 'core_hamiltonian' or 'extended_huckel'. Default is 'extended_huckel'.

    Returns
    -------
    results : dict
        Hartree Fock calculation results.
    """
    # Orbital classification
    nao = S.shape[0]
    nso = 2 * nao
    nelec = nalpha + nbeta

    # Build orthogonalization matrix for GHF (block-diagonal)
    # X = S^(-1/2) transforms from AO to orthonormal basis
    eigval, eigvec = np.linalg.eigh(S)
    eigval_inv_sqrt = np.where(eigval > 1e-12, eigval**(-0.5), 0.0)
    X = eigvec @ np.diag(eigval_inv_sqrt) @ eigvec.T
    X_ghf = np.kron(np.eye(2), X)  # Block-diagonal for alpha and beta

    # Initial guess for HF
    if guess=='core_hamiltonian':
        C_spatial = core_hamiltonian(Hcore, S)
    elif guess=='extended_huckel':
        C_spatial = extended_huckel(Hcore, S)
    else:
        raise ValueError(f"Unknown guess method: {guess}")
    
    # Block-diagonal spin orbitals
    C = np.zeros((nso, nso))
    C[:nao, :nao] = C_spatial
    C[nao:, nao:] = C_spatial

    # Initial density matrices
    D = np.einsum('pi,qi->pq', C[:,:nelec], C[:,:nelec])    
    Da = D[:nao, :nao]      # Alpha-alpha block
    Db = D[nao:, nao:]      # Beta-beta block
    Dab = D[:nao, nao:]     # Alpha-beta block (spin mixing)
    Dba = D[nao:, :nao]     # Beta-alpha block (spin mixing)    

    # Break symmetry for open-shell systems
    np.random.seed(42)
    
    # Add perturbation to break UHF symmetry
    # This creates non-zero off-diagonal spin blocks
    perturbation = 1e-4 * np.random.randn(nao, nao)
    perturbation = (perturbation + perturbation.T) / 2  # Keep symmetric
    
    # Add perturbation to spin blocks
    Da += perturbation
    Db -= perturbation
    
    # Add off-diagonal perturbation to allow spin mixing
    Dab += 1e-4 * np.random.randn(nao, nao)
    Dba += 1e-4 * np.random.randn(nao, nao)
    
    # Rebuild full density matrix
    D = np.block([[Da, Dab], [Dba, Db]])
    
    # Ensure electron count is conserved
    N_current = np.trace(D[:nao, :nao] @ S).real + np.trace(D[nao:, nao:] @ S).real
    scale = nelec / N_current
    D = D * scale
    D_old = D.copy()

    # Initial Fock matrices based on initial density
    D_total = Da + Db
    J = np.einsum('pqrs,rs->pq', Eri, D_total)      # Coulomb matrix
    Kaa = np.einsum('prqs,rs->pq', Eri, Da)         # Alpha exchange
    Kbb = np.einsum('prqs,rs->pq', Eri, Db)         # Beta exchange
    Kab = np.einsum('prqs,rs->pq', Eri, Dab)        # Alpha-beta exchange
    Kba = np.einsum('prqs,rs->pq', Eri, Dba)        # Beta-alpha exchange

    Fa = Hcore + J - Kaa    # Alpha Fock matrix
    Fb = Hcore + J - Kbb    # Beta Fock matrix
    Fab = -Kab              # Alpha-beta Fock matrix (off-diagonal)
    Fba = -Kba              # Beta-alpha Fock matrix (off-diagonal)

    # Initial electronic energy
    E_old = 0.5 * np.einsum('pq,pq->', Da + Db, Hcore) + \
            0.5 * (np.einsum('pq,pq->', Fa, Da) + np.einsum('pq,pq->', Fb, Db) +
                   np.einsum('pq,pq->', Fab, Dab) + np.einsum('pq,pq->', Fba, Dba))

    # SCF Iterations
    for Iter in range(1, I_thld + 1):
        # Compute Fock matrices based on current density
        D_total = Da + Db
        J = np.einsum('pqrs,rs->pq', Eri, D_total)
        Kaa = np.einsum('prqs,rs->pq', Eri, Da)
        Kbb = np.einsum('prqs,rs->pq', Eri, Db)
        Kab = np.einsum('prqs,rs->pq', Eri, Dab)
        Kba = np.einsum('prqs,rs->pq', Eri, Dba)

        Fa = Hcore + J - Kaa
        Fb = Hcore + J - Kbb
        Fab = -Kab
        Fba = -Kba

        # Build GHF Fock matrix (2*nao x 2*nao) in AO basis
        F_ghf = np.block([[Fa, Fab], [Fba, Fb]])

        # Transform to orthonormal basis and diagonalize
        F_ortho = X_ghf.T @ F_ghf @ X_ghf
        Emo, C_ortho = np.linalg.eigh(F_ortho)
        
        # Transform back to AO basis
        C = X_ghf @ C_ortho

        # Build new density matrix from occupied orbitals
        C_occ = C[:, :nelec]
        D = C_occ @ C_occ.T
        
        # Extract updated spin blocks
        Da = D[:nao, :nao]
        Db = D[nao:, nao:]
        Dab = D[:nao, nao:]
        Dba = D[nao:, :nao]

        # Calculate electronic energy using current Fock and density
        # E = 1/2 Tr(H * D) + 1/2 Tr(F * D)
        E = 0.5 * np.einsum('pq,pq->', Da + Db, Hcore) + \
            0.5 * (np.einsum('pq,pq->', Fa, Da) + np.einsum('pq,pq->', Fb, Db) +
                   np.einsum('pq,pq->', Fab, Dab) + np.einsum('pq,pq->', Fba, Dba))

        # Check convergence
        dE = abs(E - E_old)
        dRMS = np.mean((D - D_old) ** 2) ** 0.5

        # Update for next iteration
        D_old = D.copy()
        E_old = E

        # Check convergence
        if (dE < E_thld) and (dRMS < D_thld):
            break
        
        if Iter == I_thld:
            raise RuntimeError(f"GHF SCF did not converge in {I_thld} iterations. Last ΔE = {dE:.2e}, ΔRMS = {dRMS:.2e}")
        
    # Compute <S^2> and multiplicity for diagnostic purposes
    S2, mult = spin_expectation(C, S, nelec)

    # Prepare results dictionary
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


def spin_expectation(C, S, nelec):
    """
    Compute S^2 expectation value for GHF wavefunction.
    
    Parameters
    ----------
    C : np.ndarray
        GHF MO coefficients (2*nao, 2*nao)
    S : np.ndarray
        Atomic overlap matrix (nao, nao)
    nelec : int
        Number of electrons
    
    Returns
    -------
    s2 : float
        Expectation value of S^2
    mult : float
        Multiplicity (2S+1)
    """
    nao = C.shape[0] // 2
    
    # Extract alpha and beta MO coefficients
    mo_a = C[:nao, :nelec]  # Alpha occupied orbitals
    mo_b = C[nao:, :nelec]  # Beta occupied orbitals
    
    # Compute overlap matrices in MO basis
    saa = mo_a.conj().T @ S @ mo_a
    sbb = mo_b.conj().T @ S @ mo_b
    sab = mo_a.conj().T @ S @ mo_b
    sba = sab.conj().T
    
    # Number of alpha and beta electrons
    nocc_a = np.trace(saa).real
    nocc_b = np.trace(sbb).real
    
    # Compute <S_x^2 + S_y^2> = (N_alpha + N_beta)/2 + Tr(S_ba) * Tr(S_ab) - Tr(S_ba @ S_ab)
    ssxy = (nocc_a + nocc_b) * 0.5
    ssxy += sba.trace() * sab.trace() - np.einsum('ij,ji->', sba, sab)
    
    # Compute <S_z^2>
    ssz = (nocc_a + nocc_b) * 0.25
    ssz += (nocc_a - nocc_b)**2 * 0.25
    tmp = saa - sbb
    ssz -= np.einsum('ij,ji', tmp, tmp) * 0.25
    
    # Total S^2
    s2 = (ssxy + ssz).real
    
    # Multiplicity 2S+1
    mult = 2 * (np.sqrt(s2 + 0.25) - 0.5) + 1
    
    return s2, mult