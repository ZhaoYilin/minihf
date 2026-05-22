# MiniHF

MiniHF: A minimal pedagogical implementation of Hartree-Fock calculation from scratch.

## Features

- **Restricted Hartree-Fock (RHF)** - For closed-shell systems
- **Unrestricted Hartree-Fock (UHF)** - For open-shell systems with different alpha/beta electron counts
- **Restricted Open-shell Hartree-Fock (ROHF)** - For open-shell systems with spatial orbital restrictions
- **Generalized Hartree-Fock (GHF)** - For non-collinear spin systems

## Installation

```bash
git clone https://github.com/yourusername/minihf.git
cd minihf
pip install -e .
```

## Quick Start

```python
from minihf.molecule import Molecule
from minihf.basis import BasisSet
from minihf.hamiltonian import Hamiltonian
from minihf.scf import HFSolver

# Build molecule (H2)
mol = Molecule.build(["H 0 0 0", "H 0 0 0.74"], charge=0, multiplicity=1)

# Build basis set
basis = BasisSet.build(mol, 'sto-3g.nwchem')

# Build Hamiltonian
ham = Hamiltonian.build(mol, basis)

# Initialize solver and run RHF calculation
hf = HFSolver(ham)
results = hf.rhf()

print(f"Total energy: {results['energy_total']}")
```

## Running Tests

```bash
cd minihf
python -m pytest tests/ -v
```

# MiniHF

MiniHF: A minimal pedagogical implementation of Hartree-Fock calculation from scratch.

## Features

- **Restricted Hartree-Fock (RHF)** - For closed-shell systems
- **Unrestricted Hartree-Fock (UHF)** - For open-shell systems with different alpha/beta electron counts
- **Restricted Open-shell Hartree-Fock (ROHF)** - For open-shell systems with spatial orbital restrictions
- **Generalized Hartree-Fock (GHF)** - For non-collinear spin systems

## Installation

```bash
git clone https://github.com/yourusername/minihf.git
cd minihf
pip install -e .
```

## Quick Start

```python
from minihf.molecule import Molecule
from minihf.basis import BasisSet
from minihf.hamiltonian import Hamiltonian
from minihf.scf import HFSolver

# Build molecule (H2)
mol = Molecule.build(["H 0 0 0", "H 0 0 0.74"], charge=0, multiplicity=1)

# Build basis set
basis = BasisSet.build(mol, 'sto-3g.nwchem')

# Build Hamiltonian
ham = Hamiltonian.build(mol, basis)

# Initialize solver and run RHF calculation
hf = HFSolver(ham)
results = hf.rhf()

print(f"Total energy: {results['energy_total']}")
```

## Running Tests

```bash
cd minihf
python -m pytest tests/ -v
```