import unittest
import numpy as np
from minihf.hamiltonian.hamiltonian import Hamiltonian
from minihf.molecule.molecule import Molecule
from minihf.basis.basis_set import BasisSet


class TestHamiltonian(unittest.TestCase):

    def test_build_hamiltonian_single_h(self):
        mol = Molecule.build(["H 0 0 0"], charge=0, multiplicity=1)
        basis = BasisSet.build(mol, 'sto-3g.nwchem')
        ham = Hamiltonian.build(mol, basis)

        self.assertIn('S', ham)
        self.assertIn('T', ham)
        self.assertIn('V', ham)
        self.assertIn('Hcore', ham)
        self.assertIn('Eri', ham)

        self.assertTrue(np.allclose(ham['Hcore'], ham['T'] + ham['V']))
        self.assertEqual(ham['S'].shape, ham['T'].shape)
        self.assertEqual(ham['S'].shape, ham['V'].shape)
        self.assertEqual(ham['S'].shape, ham['Hcore'].shape)
        self.assertEqual(ham['Eri'].shape, ham['S'].shape + ham['S'].shape)
        self.assertGreater(ham['S'][0, 0], 0.0)

    def test_build_hamiltonian_h2_geometry(self):
        mol = Molecule.build(["H 0 0 0", "H 0 0 0.74"], charge=0, multiplicity=1)
        basis = BasisSet.build(mol, 'sto-3g.nwchem')
        ham = Hamiltonian.build(mol, basis)

        self.assertEqual(ham['S'].shape[0], len(basis))
        self.assertTrue(np.allclose(ham['S'], ham['S'].T))
        self.assertLess(ham['V'][0, 1], 0.0)
        self.assertGreater(mol.energy_nuc, 0.0)
        self.assertGreater(ham['Eri'][0, 0, 0, 0], 0.0)


if __name__ == '__main__':
    unittest.main()
