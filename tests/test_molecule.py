import unittest
import numpy as np
from minihf.molecule.molecule import Molecule


class TestMolecule(unittest.TestCase):

    def test_build_and_electron_count(self):
        mol = Molecule.build(["H 0 0 0", "H 0 0 0.74"], charge=0, multiplicity=1)
        self.assertEqual(len(mol), 2)
        self.assertEqual(mol.nelec, 2)
        self.assertEqual(mol.nalpha, 1)
        self.assertEqual(mol.nbeta, 1)
        self.assertAlmostEqual(mol.bond_length(0, 1), 0.74, places=7)

    def test_center_of_mass(self):
        mol = Molecule.build(["H 0 0 0", "H 0 0 1.0"], charge=0, multiplicity=1)
        com = mol.center_of_mass
        np.testing.assert_allclose(com, np.array([0.0, 0.0, 0.5]), atol=1e-8)

    def test_bond_angle(self):
        mol = Molecule.build([
            "O 0 0 0",
            "H 0.757 0.586 0",
            "H -0.757 0.586 0"
        ], charge=0, multiplicity=1)
        angle = mol.bond_angle(1, 0, 2)
        self.assertAlmostEqual(angle, 104.5, places=1)

    def test_dihedral_angle(self):
        mol = Molecule.build([
            "C 0 0 0",
            "H 1 0 0",
            "H 1 1 0",
            "H 1 1 1"
        ], charge=0, multiplicity=1)
        dihedral = mol.dihedral_angle(0, 1, 2, 3)
        self.assertAlmostEqual(abs(dihedral), 90.0, places=6)

    def test_energy_nuc_positive(self):
        mol = Molecule.build(["H 0 0 0", "H 0 0 0.74"], charge=0, multiplicity=1)
        self.assertGreater(mol.energy_nuc, 0.0)


if __name__ == "__main__":
    unittest.main()
