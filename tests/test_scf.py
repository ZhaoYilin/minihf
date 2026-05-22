import unittest
import numpy as np
from minihf.scf.rhf import rhf
from minihf.scf.uhf import uhf
from minihf.scf.rohf import rohf
from minihf.scf.ghf import ghf


class TestScfSolvers(unittest.TestCase):
    def setUp(self):
        self.S = np.eye(2)
        self.Hcore = np.array([[1.0, 0.0], [0.0, 2.0]])
        self.Eri = np.zeros((2, 2, 2, 2))

    def test_rhf_energy_and_shapes(self):
        result = rhf(self.S, self.Hcore, self.Eri, nocc=1, energy_nuc=0.0, I_thld=10, E_thld=1e-10, D_thld=1e-10)
        self.assertAlmostEqual(result["energy_elec"], 2.0)
        self.assertEqual(result["mo_coeff"].shape, (2, 2))
        self.assertEqual(result["rdm1"].shape, (2, 2))

    def test_uhf_energy_and_spin_blocks(self):
        result = uhf(self.S, self.Hcore, self.Eri, nalpha=1, nbeta=1, energy_nuc=0.0, I_thld=10, E_thld=1e-10, D_thld=1e-10)
        self.assertAlmostEqual(result["energy_elec"], 2.0)
        self.assertEqual(len(result["mo_coeff"]), 2)
        self.assertEqual(result["rdm1"][0].shape, (2, 2))
        self.assertEqual(result["rdm1"][1].shape, (2, 2))

    def test_rohf_energy_and_shapes(self):
        result = rohf(self.S, self.Hcore, self.Eri, nalpha=1, nbeta=1, energy_nuc=0.0, I_thld=10, E_thld=1e-10, D_thld=1e-10)
        self.assertAlmostEqual(result["energy_elec"], 2.0)
        self.assertEqual(result["mo_coeff"].shape, (2, 2))
        self.assertEqual(len(result["rdm1"]), 2)
        self.assertEqual(result["rdm1"][0].shape, (2, 2))
        self.assertEqual(result["rdm1"][1].shape, (2, 2))

    def test_ghf_energy_and_spin_orbital_size(self):
        result = ghf(self.S, self.Hcore, self.Eri, nalpha=1, nbeta=1, energy_nuc=0.0, I_thld=10, E_thld=1e-10, D_thld=1e-10)
        self.assertAlmostEqual(result["energy_elec"], 2.0)
        self.assertEqual(result["mo_coeff"].shape, (4, 4))
        self.assertEqual(result["rdm1"].shape, (4, 4))


if __name__ == "__main__":
    unittest.main()