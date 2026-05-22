import unittest
from minihf.basis.basis_set import BasisSet
from minihf.basis.gauss import ContractedGaussian
from minihf.molecule.molecule import Molecule


class TestBasisSet(unittest.TestCase):

    def test_contract_gaussian_arithmetic(self):
        cg = ContractedGaussian([1.0, 0.5], [0.0, 0.0, 0.0], (0, 0, 0), [1.0, 0.5])
        value = cg(0.0, 0.0, 0.0)
        self.assertAlmostEqual(value, 1.5)

        neg = -cg
        self.assertEqual(neg.coefficients, [-1.0, -0.5])

        scaled = cg * 2
        self.assertEqual(scaled.coefficients, [2.0, 1.0])

        divided = cg / 2
        self.assertEqual(divided.coefficients, [0.5, 0.25])

    def test_basis_set_append_type_error(self):
        basis_set = BasisSet()
        with self.assertRaises(TypeError):
            basis_set.append("not a basis")

    def test_basis_set_build_sto3g(self):
        mol = Molecule.build(["H 0 0 0"], charge=0, multiplicity=1)
        basis_set = BasisSet.build(mol, 'sto-3g.nwchem')
        self.assertGreater(len(basis_set), 0)
        self.assertTrue(all(isinstance(basis, ContractedGaussian) for basis in basis_set))


if __name__ == "__main__":
    unittest.main()
