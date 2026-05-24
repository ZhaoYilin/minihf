import numpy as np
import itertools

from math import pi, sqrt, exp
from minihf.hamiltonian.integral.boys import *
from minihf.basis.gauss import PrimitiveGaussian

class NuclearAttraction:
    """The Obara-Saika scheme for nuclear attraction integral.

    Methods
    -------
    contracted(self, cga, cgb)
        Evaluate integral over contracted GTOs.

    primitive(self, pga, pgb)
        Evaluate integral over primitive GTOs.

    target(self, r, pga, pgb)
        Obtain the target integral in the Obara-Saika recurrence relations.

    recurrence(self, r, pga, pgb, pga_1, pga_2, pgb_1)
        Run the recurrence.

    new_gaussian(self, r, pga, pgb):
        Generate all the new primitive GTOs in the Obara-Saika recurrence relations.
    """
    def __init__(self):
        """Initialize the instance.
        """
        self.boys_dict = {}

    def contracted(self, cga, cgb, mol):
        """Evaluate integral over contracted GTOs.

        Parameters
        ----------
        cga: ContractedGaussian
            The first contracted gaussian orbital.

        cgb: ContractedGaussian
            The second contracted gaussian orbital.

        mol: Molecule
            The instance of Molecule class.

        Return
        ------
        result : float
            Integral value.
        """
        primitives_a = cga.expansion
        primitives_b = cgb.expansion
        result = 0.0
        for pga, pgb in itertools.product(primitives_a, primitives_b):
            ca = pga.coefficient
            cb = pgb.coefficient
            na = pga.norm
            nb = pgb.norm
            contraction = ca * cb * na * nb
            for atom in mol:
                C = atom.coords
                result += -atom.number*contraction*self.primitive(pga,pgb,C)

        return result

    def primitive(self, pga, pgb, C):
        """Evaluate integral over primitive gaussian orbitals.

        Parameters
        ----------
        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.
    
        C: List[float,float,float]
            Coordinate of nuclei.

        Return
        ------
        result : float
            Integral value.
        """
        l_total = sum(pga.shell) + sum(pgb.shell)

        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a*b)/(a+b)

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p

        RAB = np.linalg.norm(A-B)
        RPC = np.linalg.norm(P-C)

        Kab = exp(-mu*RAB**2)

        self.p = p
        self.mu = mu
        self.P = P
        self.C = C
        self.Kab = Kab

        # Build boys function F_{N}(x)
        N = l_total
        x = p*RPC**2
        boys_pre_factor = (2*pi)/p*Kab
        boys_function = boys(l_total, x)
        Theta_N_000000 = boys_pre_factor * boys_function
        self.boys_dict = {l_total: Theta_N_000000}

        while N >= 1:
            boys_function = boys_recursion(N, x, boys_function)
            N -= 1
            Theta_N_000000 = boys_pre_factor * boys_function
            self.boys_dict[N] = Theta_N_000000

        result = self.target(0, pga, pgb)
        return result

    def target(self, N, pga, pgb):
        """Obtain the target integral in the Obara-Saika recurrence relations.

        Parameters
        ----------
        N : int
            Order of the boys function F_{N}(x).

        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.
        """
        shell_A = pga.shell
        shell_B = pgb.shell

        if shell_A[0] > 0:
            return self.recurrence(0, N, pga, pgb, *self.new_gaussian(0, pga, pgb))
        elif shell_A[1] > 0:
            return self.recurrence(1, N, pga, pgb, *self.new_gaussian(1, pga, pgb))
        elif shell_A[2] > 0:
            return self.recurrence(2, N, pga, pgb, *self.new_gaussian(2, pga, pgb))
        elif shell_B[0] > 0:
            return self.recurrence(0, N, pgb, pga, *self.new_gaussian(0, pgb, pga))
        elif shell_B[1] > 0:
            return self.recurrence(1, N, pgb, pga, *self.new_gaussian(1, pgb, pga))
        elif shell_B[2] > 0:
            return self.recurrence(2, N, pgb, pga, *self.new_gaussian(2, pgb, pga))
        else:
            return self.boys_dict[N]

    def recurrence(self, r, N, pga, pgb, pga_1, pga_2, pgb_1):
        """Run the recurrence.

        Parameters
        ----------
        r : int
            Cartesian direction index 0, 1, 2. 

        N : int
            Order of the boys function F_{N}(x).

        pga : PrimitiveGaussian
            The first primitive gaussian orbital with angular momentum i.

        pgb : PrimitiveGaussian
            The second primitive gaussian orbital with angular momentum j.

        pga_1 : PrimitiveGaussian
            The first primitive gaussian orbital with angular momentum i-1.

        pga_2 : PrimitiveGaussian
            The first primitive gaussian orbital with angular momentum i-2.

        pgb_1 : PrimitiveGaussian
            The second primitive gaussian orbital with angular momentum j-1.

        Return
        ------
        result : float
            Integral value.
        """
        a = pga.exponent
        b = pgb.exponent
        p = a+b

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a*A+b*B)/p
        C = self.C

        XPA = np.array(P) - np.array(A)
        XPC = np.array(P) - np.array(C)

        term1 = term2 = term3 = term4 = term5 = term6 = 0
        if np.array_equal(P,A) is False:
            term1 = XPA[r] * self.target(N, pga_1, pgb)
        if pga_1.shell[r] > 0:
            term2 = pga_1.shell[r] * (1 / (2 * p)) * self.target(N, pga_2, pgb)
        if pgb.shell[r] > 0:
            term3 = pgb.shell[r] * (1 / (2 * p)) * self.target(N, pga_1, pgb_1)
        if np.array_equal(P,C) is False:
            term4 = XPC[r] * self.target(N+1, pga_1, pgb)
        if pga_1.shell[r] > 0:
            term5 = pga_1.shell[r] * (1 / (2 * p)) * self.target(N+1, pga_2, pgb)
        if pgb.shell[r] > 0:
            term6 = pgb.shell[r] * (1 / (2 * p)) * self.target(N+1, pga_1, pgb_1)

        result = term1+term2+term3-term4-term5-term6
        return result

    def new_gaussian(self, r, pga, pgb):
        """Generate all the new primitive GTOs in the Obara-Saika recurrence relations.

        Parameters
        ----------
        r : int
            Cartesian direction index 0, 1, 2. 

        pga : PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The second primitive gaussian orbital.

        Return
        ------
        result : Tuple(pg, pg, pg)
            Tuple of 3 PrimitiveGaussian orbital instance. 
        """
        ca = pga.coefficient
        cb = pgb.coefficient

        a = pga.exponent
        b = pgb.exponent

        A = pga.origin
        B = pgb.origin

        i,k,m = pga.shell
        j,l,n = pgb.shell

        if r == 0:
            pga_i_1 = PrimitiveGaussian(ca, A, (i - 1, k, m), a)
            pga_i_2 = PrimitiveGaussian(ca, A, (i - 2, k, m), a)
            pgb_j_1 = PrimitiveGaussian(cb, B, (j - 1, l, n), b)
            return pga_i_1, pga_i_2, pgb_j_1
        elif r == 1:
            pga_k_1 = PrimitiveGaussian(ca, A, (i, k - 1, m), a)
            pga_k_2 = PrimitiveGaussian(ca, A, (i, k - 2, m), a)
            pgb_l_1 = PrimitiveGaussian(cb, B, (j, l - 1, n), b)
            return pga_k_1, pga_k_2, pgb_l_1
        elif r == 2:
            pga_m_1 = PrimitiveGaussian(ca, A, (i, k, m - 1), a)
            pga_m_2 = PrimitiveGaussian(ca, A, (i, k, m - 2), a)
            pgb_n_1 = PrimitiveGaussian(cb, B, (j, l, n - 1), b)
            return pga_m_1, pga_m_2, pgb_n_1