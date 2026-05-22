import numpy as np
import itertools
import math

class Overlap:
    """The Obara-Saika scheme for overlap integral.

    Methods
    -------
    contracted(self, cga, cgb)
        Evaluate integral over contracted GTOs.

    primitive(self, pga, pgb)
        Evaluate integral over primitive GTOs.

    S1d(self, r, pga, pgb)
        Obtain the target integral in the Obara-Saika recurrence relations.
    """
    def contracted(self, cga, cgb):
        """Evaluate integral over contracted GTOs.

        Parameters
        ----------
        cga: ContractedGaussian
            The first contracted gaussian orbital.

        cgb: ContractedGaussian
            The second contracted gaussian orbital.

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
            result += contraction*self.primitive(pga,pgb)
        return result

    def primitive(self, pga, pgb):
        """Evaluate integral over primitive gaussian orbitals.

        Parameters
        ----------
        pga: PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb: PrimitiveGaussian
            The second primitive gaussian orbital.
    
        Return
        ------
        result : float
            Integral value.
        """
        result = 1
        a = pga.exponent
        b = pgb.exponent
        origin_A = pga.origin
        origin_B = pgb.origin
        shell_A = pga.shell
        shell_B = pgb.shell
        for r in range(3):
            result *= self.S1d(origin_A[r],shell_A[r],a,origin_B[r],shell_B[r],b)
        return result

    def S1d(self, A, i, a, B, j, b):
        """Obtain the target integral in the Obara-Saika recurrence relations.

        Parameters
        ----------
        A : float
            Origin of the basis for one direction.

        i : int
            Angular momentum for one direction.

        a : float
            Primitive Gaussian exponent for one direction.

        B : float
            Origin of the basis for one direction.

        j : int
            Angular momentum for one direction.

        b : float
            Primitive Gaussian exponent for one direction.

        Return
        ------
        result : float
            Integral value for one direction.
        """
        p = a + b
        mu = (a*b)/(a+b)
        P = (1./p)*(a*A + b*B)

        if i<0 or j<0:
            return 0.
        elif i>0:
            # decrement index i.
            return  (P-A)*self.S1d(A, i-1, a, B, j, b) +\
                    1./(2*p)*(i-1)*self.S1d(A, i-2, a, B, j, b) +\
                    1./(2*p)*j*self.S1d(A, i-1, a, B, j-1, b)
        elif j>0:
            # decrement index j.
            return  (P-B)*self.S1d(A, i, a, B, j-1, b) +\
                    1./(2*p)*i*self.S1d(A, i-1, a, B, j-1, b) +\
                    1./(2*p)*(j-1)*self.S1d(A, i, a, B, j-2, b)
        else: 
            # boundary condition i==j==0.
            S00 = math.sqrt(np.pi / p) * math.exp(-mu * (A - B)**2)
            return S00