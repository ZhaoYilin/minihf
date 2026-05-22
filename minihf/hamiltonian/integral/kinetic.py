import numpy as np
import itertools
import math

class Kinetic:
    """The Obara-Saika scheme for kinetic energy integral.

    Methods
    -------
    __init__(self)
        Initialize the instance.

    contracted(self, cga, cgb)
        Evaluates integral over contracted GTOs.

    primitive(self, pga, pgb)
        Evaluates integral over primitive GTOs.

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
        from minihf.hamiltonian.integral.overlap import Overlap
        self.overlap = Overlap()

    def contracted(self, cga, cgb):
        """Evaluate integral over two contracted GTOs.

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
        """Evaluate integral over two primitive gaussian orbitals.

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
        a = pga.exponent
        b = pgb.exponent
        Ax,Ay,Az = pga.origin
        Bx,By,Bz = pgb.origin
        i,k,m = pga.shell
        j,l,n = pgb.shell

        Sij = self.S1d(Ax,i,a,Bx,j,b)
        Skl = self.S1d(Ay,k,a,By,l,b)
        Smn = self.S1d(Az,m,a,Bz,n,b)


        Tij = self.T1d(Ax,i,a,Bx,j,b)
        Tkl = self.T1d(Ay,k,a,By,l,b)
        Tmn = self.T1d(Az,m,a,Bz,n,b)

        Tab = Tij*Skl*Smn+Sij*Tkl*Smn+Sij*Skl*Tmn
        return Tab

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
        
    def T1d(self, A, i, a, B, j, b):
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
        P = (1./p)*(a*A + b*B)

        if i<0 or j<0:
            return 0.
        elif i>0:
            # decrement index i
            return  (P-A)*self.T1d(A,i-1,a,B,j,b) +\
                    1./(2*p)*(i-1)*self.T1d(A,i-2,a,B,j,b) +\
                    1./(2*p)*j*self.T1d(A,i-1,a,B,j-1,b) +\
                    2*a*b/p*self.overlap.S1d(A,i,a,B,j,b) -\
                    b/p*(i-1)*self.overlap.S1d(A,i-2,a,B,j,b)
        elif j>0:
            # decrement index j
            return  (P-B)*self.T1d(A,i,a,B,j-1,b) +\
                    1./(2*p)*i*self.T1d(A,i-1,a,B,j-1,b) +\
                    1./(2*p)*(j-1)*self.T1d(A,i,a,B,j-2,b) +\
                    2*a*b/p*self.overlap.S1d(A,i,a,B,j,b) -\
                    a/p*(j-1)*self.overlap.S1d(A,i,a,B,j-2,b)
        else:
            # boundary conditon i==j==0
            T00 = (a-2*a**2*((P-A)**2+1./(2*p)))*self.overlap.S1d(A,0,a,B,0,b)
            return T00