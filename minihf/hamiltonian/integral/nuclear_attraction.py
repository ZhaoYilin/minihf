import numpy as np
import itertools

from minihf.hamiltonian.integral.boys import *
from minihf.basis.gauss import PrimitiveGaussian

class NuclearAttraction(object):
    """The Obara-Saika scheme for nuclear attraction integrals.
    
    This class implements the Obara-Saika recurrence relations for computing
    nuclear attraction integrals between Gaussian-type orbitals.
    
    Methods
    -------
    contracted(self, cga, cgb, mol)
        Evaluate integral over contracted GTOs.
    primitive(self, pga, pgb, C)
        Evaluate integral over primitive GTOs.
    target(self, N, shell_a, shell_b)
        Obtain the target integral in the Obara-Saika recurrence relations.
    recurrence(self, r, N, shell_a, shell_b)
        Run the recurrence for angular momentum on shell_a.
    recurrence_swapped(self, r, N, shell_a, shell_b)
        Run the recurrence for angular momentum on shell_b.
    """
    
    def __init__(self):
        """Initialize the NuclearAttraction instance.
        
        Sets up caches for Boys function values and target integrals.
        """
        self.boys_dict = {}
        self._na_target_cache = {}
        self._boys_cache = {}

    def contracted(self, cga, cgb, mol):
        """Evaluate nuclear attraction integral over contracted GTOs.
        
        Parameters
        ----------
        cga : ContractedGaussian
            The first contracted Gaussian orbital.
        cgb : ContractedGaussian
            The second contracted Gaussian orbital.
        mol : Molecule
            The instance of Molecule class containing nuclear coordinates and charges.
            
        Returns
        -------
        result : float
            The nuclear attraction integral value.
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
                result += -atom.number * contraction * self.primitive(pga, pgb, C)
                
        return result

    def primitive(self, pga, pgb, C):
        """Evaluate nuclear attraction integral over primitive Gaussian orbitals.
        
        Parameters
        ----------
        pga : PrimitiveGaussian
            The first primitive Gaussian orbital.
        pgb : PrimitiveGaussian
            The second primitive Gaussian orbital.
        C : array_like
            The coordinates of the nucleus (3-element array).
            
        Returns
        -------
        result : float
            The nuclear attraction integral value.
        """
        l_total = sum(pga.shell) + sum(pgb.shell)

        a = pga.exponent
        b = pgb.exponent
        p = a + b
        mu = (a * b) / p

        A = np.array(pga.origin)
        B = np.array(pgb.origin)
        P = (a * A + b * B) / p
        C = np.array(C)

        RAB2 = np.dot(A - B, A - B)
        RPC2 = np.dot(P - C, P - C)

        Kab = np.exp(-mu * RAB2)
        x = p * RPC2
        boys_pre_factor = (2.0 * np.pi) / p * Kab

        # Pre-compute Boys function values
        self.boys_dict = {
            l_total: boys_pre_factor * self._boys_cached(l_total, x)
        }
        boys_function = self._boys_cached(l_total, x)

        # Compute Boys function for all lower N using recursion
        for N in range(l_total, 0, -1):
            boys_function = boys_recursion(N, x, boys_function)
            self.boys_dict[N - 1] = boys_pre_factor * boys_function

        # Store recurrence data for later use
        self.p = p
        self.P = P
        self.C = C
        self.Kab = Kab
        self.a = a
        self.b = b
        self.A = A
        self.B = B

        result = self.target(0, pga.shell, pgb.shell)
        return result

    def _boys_cached(self, N, x):
        """Cached Boys function evaluation.
        
        Parameters
        ----------
        N : int
            Order of the Boys function.
        x : float
            Argument of the Boys function.
            
        Returns
        -------
        float
            The Boys function value F_N(x).
        """
        key = (N, x)
        if key not in self._boys_cache:
            self._boys_cache[key] = boys(N, x)
        return self._boys_cache[key]

    def target(self, N, shell_a, shell_b):
        """Obtain the target integral using shell angular momentum tuples.
        
        This is the core recursive function that computes the integral
        for given angular momentum components.
        
        Parameters
        ----------
        N : int
            Order of the Boys function (N = l_total - current angular momentum).
        shell_a : tuple
            Angular momentum components (i, k, m) for the first GTO.
        shell_b : tuple
            Angular momentum components (j, l, n) for the second GTO.
            
        Returns
        -------
        float
            The integral value for the given angular momentum.
        """
        i, k, m = shell_a
        j, l, n = shell_b
        
        # Base case: all angular momentum components are zero
        if i == 0 and k == 0 and m == 0 and j == 0 and l == 0 and n == 0:
            return self.boys_dict[N]
        
        # Apply recurrence along non-zero components
        if i > 0:
            return self.recurrence(0, N, shell_a, shell_b)
        elif k > 0:
            return self.recurrence(1, N, shell_a, shell_b)
        elif m > 0:
            return self.recurrence(2, N, shell_a, shell_b)
        elif j > 0:
            # Swapped case: angular momentum on the second GTO
            return self.recurrence_swapped(0, N, shell_a, shell_b)
        elif l > 0:
            return self.recurrence_swapped(1, N, shell_a, shell_b)
        elif n > 0:
            return self.recurrence_swapped(2, N, shell_a, shell_b)
        else:
            return self.boys_dict[N]

    def recurrence(self, r, N, shell_a, shell_b):
        """Run Obara-Saika recurrence for angular momentum on shell_a.
        
        This implements the recurrence relation:
        [i|j] = X_PA [i-1|j] + (i-1)/(2p) [i-2|j] + j/(2p) [i-1|j-1]
               - X_PC [i-1|j]_{N+1} - (i-1)/(2p) [i-2|j]_{N+1} - j/(2p) [i-1|j-1]_{N+1}
        
        Parameters
        ----------
        r : int
            Cartesian component (0=x, 1=y, 2=z) to apply recurrence on.
        N : int
            Current Boys function order.
        shell_a : tuple
            Angular momentum components for the first GTO.
        shell_b : tuple
            Angular momentum components for the second GTO.
            
        Returns
        -------
        float
            The integral value after recurrence.
        """
        a = self.a
        b = self.b
        p = self.p
        
        A = self.A
        B = self.B
        P = self.P
        C = self.C
        
        i, k, m = shell_a
        j, l, n = shell_b
        
        # Create reduced angular momentum tuples
        if r == 0:  # x-component
            shell_a_1 = (i - 1, k, m)
            shell_a_2 = (i - 2, k, m)
            shell_b_1 = (j - 1, l, n)
        elif r == 1:  # y-component
            shell_a_1 = (i, k - 1, m)
            shell_a_2 = (i, k - 2, m)
            shell_b_1 = (j, l - 1, n)
        else:  # r == 2, z-component
            shell_a_1 = (i, k, m - 1)
            shell_a_2 = (i, k, m - 2)
            shell_b_1 = (j, l, n - 1)
        
        XPA = P[r] - A[r]
        XPC = P[r] - C[r]
        
        term1 = term2 = term3 = term4 = term5 = term6 = 0.0
        
        # term1: (P-A) * [i-1|j]
        if abs(XPA) > 1e-12:
            term1 = XPA * self.target(N, shell_a_1, shell_b)
        
        # term2: (i-1)/(2p) * [i-2|j]
        if shell_a_1[r] > 0:
            term2 = shell_a_1[r] * (1.0 / (2 * p)) * self.target(N, shell_a_2, shell_b)
        
        # term3: j/(2p) * [i-1|j-1]
        if shell_b[r] > 0:
            term3 = shell_b[r] * (1.0 / (2 * p)) * self.target(N, shell_a_1, shell_b_1)
        
        # term4: (P-C) * [i-1|j]_{N+1}
        if abs(XPC) > 1e-12:
            term4 = XPC * self.target(N + 1, shell_a_1, shell_b)
        
        # term5: (i-1)/(2p) * [i-2|j]_{N+1}
        if shell_a_1[r] > 0:
            term5 = shell_a_1[r] * (1.0 / (2 * p)) * self.target(N + 1, shell_a_2, shell_b)
        
        # term6: j/(2p) * [i-1|j-1]_{N+1}
        if shell_b[r] > 0:
            term6 = shell_b[r] * (1.0 / (2 * p)) * self.target(N + 1, shell_a_1, shell_b_1)
        
        return term1 + term2 + term3 - term4 - term5 - term6
    
    def recurrence_swapped(self, r, N, shell_a, shell_b):
        """Run Obara-Saika recurrence for angular momentum on shell_b.
        
        This handles the case when the angular momentum to reduce is on the
        second GTO (shell_b). The recurrence relation is analogous but uses
        X_PB = P - B instead of X_PA.
        
        Parameters
        ----------
        r : int
            Cartesian component (0=x, 1=y, 2=z) to apply recurrence on.
        N : int
            Current Boys function order.
        shell_a : tuple
            Angular momentum components for the first GTO.
        shell_b : tuple
            Angular momentum components for the second GTO.
            
        Returns
        -------
        float
            The integral value after recurrence.
        """
        a = self.a
        b = self.b
        p = self.p
        
        A = self.A
        B = self.B
        P = self.P
        C = self.C
        
        i, k, m = shell_a
        j, l, n = shell_b
        
        # Create reduced angular momentum tuples (swap order)
        if r == 0:  # x-component
            shell_b_1 = (j - 1, l, n)
            shell_b_2 = (j - 2, l, n)
            shell_a_1 = (i - 1, k, m)
        elif r == 1:  # y-component
            shell_b_1 = (j, l - 1, n)
            shell_b_2 = (j, l - 2, n)
            shell_a_1 = (i, k - 1, m)
        else:  # r == 2, z-component
            shell_b_1 = (j, l, n - 1)
            shell_b_2 = (j, l, n - 2)
            shell_a_1 = (i, k, m - 1)
        
        XPB = P[r] - B[r]  # Note: P - B instead of P - A
        XPC = P[r] - C[r]
        
        term1 = term2 = term3 = term4 = term5 = term6 = 0.0
        
        # term1: (P-B) * [i|j-1]
        if abs(XPB) > 1e-12:
            term1 = XPB * self.target(N, shell_a, shell_b_1)
        
        # term2: (j-1)/(2p) * [i|j-2]
        if shell_b_1[r] > 0:
            term2 = shell_b_1[r] * (1.0 / (2 * p)) * self.target(N, shell_a, shell_b_2)
        
        # term3: i/(2p) * [i-1|j-1]
        if shell_a[r] > 0:
            term3 = shell_a[r] * (1.0 / (2 * p)) * self.target(N, shell_a_1, shell_b_1)
        
        # term4: (P-C) * [i|j-1]_{N+1}
        if abs(XPC) > 1e-12:
            term4 = XPC * self.target(N + 1, shell_a, shell_b_1)
        
        # term5: (j-1)/(2p) * [i|j-2]_{N+1}
        if shell_b_1[r] > 0:
            term5 = shell_b_1[r] * (1.0 / (2 * p)) * self.target(N + 1, shell_a, shell_b_2)
        
        # term6: i/(2p) * [i-1|j-1]_{N+1}
        if shell_a[r] > 0:
            term6 = shell_a[r] * (1.0 / (2 * p)) * self.target(N + 1, shell_a_1, shell_b_1)
        
        return term1 + term2 + term3 - term4 - term5 - term6