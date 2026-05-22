import itertools
import numpy as np
import math

from minihf.hamiltonian.integral.boys import *
from minihf.basis.gauss import PrimitiveGaussian

    
class ElectronRepulsion:
    """The Obara-Saika scheme for three-dimensional nuclear attraction integral over
    primitive Gaussian orbitals.

    Attributes
    ----------

    Methods
    -------
    __init__(self)
        Initialize the instance.

    contracted(self, cga, cgb, cgc, cgd)
        Evaluate integral over contracted GTOs.

    target(self, N, pga, pgb, pgc, pgd)
        Obtain the target integral in the Obara-Saika recurrence relations.

    recurrence(self, r, N, pga, pgb, pgc, pgd, pga_1, pga_2, pgb_1, pgc_1, pgd_1)
        Run the recurrence.

    new_gaussian(self, r, pga, pgb, pgc, pgd)
        Generate all the new primitive GTOs in the Obara-Saika recurrence relations.
    """
    def __init__(self):
        """Initialize the instance.
        """
        self.alpha = 0
        self.R = ()
        self.boys_dict = {}
        self._eri_target_cache = {}

    def contracted(self, cga, cgb, cgc, cgd):
        """Evaluate integral over contracted GTOs.

        Parameters
        ----------
        cga: ContractedGaussian
            The first contracted gaussian orbital.

        cgb: ContractedGaussian
            The second contracted gaussian orbital.

        cgc: ContractedGaussian
            The third contracted gaussian orbital.

        cgd: ContractedGaussian
            The fourth contracted gaussian orbital.

        Return
        ------
        result : float
            Integral value.
        """
        l_total = sum(cga.shell) + sum(cgb.shell) + sum(cgc.shell) + sum(cgd.shell)

        A = np.array(cga.origin)
        B = np.array(cgb.origin)
        C = np.array(cgc.origin)
        D = np.array(cgd.origin)
        RAB2 = np.dot(A - B, A - B)
        RCD2 = np.dot(C - D, C - D)

        primitives_a = cga.expansion
        primitives_b = cgb.expansion
        primitives_c = cgc.expansion
        primitives_d = cgd.expansion

        result = 0.0
        for pga, pgb, pgc, pgd in itertools.product(primitives_a, primitives_b, primitives_c, primitives_d):
            ca = pga.coefficient
            cb = pgb.coefficient
            cc = pgc.coefficient
            cd = pgd.coefficient
            na = pga.norm
            nb = pgb.norm
            nc = pgc.norm
            nd = pgd.norm
            contraction = ca * cb * cc * cd * na * nb * nc * nd

            a = pga.exponent
            b = pgb.exponent
            c = pgc.exponent
            d = pgd.exponent
            p = a + b
            q = c + d
            mu = (a*b)/(a+b)
            nu = (c*d)/(c+d)
            self.alpha = (p * q) / (p + q)

            P = (a*A+b*B)/(a+b)
            Q = (c*C+d*D)/(c+d)
            self.R = (p*P+q*Q)/(p+q)
            RPQ2 = np.dot(P - Q, P - Q)

            # Build boys function F_{N}(x)
            Kab = math.exp(-mu * RAB2)
            Kcd = math.exp(-nu * RCD2)
            boys_pre_factor = (2*math.pi**(5/2))/(p*q*math.sqrt(p+q))*Kab*Kcd
            N = l_total
            x = self.alpha*RPQ2
            boys_function = self._boys_cached(l_total, x)
            Theta_N_0000_0000_0000 = boys_pre_factor * boys_function
            self.boys_dict = {l_total: Theta_N_0000_0000_0000}

            for _ in range(l_total, 0, -1):
                boys_function = boys_recursion(N, x, boys_function)
                N -= 1
                self.boys_dict[N] = boys_pre_factor * boys_function
            result += contraction * self.target(0, pga, pgb, pgc, pgd)
        return result

    @staticmethod
    @lru_cache(maxsize=65536)
    def _boys_cached(N, x):
        return boys(N, x)

    def target(self, N, pga, pgb, pgc, pgd):
        """Obtain the target integral in the Obara-Saika recurrence relations."""
        shell_a = pga.shell
        shell_b = pgb.shell
        shell_c = pgc.shell
        shell_d = pgd.shell
        
        key = (
            N,
            shell_a,
            shell_b,
            shell_c,
            shell_d,
            pga.exponent,
            pgb.exponent,
            pgc.exponent,
            pgd.exponent,
            tuple(pga.origin),
            tuple(pgb.origin),
            tuple(pgc.origin),
            tuple(pgd.origin),
            self.alpha,
            tuple(self.R),
        )
        if key in self._eri_target_cache:
            return self._eri_target_cache[key]

        sa0, sa1, sa2 = shell_a
        sb0, sb1, sb2 = shell_b
        sc0, sc1, sc2 = shell_c
        sd0, sd1, sd2 = shell_d

        if sa0 > 0:
            result = self.recurrence(0, N, pga, pgb, pgc, pgd, *self.new_gaussian(0, pga, pgb, pgc, pgd))
        elif sa1 > 0:
            result = self.recurrence(1, N, pga, pgb, pgc, pgd, *self.new_gaussian(1, pga, pgb, pgc, pgd))
        elif sa2 > 0:
            result = self.recurrence(2, N, pga, pgb, pgc, pgd, *self.new_gaussian(2, pga, pgb, pgc, pgd))
        elif sb0 > 0:
            result = self.recurrence(0, N, pgb, pga, pgd, pgc, *self.new_gaussian(0, pgb, pga, pgd, pgc))
        elif sb1 > 0:
            result = self.recurrence(1, N, pgb, pga, pgd, pgc, *self.new_gaussian(1, pgb, pga, pgd, pgc))
        elif sb2 > 0:
            result = self.recurrence(2, N, pgb, pga, pgd, pgc, *self.new_gaussian(2, pgb, pga, pgd, pgc))
        elif sc0 > 0:
            result = self.recurrence(0, N, pgc, pgd, pga, pgb, *self.new_gaussian(0, pgc, pgd, pga, pgb))
        elif sc1 > 0:
            result = self.recurrence(1, N, pgc, pgd, pga, pgb, *self.new_gaussian(1, pgc, pgd, pga, pgb))
        elif sc2 > 0:
            result = self.recurrence(2, N, pgc, pgd, pga, pgb, *self.new_gaussian(2, pgc, pgd, pga, pgb))
        elif sd0 > 0:
            result = self.recurrence(0, N, pgd, pgc, pgb, pga, *self.new_gaussian(0, pgd, pgc, pgb, pga))
        elif sd1 > 0:
            result = self.recurrence(1, N, pgd, pgc, pgb, pga, *self.new_gaussian(1, pgd, pgc, pgb, pga))
        elif sd2 > 0:
            result = self.recurrence(2, N, pgd, pgc, pgb, pga, *self.new_gaussian(2, pgd, pgc, pgb, pga))
        else:
            result = self.boys_dict[N]

        self._eri_target_cache[key] = result
        return result

    def recurrence(self, r, N, pga, pgb, pgc, pgd, pga_1, pga_2, pgb_1, pgc_1, pgd_1):
        """Run the recurrence."""
        # Precompute constants
        a = pga.exponent
        b = pgb.exponent
        c = pgc.exponent
        d = pgd.exponent
        p = a + b
        q = c + d
        alpha = self.alpha
        
        A = pga.origin
        B = pgb.origin
        C = pgc.origin
        D = pgd.origin
        
        # Compute P and Q efficiently
        Px = (a*A[0] + b*B[0]) / p
        Py = (a*A[1] + b*B[1]) / p
        Pz = (a*A[2] + b*B[2]) / p
        Qx = (c*C[0] + d*D[0]) / q
        Qy = (c*C[1] + d*D[1]) / q
        Qz = (c*C[2] + d*D[2]) / q
        
        XPA_r = (Px - A[0], Py - A[1], Pz - A[2])[r]
        XPQ_r = (Px - Qx, Py - Qy, Pz - Qz)[r]
        
        # Precompute common terms
        two_p = 2.0 * p
        two_p_sq = two_p * p
        two_pq = 2.0 * (p + q)
        alpha_over_p = alpha / p
        alpha_over_2p = alpha / two_p
        alpha_over_2p_sq = alpha / two_p_sq
        
        term1 = term2 = term3 = term4 = term5 = term6 = term7 = term8 = 0.0
        target = self.target
        
        if XPA_r != 0:
            term1 = XPA_r * target(N, pga_1, pgb, pgc, pgd)
        if XPQ_r != 0:
            term2 = alpha_over_p * XPQ_r * target(N+1, pga_1, pgb, pgc, pgd)
        
        shell_pga1_r = pga_1.shell[r]
        if shell_pga1_r > 0:
            term3 = shell_pga1_r * (0.5 / p) * target(N, pga_2, pgb, pgc, pgd)
            term4 = shell_pga1_r * alpha_over_2p_sq * target(N+1, pga_2, pgb, pgc, pgd)
        
        shell_pgb_r = pgb.shell[r]
        if shell_pgb_r > 0:
            term5 = shell_pgb_r * (0.5 / p) * target(N, pga_1, pgb_1, pgc, pgd)
            term6 = shell_pgb_r * alpha_over_2p_sq * target(N+1, pga_1, pgb_1, pgc, pgd)
        
        shell_pgc_r = pgc.shell[r]
        if shell_pgc_r > 0:
            term7 = shell_pgc_r * (0.5 / (p + q)) * target(N+1, pga_1, pgb, pgc_1, pgd)
        
        shell_pgd_r = pgd.shell[r]
        if shell_pgd_r > 0:
            term8 = shell_pgd_r * (0.5 / (p + q)) * target(N+1, pga_1, pgb, pgc, pgd_1)

        return term1 - term2 + term3 - term4 + term5 - term6 + term7 + term8

    def new_gaussian(self, r, pga, pgb, pgc, pgd):
        """Generate all the new primitive GTOs in the Obara-Saika recurrence relations.

        Parameters
        ----------
        r : int
            Cartesian index 0, 1, 2. 

        N : int
            Order of the boys function F_{N}(x).

        pga : PrimitiveGaussian
            The first primitive gaussian orbital.

        pgb : PrimitiveGaussian
            The second primitive gaussian orbital.

        pgc : PrimitiveGaussian
            The third primitive gaussian orbital.

        pgd : PrimitiveGaussian
            The fourth primitive gaussian orbital.

        Return
        ------
        result : Tuple(pg,pg,pg,pg,pg)
            Tuple of 5 PrimitiveGaussian orbital instance. 
        """
        ca = pga.coefficient
        cb = pgb.coefficient
        cc = pgc.coefficient
        cd = pgd.coefficient

        a = pga.exponent
        b = pgb.exponent
        c = pgc.exponent
        d = pgd.exponent

        A = pga.origin
        B = pgb.origin
        C = pgc.origin
        D = pgd.origin

        ix,iy,iz = pga.shell
        jx,jy,jz = pgb.shell
        kx,ky,kz = pgc.shell
        lx,ly,lz = pgd.shell

        if r == 0:
            pga_1 = PrimitiveGaussian(ca, A, (ix - 1, iy, iz), a)
            pga_2 = PrimitiveGaussian(ca, A, (ix - 2, iy, iz), a)
            pgb_1 = PrimitiveGaussian(cb, B, (jx - 1, jy, jz), b)
            pgc_1 = PrimitiveGaussian(cc, C, (kx - 1, ky, kz), c)
            pgd_1 = PrimitiveGaussian(cd, D, (lx - 1, ly, lz), d)
            return pga_1, pga_2, pgb_1, pgc_1, pgd_1
        elif r == 1:
            pga_1 = PrimitiveGaussian(ca, A, (ix, iy-1, iz), a)
            pga_2 = PrimitiveGaussian(ca, A, (ix, iy-2, iz), a)
            pgb_1 = PrimitiveGaussian(cb, B, (jx, jy-1, jz), b)
            pgc_1 = PrimitiveGaussian(cc, C, (kx, ky-1, kz), c)
            pgd_1 = PrimitiveGaussian(cd, D, (lx, ly-1, lz), d)
            return pga_1, pga_2, pgb_1, pgc_1, pgd_1
        elif r == 2:
            pga_1 = PrimitiveGaussian(ca, A, (ix, iy, iz-1), a)
            pga_2 = PrimitiveGaussian(ca, A, (ix, iy, iz-2), a)
            pgb_1 = PrimitiveGaussian(cb, B, (jx, jy, jz-1), b)
            pgc_1 = PrimitiveGaussian(cc, C, (kx, ky, kz-1), c)
            pgd_1 = PrimitiveGaussian(cd, D, (lx, ly, lz-1), d)
            return pga_1, pga_2, pgb_1, pgc_1, pgd_1