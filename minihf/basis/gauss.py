import itertools
import math
import numpy as np
from scipy.special import factorial2 as fact2

__all__ = ['PrimitiveGaussian','ContractedGaussian']

class PrimitiveGaussian(object):
    """Primitive Gaussian functions.
    
    Attributes
    ----------
    coefficient : float
        Primitive Gaussian coefficient.

    origin : List[float,float,float]
        Coordinate of the nuclei.

    shell : List[int,int,int]
        Angular momentum.

    exponent : float
        Primitive Gaussian exponent.
    
    Properties
    ----------
    norm: float
        Normalization factor.
    
    Methods
    -------
    __init__(self,type,origin,n,shell,coef,exp)
        Initialize the instance.

    __call__(self,x,y,z)
        Call the instance as function.
    """
    def __init__(self,coefficient,origin,shell,exponent):
        """Initialize the instance.

        Parameters
        ----------
        coefficient : float
            Primitive Gaussian coefficient.

        origin : List[float,float,float]
            Coordinate of the nuclei.

        shell : List[int,int,int]
            Angular momentum.

        exponent : float
            Primitive Gaussian exponent.
        """
        self.coefficient = coefficient
        self.origin = origin
        self.shell = shell
        self.exponent  = exponent
        self.norm_memo = None

    def __call__(self,x,y,z):
        """Call the instance as function.
        
        Parameters
        ----------
        x : float
            Coordinate of the point.
        y : float
            Coordinate of the point.
        z : float
            Coordinate of the point.
        """
        X = x-self.origin[0]
        Y = y-self.origin[1]
        Z = z-self.origin[2]
        rr = X**2+Y**2+Z**2
        result = np.power(X,self.shell[0])*\
            np.power(Y,self.shell[1])*\
            np.power(Z,self.shell[2])*\
            np.exp(-self.exponent*rr)
        return result

    @property
    def norm(self):
        """Normalization factors. 
        
        Return
        ------
        norm : list
            Normalization factors
        """
        if self.norm_memo is None:
            i,j,k = self.shell
            term1 = (2*self.exponent/math.pi)**(3/2)
            term2 = (4*self.exponent)**(i+j+k)
            
            # Handle cases where angular momentum quantum numbers are 0
            term3_components = []
            for l in [i, j, k]:
                if 2*l-1 >= 0:
                    term3_components.append(fact2(2*l-1))
                else:
                    # For l=0, fact2(-1) is not defined, use 1.0 as default
                    term3_components.append(1.0)
            
            term3 = term3_components[0] * term3_components[1] * term3_components[2]
            
            # Avoid division by zero
            if term3 == 0:
                self.norm_memo = 0.0
            else:
                self.norm_memo = math.sqrt(term1*term2/term3)
        return self.norm_memo

class ContractedGaussian(object):
    """Predetermined linear combination of radial parts of GTOs
    Atomic orbtial represented by Contracted Gaussian functions.
    
    Attributes
    ----------
    coefficients : List of float
        Primitive Gaussian coefficient.

    origin : List[float,float,float]
        Coordinate of the nuclei.

    shell : List[int,int,int]
        Angular momentum.

    exponents : List of float
        Primitive Gaussian exponent.
    
    Properties
    ----------
    norm: list
        Normalization factor.
    
    Methods
    -------
    __init__(self,type,origin,n,shell=(),coefs=[],exps=[])
        Initialize the instance.

    """
    def __init__(self,coefficients=[],origin=[],shell=[],exponents=[]):
        """Initialize the instance.

        Parameters
        ----------
        coefficients : List of float
            Primitive Gaussian coefficient.

        origin : List[float,float,float]
            Coordinate of the nuclei.

        shell : List[int,int,int]
            Angular momentum.

        exponents : List of float
            Primitive Gaussian exponent.
        """
        self.coefficients = coefficients
        self.origin = origin
        self.shell = shell
        self.exponents  = exponents
        self.norm_memo = None

    def __call__(self,x,y,z):
        print("ContractedGaussian called with x={}, y={}, z={}".format(x,y,z))
        X = x-self.origin[0]
        Y = y-self.origin[1]
        Z = z-self.origin[2]
        rr = X**2+Y**2+Z**2

        cg = 0.0
        for coef, exp in zip(self.coefficients, self.exponents):
            cg += coef * np.power(X, self.shell[0]) * \
                np.power(Y, self.shell[1]) * \
                np.power(Z, self.shell[2]) * \
                np.exp(-exp * rr)
        return cg

    def __neg__(self):
        coefficients = [-coefficient for coefficient in self.coefficients]
        origin = self.origin
        shell = self.shell
        exponents = self.exponents
        new_CGTO = self.__class__(coefficients,origin,shell,exponents)
        return new_CGTO

    def __add__(self,other):
        if self.parallel(other):
            coefficients = [a+b for a,b in zip(self.coefficients,other.coefficients)]
            origin = self.origin
            shell = self.shell
            exponents = self.exponents
            new_CGTO = self.__class__(coefficients,origin,shell,exponents)
            return new_CGTO
        else:
            return NotImplemented

    def __sub__(self,other):
        if self.parallel(other):
            coefficients = [a-b for a,b in zip(self.coefficients,other.coefficients)]
            origin = self.origin
            shell = self.shell
            exponents = self.exponents
            new_CGTO = self.__class__(coefficients,origin,shell,exponents)
            return new_CGTO
        else:
            return NotImplemented

    def __mul__(self,other):
        if isinstance(other,(int,float)):
            coefficients = [other*coefficient for coefficient in self.coefficients]
            origin = self.origin
            shell = self.shell
            exponents = self.exponents
            new_CGTO = self.__class__(coefficients,origin,shell,exponents)
            return new_CGTO
        elif isinstance(other,self.__class__):
            return NotImplemented
        else:
            return NotImplemented

    def __rmul__(self,other):
        return(self.__mul__(other))

    def __truediv__(self,other):
        if isinstance(other,(int,float)):
            coefficients = [coefficient/other for coefficient in self.coefficients]
            origin = self.origin
            shell = self.shell
            exponents = self.exponents
            new_CGTO = self.__class__(coefficients,origin,shell,exponents)
            return new_CGTO
        elif isinstance(other,self.__class__):
            return NotImplemented
        else:
            return NotImplemented

    @property
    def norm(self):
        """Normalization factors. 
        
        Return
        ------
        norm : list
            Normalization factors
        """
        if self.norm_memo is None:
            l, m, n = self.shell

            result = 0.0
            for pga, pgb in itertools.product(self.expansion, repeat=2):
                a = pga.exponent
                b = pgb.exponent
                ca = pga.coefficient
                cb = pgb.coefficient
                na = pga.norm
                nb = pgb.norm

                term1 = fact2(2 * l - 1) * fact2(2 * m - 1) * fact2(2 * n - 1)
                term2 = (math.pi / (a + b)) ** (3 / 2)
                term3 = (2 * (a + b)) ** (l + m + n)
                result += (ca * cb * na * nb * term1 * term2) / term3
            self.norm_memo = 1 / math.sqrt(result)

        return self.norm_memo

    @property
    def expansion(self):
        """Normalization factors. 
        
        Return
        ------
        primitives : list
            Normalization factors
        """
        n = len(self.coefficients)
        primitives = []
        for i in range(n):
            coefficient = self.coefficients[i]
            exponent = self.exponents[i]
            pg = PrimitiveGaussian(coefficient,self.origin,self.shell,exponent)
            primitives.append(pg)
        return primitives

    def parallel(self,other):
        if isinstance(other,self.__class__):
            attributes_self = [self.origin,self.shell,self.exponents]
            attributes_other = [other.origin,other.shell,other.exponents]
            if attributes_self == attributes_other:
                return True
            else:
                return False
        else:
            return False