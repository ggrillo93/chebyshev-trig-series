import numpy as np
from scipy.fft import dct, dst, idct, idst
from nfft import nfft_adjoint
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
from findiff import FinDiff
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

class TrigExpansion:
    """ useFFT = True is much faster for evenly spaced samples and slower up to nTheta ~ 50 for unevenly spaced samples """
    def __init__(self, coeffs=None, thetas=None, vals=None, deg=None, useFFT=True, trig_type='cos', check_periodicity = True):
        assert trig_type in ['cos', 'sin'], "trig_type must be 'cos' or 'sin'"
        self.trig_type = trig_type
        self.coeffs, self.deg = self._initialize_coeffs(coeffs, thetas, vals, deg, useFFT, check_periodicity)

    @classmethod
    def zero_expansion(cls, deg=0):
        """Initialize an expansion with all coefficients set to zero."""
        coeffs = np.zeros(deg + 1)
        return cls(coeffs=coeffs)

    @classmethod
    def unit_expansion(cls, deg=0):
        """Initialize an expansion with the first harmonic equal to 1 and all others zero"""
        coeffs = np.zeros(deg + 1)
        if cls == cosExpansion:
            coeffs[1] = 1
        else:
            coeffs[0] = 1
        return cls(coeffs=coeffs)

    def _initialize_coeffs(self, coeffs, thetas, vals, deg, useFFT, check_periodicity):
        if coeffs is not None:
            return coeffs, len(coeffs) - 1

        if useFFT:
            if check_periodicity:
                vals, thetas = self._enforce_periodicity(vals, thetas)
            if thetas is None:
                fftCoeff = self._perform_fft(vals)
            else:
                assert len(thetas) == len(vals), "Values for fit need to be equal to number of points"
                fftCoeff = self._perform_nfft(vals, thetas)
            deg = deg if deg is not None else len(fftCoeff) - 1
            return fftCoeff[:deg + 1], deg
        else:
            assert len(thetas) == len(vals), "Values for fit need to be equal to number of points"
            return self.fit(thetas, vals, deg), deg
        
    def _enforce_periodicity(self, vals, thetas):
        sign = 1 if self.trig_type == 'cos' else -1
        if not np.allclose(vals[0], sign * vals[-1]):
            vals = np.concatenate((vals, sign * np.flip(vals)))
            if thetas is not None:
                thetas = np.concatenate((thetas, thetas + np.pi))
        return vals, thetas

    def _perform_fft(self, vals):
        """ Uniform FFT """
        if self.trig_type == 'cos':
            coeff = dct(vals) / len(vals)
            # coeff[0] *= 0.5
            return coeff[::2]
        else:
            coeff = dst(vals) / len(vals)
            return coeff[1::2]
        
    def _perform_nfft(self, vals, thetas):
        """ Non-uniform FFT """
        N = len(vals)
        fft = nfft_adjoint(0.5 * thetas / np.pi, vals, N)
        left, right = np.split(fft, 2)
        if self.trig_type == 'cos':
            coeff = 2 * np.real(right) / N
            # coeff[0] *= 0.5
        else:
            coeff = -2 * np.flip(np.imag(left)) / N
        return coeff

    def __add__(self, obj):
        if isinstance(obj, TrigExpansion) and self.trig_type == obj.trig_type:
            if self.trig_type == 'cos':
                return cosExpansion(coeffs=self.coeffs + obj.coeffs)
            else:
                return sinExpansion(coeffs=self.coeffs + obj.coeffs)
        elif isinstance(obj, (int, float, np.integer, np.floating)) and self.trig_type == 'cos':
            newCoeff = np.copy(self.coeffs)
            # newCoeff[0] += obj
            newCoeff[0] += 2 * obj
            return cosExpansion(coeffs=newCoeff)
        return NotImplemented
    
    def __radd__(self, obj):
        """Handle addition when the instance is on the right side of the + operator."""
        # Redirect to __add__ since addition is commutative
        return self.__add__(obj)

    def __sub__(self, obj):
        if isinstance(obj, TrigExpansion) and self.trig_type == obj.trig_type:
            if self.trig_type == 'cos':
                return cosExpansion(coeffs=self.coeffs - obj.coeffs)
            else:
                return sinExpansion(coeffs=self.coeffs - obj.coeffs)
        elif isinstance(obj, (int, float, np.integer, np.floating)) and self.trig_type == 'cos':
            newCoeff = np.copy(self.coeffs)
            # newCoeff[0] -= obj
            newCoeff[0] -= 2 * obj
            return cosExpansion(coeffs=newCoeff)
        return NotImplemented

    def __mul__(self, obj):
        if isinstance(obj, TrigExpansion):
            new_vals = self.evalGridFFT() * obj.evalGridFFT()
            if self.trig_type == obj.trig_type:
                return cosExpansion(vals=new_vals)
            else:
                return sinExpansion(vals=new_vals)
        elif isinstance(obj, (int, float, np.integer, np.floating)):
            newCoeff = np.copy(self.coeffs)
            newCoeff *= obj
            if self.trig_type == 'cos':
                return cosExpansion(coeffs = newCoeff)
            else:
                return sinExpansion(coeffs = newCoeff)
        return NotImplemented
    
    def __rmul__(self, obj):
        """Handle multiplication when the instance is on the right side."""
        # Call __mul__ with swapped order
        return self.__mul__(obj)

    def __truediv__(self, obj):
        if isinstance(obj, TrigExpansion):
            new_vals = self.evalGridFFT() / obj.evalGridFFT()
            if self.trig_type == obj.trig_type:
                return cosExpansion(vals=new_vals)
            else:
                return sinExpansion(vals=new_vals)
        elif isinstance(obj, (int, float, np.integer, np.floating)):
            newCoeff = np.copy(self.coeffs)
            newCoeff /= obj
            if self.trig_type == 'cos':
                return cosExpansion(coeffs = newCoeff)
            else:
                return sinExpansion(coeffs = newCoeff)
        return NotImplemented
    
    def __pow__(self, exponent):
        if not isinstance(exponent, (int, np.integer)) and not np.allclose(exponent, 0.5):
            raise ValueError("Exponent must be an integer or 0.5")
        if np.allclose(exponent, 0):
            return cosExpansion.unit_expansion(deg = self.deg)
        elif np.allclose(exponent, 1):
            return self
        elif np.allclose(exponent, 0.5):
            return self.sqrt()
        else:
            result = cosExpansion.zero_expansion(deg = self.deg) + 1
            if exponent < 0:
                for i in range(-exponent):
                    result /= self
            else:
                for i in range(exponent):
                    result *= self
            return result
        
    def sqrt(self): # Highly suspect. Only use this when you know that the underlying series is the square of another series
        if self.trig_type == 'cos':
            new_vals = self.evalGridFFT() ** 0.5
            return cosExpansion(vals = new_vals)
        return NotImplemented # need to think about the sine case

    def fit(self, thetas, vals, deg): # seems to be working as of 13/9/24
        M = len(vals)
        assert M >= deg + 1, "Number of values needs to be at least deg + 1"
        A = np.zeros((M, deg + 1))
        if self.trig_type == 'cos':
            # A[:, 0] = 1
            A[:, 0] = 0.5
            for i in range(deg):
                A[:, i + 1] = np.cos((i + 1) * thetas)
        else:
            for i in range(deg + 1):
                A[:, i] = np.sin((i + 1) * thetas)
        B = A.T @ A
        return np.linalg.solve(B, A.T @ vals)

    def evalGrid(self, thetas):
        res = 0
        if self.trig_type == 'cos':
            # res += self.coeffs[0]
            res += 0.5 * self.coeffs[0]
            for i in range(1, len(self.coeffs)):
                res += self.coeffs[i] * np.cos(i * thetas)
        else:
            for i in range(len(self.coeffs)):
                res += self.coeffs[i] * np.sin((i+1) * thetas)
        return res

    def evalGridFFT(self): # idct and idst assume the presence of half frequencies. They are all zero in our case so we just set the respective coefficients to zero
        if self.trig_type == 'cos':
            newCoeff = np.zeros(2 * (self.deg + 1))
            newCoeff[::2] = np.copy(self.coeffs)
            # newCoeff[0] *= 2
            result = idct(newCoeff) * len(newCoeff)
            return result
        else:
            newCoeff = np.zeros(2 * (self.deg + 1))
            newCoeff[1::2] = self.coeffs
            result = idst(newCoeff) * len(newCoeff)
            return result

    def eval(self, theta):
        res = 0
        if self.trig_type == 'cos':
            # res += self.coeffs[0]
            res += 0.5 * self.coeffs[0]
            for i in range(1, len(self.coeffs)):
                res += self.coeffs[i] * np.cos(i * theta)
        else:
            for i in range(len(self.coeffs)):
                res += self.coeffs[i] * np.sin((i+1) * theta)
        return res

    def derivative(self):
        newCoeffs = np.zeros(self.deg + 1)
        if self.trig_type == 'cos': # derivative will be a sine series of deg = deg - 1
            for m in range(1, self.deg + 1):
                newCoeffs[m - 1] = -m * self.coeffs[m]
            return sinExpansion(coeffs = newCoeffs)
        else:
            for m in range(1, self.deg + 1):
                newCoeffs[m] = m * self.coeffs[m-1]
            return cosExpansion(coeffs=newCoeffs)
        
    def defIntegral(self):
        if self.trig_type == 'sin':
            m = np.arange(len(self.coeffs)) + 1
            intExp = cosExpansion(coeffs = -self.coeffs / m)
            return lambda theta1, theta0: intExp.eval(theta1) - intExp.eval(theta0)
        else:
            m = np.arange(len(self.coeffs) - 1) + 1
            intExp = sinExpansion(coeffs = self.coeffs[1:] / m)
            return lambda theta1, theta0: 0.5 * self.coeffs[0] * (theta1 - theta0) + intExp.eval(theta1) - intExp.eval(theta0)

    def integralAverage(self):
        if self.trig_type == 'cos':
            return np.pi * self.coeffs[0]
        else:
            return 0
    
    def copy(self):
        if self.trig_type == 'cos':
            return cosExpansion(coeffs = self.coeffs)
        else:
            return sinExpansion(coeffs = self.coeffs)

class cosExpansion(TrigExpansion):
    def __init__(self, coeffs=None, thetas=None, vals=None, deg=None, useDCT=True, check_periodicity = True):
        super().__init__(coeffs, thetas, vals, deg, useDCT, trig_type='cos', check_periodicity=check_periodicity)

class sinExpansion(TrigExpansion):
    def __init__(self, coeffs=None, thetas=None, vals=None, deg=None, useDST=True, check_periodicity = True):
        super().__init__(coeffs, thetas, vals, deg, useDST, trig_type='sin', check_periodicity=check_periodicity)

class TrigExpansionArray: # should add option to set qty name, also should change N to nRho
    def __init__(self, GridVals = None, expansions = None, coeffGrid = None, nRho = None, trig_type = None, deg = None, parities = None, rho1D = None):
        """
        Initialize the TrigExpansionArray with an optional list of sinExpansion or cosExpansion objects.
        """
        assert(parities is None or (len(parities) == 2 and (parities[0] == 'even' or parities[0] == 'odd') \
                                    and (parities[1] == 'even' or parities[1] == 'odd')))

        if expansions is None and GridVals is None and coeffGrid is None:
            assert(nRho is not None and trig_type is not None and deg is not None and parities is None)
            self.nRho = nRho
            self.trig_type = trig_type
            self.deg = deg
            self.expansions = self._initialize_empty()
            self.coeffGrid = np.zeros([self.nRho, self.deg + 1])
            self.parities = None
            self.rho1D = None
        else:
            assert(rho1D is not None)
            assert(not np.allclose(rho1D[0], 0)) # can relax this at a later time
            self.rho1D = np.copy(rho1D)
            self.nRho = len(rho1D)
            if expansions is not None:
                assert(len(expansions) == self.nRho)
                self.expansions, self.trig_type, self.deg = self._initialize_with_expansions(expansions)
                self.coeffGrid = self.calcCoeffGrid()
                self.GridVals = None
            elif GridVals is not None:
                assert(trig_type is not None)
                assert(len(GridVals) == self.nRho)
                self.trig_type = trig_type
                self.expansions, self.GridVals = self._initialize_with_grid(GridVals)
                self.deg = self.expansions[0].deg
                self.coeffGrid = self.calcCoeffGrid()
            elif coeffGrid is not None:
                assert(trig_type is not None)
                assert(len(coeffGrid) == self.nRho)
                self.trig_type = trig_type
                self.expansions, self.coeffGrid = self._initialize_with_coeffs(coeffGrid)
                self.deg = self.expansions[0].deg
                self.GridVals = None
            
            self.nTheta = 2 * (self.deg + 1)
            self.theta1D = np.array([2 * (i + 0.5) * np.pi / self.nTheta for i in range(self.nTheta)])

            # initialize other quantities
            self.twoSidedRho = None
            self.twoSidedCoeff = None
            self.twoSidedGridVals = None
            self.oppParities = None
            self.interp = None

            if parities is None:
                self.parities = self.calcParities()
            else:
                self.parities = parities

    # Initialization functions

    def _initialize_empty(self):
        if self.trig_type == 'cos':
            return np.array([cosExpansion.zero_expansion(deg = self.deg) for i in range(self.nRho)])
        else:
            return np.array([sinExpansion.zero_expansion(deg = self.deg) for i in range(self.nRho)])
    
    def _initialize_with_grid(self, GridVals):

        # check whether the grid is periodic
        if self.trig_type == 'cos' and not np.allclose(GridVals[:, 0], GridVals[:, -1]):
            GridVals = np.hstack((GridVals, np.fliplr(GridVals)))
        elif self.trig_type == 'sin' and not np.allclose(GridVals[:, 0], -GridVals[:, -1]):
            GridVals = np.hstack((GridVals, -np.fliplr(GridVals)))
        
        selftype = cosExpansion if self.trig_type == 'cos' else sinExpansion
        expansions = np.zeros(self.nRho, dtype = selftype)
        for i in range(self.nRho):
            expansions[i] = selftype(vals = GridVals[i], check_periodicity=False)
        
        return expansions, GridVals

    def _initialize_with_expansions(self, expansions):
        """
        Ensure all expansions are of the same type (all sinExpansion or all cosExpansion).
        """
        trig_type = expansions[0].trig_type
        deg = expansions[0].deg
        if any(exp.trig_type != trig_type for exp in expansions):
            raise ValueError("All expansions must be of the same trigonometric type.")
        if any(exp.deg != deg for exp in expansions):
            raise ValueError("All expansions must have the same degree.")
        return expansions, trig_type, deg
    
    def _initialize_with_coeffs(self, coeffGrid):
        selftype = cosExpansion if self.trig_type == 'cos' else sinExpansion
        expansions = np.zeros(self.nRho, dtype = selftype)
        for i in range(self.nRho):
            expansions[i] = selftype(coeffs = coeffGrid[i])
        return expansions, coeffGrid

    # Basic operations

    def __len__(self):
        """
        Get the number of expansions in the array.
        """
        return len(self.expansions)

    def __add__(self, other):
        """
        Add another TrigExpansionArray or a scalar value.
        """
        if isinstance(other, TrigExpansionArray):
            if len(self.expansions) != len(other.expansions):
                raise ValueError("Arrays must be of the same length to add.")
            elif self.trig_type != other.trig_type:
                raise ValueError("Cannot add two different types of expansions.")
            elif not np.allclose(self.rho1D, other.rho1D):
                raise ValueError("Cannot add two expansions sampled at different radial points.")
            else:
                return TrigExpansionArray(expansions = [exp1 + exp2 for exp1, exp2 in zip(self.expansions, other.expansions)], rho1D = self.rho1D)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            if self.trig_type == 'sin':
                raise ValueError("Cannot add a scalar to a sine series")
            else:
                return TrigExpansionArray(expansions = [exp + other for exp in self.expansions], rho1D = self.rho1D, parities = self.getParities())
        elif isinstance(other, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in other):
            # Ensure the array length matches the length of expansions
            if len(self.expansions) != len(other):
                raise ValueError("The array of scalars must match the length of the expansions.")
            if self.trig_type == 'sin':
                raise ValueError("Cannot add scalars to a sine series.")
            else:
                # Add each scalar to the corresponding expansion
                return TrigExpansionArray(expansions=[exp + scalar for exp, scalar in zip(self.expansions, other)], rho1D=self.rho1D)
        else:
            return NotImplemented
        
    def __radd__(self, obj):
        """Handle addition when the instance is on the right side of the + operator."""
        # Redirect to __add__ since addition is commutative
        return self.__add__(obj)
    
    def __sub__(self, other):
        """
        Subtract another TrigExpansionArray or a scalar value.
        """
        return self + -1 * other

    def __mul__(self, other):
        """
        Multiply each expansion in the array by another TrigExpansionArray or a scalar.
        """
        if isinstance(other, TrigExpansionArray):
            if len(self.expansions) != len(other.expansions):
                raise ValueError("Arrays must be of the same length to multiply.")
            elif not np.allclose(self.rho1D, other.rho1D):
                raise ValueError("Cannot multiply two expansions sampled at different radial points.")
            else:
                return TrigExpansionArray(expansions = [exp1 * exp2 for exp1, exp2 in zip(self.expansions, other.expansions)], rho1D = self.rho1D)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            return TrigExpansionArray(expansions = [exp * other for exp in self.expansions], rho1D = self.rho1D, parities = self.parities)
        elif isinstance(other, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in other):
            # Ensure the array length matches the length of expansions
            if len(self.expansions) != len(other):
                raise ValueError("The array of scalars must match the length of the expansions.")
            else:
                return TrigExpansionArray(expansions=[exp * scalar for exp, scalar in zip(self.expansions, other)], rho1D=self.rho1D)
        else:
            return NotImplemented
    
    def __rmul__(self, obj):
        """Handle multiplication when the instance is on the right side."""
        # Call __mul__ with swapped order
        return self.__mul__(obj)
        
    def __truediv__(self, other):
        """
        Divide each expansion in the array by another TrigExpansionArray or a scalar.
        """
        if isinstance(other, TrigExpansionArray) and np.allclose(self.rho1D, other.rho1D):
            if len(self.expansions) != len(other.expansions):
                raise ValueError("Arrays must be of the same length to divide.")
            elif not np.allclose(self.rho1D, other.rho1D):
                raise ValueError("Cannot divide two expansions sampled at different radial points.")
            else:
                return TrigExpansionArray(expansions = [exp1 / exp2 for exp1, exp2 in zip(self.expansions, other.expansions)], rho1D = self.rho1D)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            return TrigExpansionArray(expansions = [exp / other for exp in self.expansions], rho1D = self.rho1D, parities = self.parities)
        elif isinstance(other, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in other):
            # Ensure the array length matches the length of expansions
            if len(self.expansions) != len(other):
                raise ValueError("The array of scalars must match the length of the expansions.")
            else:
                return TrigExpansionArray(expansions=[exp / scalar for exp, scalar in zip(self.expansions, other)], rho1D=self.rho1D)
        else:
            return NotImplemented

    def __pow__(self, exponent):
        """
        Raise each expansion in the array to a power.
        """
        if isinstance(exponent, (int, float, np.integer, np.floating)):
            return TrigExpansionArray(expansions = [exp ** exponent for exp in self.expansions], rho1D = self.rho1D)
        elif isinstance(exponent, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in exponent):
            # Ensure the array length matches the length of expansions
            if len(self.expansions) != len(exponent):
                raise ValueError("The array of scalars must match the length of the expansions.")
            else:
                return TrigExpansionArray(expansions=[exp ** scalar for exp, scalar in zip(self.expansions, exponent)], rho1D=self.rho1D)
        else:
            return NotImplemented
        
    def __getitem__(self, index):
        """
        Access an expansion by index.
        """
        return self.expansions[index]

    # Calling and evaluation functions

    def __call__(self, rho, theta): # need to thoroughly test this
        """ Don't call with rho = self.rho1D and theta = self.theta1D, just use self.getGridVals() in that case """

        if isinstance(rho, (list, np.ndarray)):
            rho = np.array(rho)
            assert(rho.ndim == 1)
            assert(np.max(rho) <= np.max(self.rho1D) and np.min(rho) >= 0)
        elif isinstance(rho, (int, float, np.integer, np.floating)):
            assert(rho <= np.max(self.rho1D) and rho >= 0)
        else:
            raise ValueError('Rho needs to be a scalar or an array')

        if isinstance(theta, (int, float, np.integer, np.floating)):
            assert(theta <= 2 * np.pi and theta >= 0)
        elif isinstance(theta, (list, np.ndarray)):
            theta = np.array(theta)
            assert(theta.ndim == 1)
        else:
            raise ValueError('Theta needs to be a scalar or an array')
        
        interp = self.getInterp()
        return interp(rho, theta)
        
    def eval_all(self, theta):
        """
        Evaluate all expansions at a given angle theta.
        Returns a list of results.
        """
        return np.array([exp.eval(theta) for exp in self.expansions])

    def eval_all_grid(self, thetas = None):
        """
        Evaluate all expansions on a grid of theta values.
        Returns a list of results.
        """

        if thetas is None:
            result = np.array([exp.evalGridFFT() for exp in self.expansions])
        else:
            if thetas.ndim == 1:
                result = np.array([exp.evalGrid(thetas) for exp in self.expansions])
            else:
                result = np.array([self.expansions[i].evalGrid(thetas[i]) for i in range(self.N)])
        
        return result
        
    # Get functions

    def getGridVals(self):
        if self.GridVals is None:
            self.GridVals = self.eval_all_grid()
        return self.GridVals
    
    def getCoeffGrid(self):
        if self.coeffGrid is None:
            self.coeffGrid = self.calcCoeffGrid()
        return self.coeffGrid
    
    def getOppositeParities(self):
        if self.oppParities is None:
            parities = self.getParities()
            oppParities = ()
            for par in parities:
                if par == 'even':
                    oppParities += ('odd',)
                else:
                    oppParities += ('even',)
            self.oppParities = oppParities
        return self.oppParities
    
    def getTwoSidedCoeff(self):
        if self.twoSidedCoeff is None:
            self.twoSidedCoeff = self.calcTwoSidedCoeff()
        return self.twoSidedCoeff
    
    def getTwoSidedRho(self):
        if self.twoSidedRho is None:
            self.twoSidedRho = np.concatenate((-np.flip(self.rho1D), self.rho1D))
        return self.twoSidedRho
    
    def getTwoSidedGridVals(self):
        if self.twoSidedGridVals is None:
            self.twoSidedGridVals = self.calcTwoSidedGridVals(self.getGridVals())
        return self.twoSidedGridVals
    
    def getParities(self):
        if self.parities is None:
            self.parities = self.calcParities()
        return self.parities
    
    def getInterp(self):
        if self.interp is None:
            self.interp = RectBivariateSpline(self.getTwoSidedRho(), self.theta1D, self.getTwoSidedGridVals(), kx = 5, ky = 5)
        return self.interp

    # Calculus
    
    def thetaDer(self): # could implement more derivatives here

        return TrigExpansionArray(expansions = [exp.derivative() for exp in self.expansions], rho1D = self.rho1D)
    
    def rhoDerFD(self, order = 1, acc = 4):

        coeffs2Side = self.getTwoSidedCoeff()
        newCoeffGrid = np.copy(self.getCoeffGrid()) * 0
        rho2Side = self.getTwoSidedRho()
        ddRho = FinDiff(0, rho2Side, order, acc = acc)
        newCoeffGrid = ddRho(coeffs2Side)[self.nRho:]
        newParities = self.getOppositeParities() if order % 2 == 1 else self.parities

        return TrigExpansionArray(coeffGrid = newCoeffGrid, trig_type = self.trig_type, parities = newParities, rho1D = self.rho1D)
    
    def integralAverage(self, derOrder = 0, acc = 4):
        if self.trig_type == 'sin' or np.allclose(self.getCoeffGrid[:, 0], np.zeros(self.nRho)):
            return np.zeros(self.nRho)
        else:
            parities = self.getParities()
            sign = 1 if parities[0] == 'even' else -1
            coeffGrid = self.getCoeffGrid()
            firstHarm = coeffGrid[:, 0]
            if derOrder == 0:
                return np.pi * firstHarm
            else:
                firstHarm2Side = np.concatenate((sign * np.flip(firstHarm), firstHarm))
                rho2Side = self.getTwoSidedRho()
                ddRho = FinDiff(0, rho2Side, derOrder, acc = acc)
                return ddRho(firstHarm2Side)[self.nRho:]
    
    # One-time calculation functions
    
    def calcCoeffGrid(self):

        coeffGrid = np.zeros([self.nRho, self.deg + 1])
        for i in range(self.nRho):
            coeffGrid[i] = self.expansions[i].coeffs
        
        return coeffGrid

    def calcTwoSidedCoeff(self): # assumes magnetic axis is not included, could add this case later
        """ Constructs two sided version of coefficients of expansion array, multiplying coefficients according to parity """

        coeffGrid = self.getCoeffGrid()
        coeffs2Side = np.vstack((np.flipud(coeffGrid), coeffGrid))
        parities = self.getParities()

        mult = np.ones(self.deg + 1)
        if parities[0] == 'even':
            mult[1::2] = -1
        else:
            mult[::2] = -1
        
        for i in range(self.nRho):
            coeffs2Side[i] *= mult

        return coeffs2Side
    
    def calcTwoSidedGridVals(self, GridVals):
        nRho, nTheta = GridVals.shape
        twoSidedGridVals = np.zeros([2 * nRho, nTheta // 2])
        parities = self.getParities()
        sign = -1 if parities[1] == 'even' else 1
        for i in range(nTheta // 2):
            right = GridVals[:, i]
            left = sign * np.flip(GridVals[:, i + nTheta // 2])
            twoSidedGridVals[:, i] = np.concatenate((left, right))
        sign = -1 if self.trig_type == 'sin' else 1
        twoSidedGridVals = np.hstack((twoSidedGridVals, sign * np.fliplr(twoSidedGridVals)))
        return twoSidedGridVals
    
    def calcParityArr(self, rho1D, arr):
        """ Determines the parity of an array of values by fitting a polynomial to the first five points """
        assert(len(rho1D) == len(arr) and len(rho1D) >= 5)
        coeff = polyfit(rho1D[:5], arr[:5], 3)
        even = sum(np.abs(coeff[::2]))
        odd = sum(np.abs(coeff[1::2]))
        parity = 'even' if even > odd else 'odd'
        return parity
    
    def calcParities(self):
        """ This calculates the parity of the first harmonic and the parity of the grid values. For the harmonic case, if the first harmonic is zero, 
        it calculates the parity of the second harmonic; the first harmonic's parity is then its opposite"""
        # coefficients
        coeffGrid = self.getCoeffGrid()
        firstHarm = coeffGrid[:, 0]
        if not np.allclose(firstHarm, np.zeros(len(firstHarm))):
            coeffParity = self.calcParityArr(self.rho1D, firstHarm)
        else:
            secondHarm = coeffGrid[:, 1]
            parity2 = self.calcParityArr(self.rho1D, secondHarm)
            coeffParity = 'odd' if parity2 == 'even' else 'even'
        # grid values
        GridVals = self.getGridVals()
        firstCol = GridVals[:, 0]
        if not np.allclose(firstCol, np.zeros(self.nRho)):
            valsParity = self.calcParityArr(self.rho1D, firstCol - firstCol[0])
        else:
            valsParity = self.calcParityArr(self.rho1D, GridVals[:, 1] - GridVals[0, 1]) # hopefully second column will not be a bunch of zeros too
        return (coeffParity, valsParity)
    
    # Plotting
            
    def plotCoeffs(self, nHarm = 5, rhoLim = 1, twoSided = True):

        if twoSided:
            coeffs = self.getTwoSidedCoeff()
            rho = self.getTwoSidedRho()
            idx = np.where((rho <= rhoLim) & (rho >= -rhoLim))
        else:
            coeffs = self.getCoeffGrid()
            rho = self.rho1D
            idx = np.where((self.rho1D <= rhoLim))

        fig, ax = plt.subplots(nrows = nHarm, ncols = 1, sharex = True, figsize = (5, 1.5 * nHarm))

        adj = 1 if self.trig_type == 'sin' else 0

        for i in range(nHarm):
            ax[i].plot(rho[idx], coeffs[:, i][idx], marker = '.', ls = '')
            ax[i].set_ylabel(r'$m = $' + str(i + adj))
            ax[i].grid()
            ax[i].ticklabel_format(axis = 'y', scilimits = (-3, 3))

        ax[-1].set_xlabel(r'$\rho$')

        plt.tight_layout()
        return
    
    def plotGridVals(self, figsize = (5, 5)):
        
        GridVals = self.getGridVals()
        fig = plt.figure(figsize = figsize)
        plt.imshow(GridVals, extent = (0, 1, 1, 0))
        plt.colorbar()
        plt.xlabel(r'$\frac{\theta}{2 \pi}$')
        plt.ylabel(r'$\frac{\rho}{\rho_{max}}$')

        return
    
    def plotSingleSurface(self, idx, figsize = (3.5, 5)):

        assert(idx < self.nRho)

        GridVals = self.getGridVals()
        fig = plt.figure(figsize = figsize)
        plt.plot(self.theta1D, GridVals[idx])
        plt.xlabel(r'$\theta$')
        plt.ylabel(r'$M(\rho = $' + str(np.around(self.rho1D[idx], 2)) + r'$)$')
        plt.grid()
        return