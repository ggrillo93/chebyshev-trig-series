import numpy as np
from scipy.fft import dct, dst, idct, idst
from nfft import nfft_adjoint
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit

class TrigExpansion:
    """ useFFT = True is much faster for evenly spaced samples and slower up to nTheta ~ 50 for unevenly spaced samples """
    def __init__(self, coeffs=None, thetas=None, vals=None, deg=None, useFFT=True, trig_type='cos'):
        assert trig_type in ['cos', 'sin'], "trig_type must be 'cos' or 'sin'"
        self.trig_type = trig_type
        self.coeffs, self.deg = self._initialize_coeffs(coeffs, thetas, vals, deg, useFFT)

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

    def _initialize_coeffs(self, coeffs, thetas, vals, deg, useFFT):
        if coeffs is not None:
            return coeffs, len(coeffs) - 1

        if useFFT:
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

    def contourIntegral(self):
        return 2 * np.pi * self.coeffs[0]
    
    def copy(self):
        if self.trig_type == 'cos':
            return cosExpansion(coeffs = self.coeffs)
        else:
            return sinExpansion(coeffs = self.coeffs)

class cosExpansion(TrigExpansion):
    def __init__(self, coeffs=None, thetas=None, vals=None, deg=None, useDCT=True):
        super().__init__(coeffs, thetas, vals, deg, useDCT, trig_type='cos')

class sinExpansion(TrigExpansion):
    def __init__(self, coeffs=None, thetas=None, vals=None, deg=None, useDST=True):
        super().__init__(coeffs, thetas, vals, deg, useDST, trig_type='sin')

class TrigExpansionArray: # should add option to initialize with coefficient grid
    def __init__(self, GridVals = None, expansions = None, N = None, trig_type = None, deg = None, parity = None, rho1D = None):
        """
        Initialize the TrigExpansionArray with an optional list of sinExpansion or cosExpansion objects.
        """
        assert(parity is None or parity == 'even' or parity == 'odd')

        if expansions is None and GridVals is None:
            assert(N is not None and trig_type is not None and deg is not None and parity is None)
            self.N = N
            self.trig_type = trig_type
            self.deg = deg
            self.expansions = self._initialize_empty()
            self.coeffGrid = np.zeros([self.N, self.deg + 1])
            self.parity = None
            self.rho1D = None
        else:
            assert(rho1D is not None)
            assert(not np.allclose(rho1D[0], 0)) # can relax this at a later time
            self.rho1D = np.copy(rho1D)
            if GridVals is None and expansions is not None:
                self.expansions = np.array(expansions)
                self.trig_type, self.deg = self._validate_type()
                self.N = len(expansions)
                self.GridVals = self.eval_all_grid()
            else:
                assert(trig_type is not None)
                self.N = len(GridVals)
                self.trig_type = trig_type
                self.expansions, self.GridVals = self._initialize_with_grid(GridVals)
                self.deg = self.expansions[0].deg
            self.coeffGrid = self.calcCoeffGrid()
            if parity is None:
                self.parity = self.calcParity()
            else:
                self.parity = parity

    def _initialize_empty(self):
        if self.trig_type == 'cos':
            return np.array([cosExpansion.zero_expansion(deg = self.deg) for i in range(self.N)])
        else:
            return np.array([sinExpansion.zero_expansion(deg = self.deg) for i in range(self.N)])
    
    def _initialize_with_grid(self, GridVals):

        # check whether the grid is periodic
        if self.trig_type == 'cos' and not np.allclose(GridVals[:, 0], GridVals[:, -1]):
            GridVals = np.hstack((GridVals, np.fliplr(GridVals)))
        elif self.trig_type == 'sin' and not np.allclose(GridVals[:, 0], -GridVals[:, -1]):
            GridVals = np.hstack((GridVals, -np.fliplr(GridVals)))
        
        if self.trig_type == 'cos':
            expansions = np.zeros(self.N, dtype = cosExpansion)
            for i in range(self.N):
                expansions[i] = cosExpansion(vals = GridVals[i])
        else:
            expansions = np.zeros(self.N, dtype = sinExpansion)
            for i in range(self.N):
                expansions[i] = sinExpansion(vals = GridVals[i])
        
        return expansions, GridVals

    def _validate_type(self):
        """
        Ensure all expansions are of the same type (all sinExpansion or all cosExpansion).
        Returns the type if valid, otherwise raises an error.
        """
        trig_type = self.expansions[0].trig_type
        deg = self.expansions[0].deg
        if any(exp.trig_type != trig_type for exp in self.expansions):
            raise ValueError("All expansions must be of the same trigonometric type.")
        if any(exp.deg != deg for exp in self.expansions):
            raise ValueError("All expansions must have the same degree.")
        return trig_type, deg

    def __len__(self):
        """
        Get the number of expansions in the array.
        """
        return len(self.expansions)

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
                return TrigExpansionArray(expansions = [exp + other for exp in self.expansions], rho1D = self.rho1D, parity = self.parity)
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
        if isinstance(other, TrigExpansionArray):
            if len(self.expansions) != len(other.expansions):
                raise ValueError("Arrays must be of the same length to add.")
            elif self.trig_type != other.trig_type:
                raise ValueError("Cannot subract two different types of expansions.")
            elif not np.allclose(self.rho1D, other.rho1D):
                raise ValueError("Cannot subtract two expansions sampled at different radial points.")
            else:
                return TrigExpansionArray(expansions = [exp1 - exp2 for exp1, exp2 in zip(self.expansions, other.expansions)], rho1D = self.rho1D)
        elif isinstance(other, (int, float, np.integer, np.floating)):
            if self.trig_type == 'sin':
                raise ValueError("Cannot subtract a scalar to a sine series")
            else:
                return TrigExpansionArray(expansions = [exp - other for exp in self.expansions], rho1D = self.rho1D, parity = self.parity)
        elif isinstance(other, (list, np.ndarray)) and all(isinstance(x, (int, float, np.integer, np.floating)) for x in other):
            # Ensure the array length matches the length of expansions
            if len(self.expansions) != len(other):
                raise ValueError("The array of scalars must match the length of the expansions.")
            if self.trig_type == 'sin':
                raise ValueError("Cannot add scalars to a sine series.")
            else:
                return TrigExpansionArray(expansions=[exp - scalar for exp, scalar in zip(self.expansions, other)], rho1D=self.rho1D)
        else:
            return NotImplemented

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
            return TrigExpansionArray(expansions = [exp * other for exp in self.expansions], rho1D = self.rho1D, parity = self.parity)
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
            return TrigExpansionArray(expansions = [exp / other for exp in self.expansions], rho1D = self.rho1D, parity = self.parity)
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
    
    def thetaDerivative(self): # could implement more derivatives here
        return TrigExpansionArray(expansions = [exp.derivative() for exp in self.expansions], rho1D = self.rho1D, parity = 'odd' if self.parity == 'even' else 'even')
    
    def calcCoeffGrid(self):

        coeffGrid = np.zeros([self.N, self.deg + 1])
        for i in range(self.N):
            coeffGrid[i] = self.expansions[i].coeffs
        
        return coeffGrid

    def twoSided(self): # assumes magnetic axis is not included, could add this case later
        """ Constructs two sided version of coefficients of expansion array, multiplying coefficients according to parity """

        coeffs2Side = np.vstack((np.flipud(self.coeffGrid), self.coeffGrid))

        mult = np.ones(self.deg + 1)
        if self.parity == 'even':
            mult[1::2] = -1
        else:
            mult[::2] = -1
        
        for i in range(self.N):
            coeffs2Side[i] *= mult

        return coeffs2Side
    
    def calcParity(self, printSums = False):
        firstHarm = self.coeffGrid[:, 0]
        if not np.allclose(firstHarm, np.zeros(len(firstHarm))):
            coeff = polyfit(self.rho1D[:5], firstHarm[:5], 3)
            even = sum(np.abs(coeff[::2]))
            odd = sum(np.abs(coeff[1::2]))
            parity = 'even' if even > odd else 'odd'
        else:
            secondHarm = self.coeffGrid[:, 1]
            coeff = polyfit(self.rho1D[:5], secondHarm[:5], 3)
            even = sum(np.abs(coeff[::2]))
            odd = sum(np.abs(coeff[1::2]))
            parity = 'odd' if even > odd else 'even'
        if printSums:
            print([even, odd])
        return parity
    
    def plotCoeffs(self, nHarm = 5, rhoLim = 1, twoSided = True):

        if twoSided:
            coeffs = self.twoSided()
            rho = np.concatenate((-np.flip(self.rho1D), self.rho1D))
            idx = np.where((rho <= rhoLim) & (rho >= -rhoLim))
        else:
            coeffs = self.coeffGrid
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