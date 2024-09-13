import numpy as np
from scipy.fft import dct, dst, idct, idst
from nfft import nfft_adjoint

class TrigExpansion:
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
            if type(thetas) == type(None):
                fftCoeff = self._perform_fft(vals)
            else:
                assert len(thetas) == len(vals), "Values for fit need to be equal to number of points"
                fftCoeff = self._perform_nfft(vals, thetas)
            deg = deg if deg is not None else len(fftCoeff) - 1
            return fftCoeff[:deg + 1], deg
        else:
            assert len(thetas) == len(vals), "Values for fit need to be equal to number of points"
            return self.fit(thetas, vals, deg), deg

    def _perform_fft(self, vals):
        """ Uniform FFT """
        if self.trig_type == 'cos':
            coeff = dct(vals) / len(vals)
            coeff[0] *= 0.5
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
        else:
            coeff = -2 * np.imag(left) / N
        return coeff

    def __add__(self, obj):
        if isinstance(obj, TrigExpansion) and self.trig_type == obj.trig_type:
            if self.trig_type == 'cos':
                return cosExpansion(coeffs=self.coeffs + obj.coeffs)
            else:
                return sinExpansion(coeffs=self.coeffs + obj.coeffs)
        elif isinstance(obj, (int, float, np.integer, np.floating)) and self.trig_type == 'cos':
            newCoeff = np.copy(self.coeffs)
            newCoeff[0] += obj
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
            newCoeff[0] -= obj
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

    def fit(self, thetas, vals, deg): # not sure if this is working properly
        M = len(vals)
        assert M >= deg + 1, "Number of values needs to be at least deg + 1"
        A = np.zeros((M, deg + 1))
        A[:, 0] = 1
        for i in range(deg):
            trig_func = np.cos if self.trig_type == 'cos' else np.sin
            A[:, i + 1] = trig_func((i + 1) * thetas)
        B = A.T @ A
        if self.trig_type == 'cos':
            return np.linalg.solve(B, A.T @ vals)
        else:
            return np.linalg.solve(B, A.T @ vals)[1:] # exclude constant term for sine since sine has no zero harmonic

    def evalGrid(self, thetas):
        res = 0
        if self.trig_type == 'cos':
            res += self.coeffs[0]
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
            newCoeff[0] *= 2
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
            res += self.coeffs[0]
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

    def integralContour(self):
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
            