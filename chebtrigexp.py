from trigexp import *
from numpy.polynomial.chebyshev import Chebyshev

class ChebExpansionArray():
    def __init__(self, polynomials = None, ChebCoeffGrid = None, TrigCoeff2Side = None, parity = None):
        """ ChebCoeffGrid has number of rows equal to the number of radial points and number of columns equal to the number of Fourier modes/Chebyshev polynomials.
            Assumes radial values have been evaluated at the Chebyshev points of the first kind """
        assert(polynomials is not None or TrigCoeff2Side is not None or ChebCoeffGrid is not None)
        if polynomials is not None:
            assert(all(isinstance(p, Chebyshev)) for p in polynomials) # check all polynomials are Chebyshev polynomials
            assert(all(len(p.degree()) == len(polynomials[0].degree()) for p in polynomials)) # check all polynomials are of the same degree
            self.polynomials = polynomials
            self.ChebDeg = polynomials[0].degree()
            if ChebCoeffGrid is not None:
                self.TrigDeg = ChebCoeffGrid.shape[-1] - 1 # one chebyshev polynomial for each Fourier mode
                self.coeffGrid = ChebCoeffGrid
            else:
                self.TrigDeg, self.coeffGrid = self.calcCoeffGrid(polynomials)
        elif ChebCoeffGrid is not None:
            self.polynomials, self.coeffGrid, self.ChebDeg, self.TrigDeg = self._initialize_with_ChebCoeff(ChebCoeffGrid)
        else:
            assert(parity is not None)
            self.parity = parity
            self.polynomials, self.coeffGrid, self.ChebDeg, self.TrigDeg = self._initialize_with_TrigCoeff(TrigCoeff2Side)
            self.checkParity()

        if parity is None:
            self.parity = self.determineParity(polynomials)
        else:
            self.parity = parity
    
    def _initialize_with_ChebCoeff(self, ChebCoeffGrid):
        ChebDeg, TrigDeg = np.array(ChebCoeffGrid.shape) - np.array([1, 1])
        polynomials = np.zeros(self.TrigDeg + 1, dtype = Chebyshev)
        for i in range(self.TrigDeg + 1):
            polynomials[i] = Chebyshev(ChebCoeffGrid[:, i])
        return polynomials, ChebCoeffGrid, ChebDeg, TrigDeg
    
    def _initialize_with_TrigCoeff(self, TrigCoeff2Side):
        ChebDeg, TrigDeg = np.array(TrigCoeff2Side.shape) - np.array([1, 1])
        ChebCoeffGrid = np.copy(TrigCoeff2Side) * 0
        polynomials = np.zeros(TrigDeg + 1, dtype = Chebyshev)
        adj = 0 if self.parity == 'even' else 1
        for i in range(self.TrigDeg + 1):
            sign = 1 if (i + adj) % 2 == 0 else -1
            coeffs = sign * dct(TrigCoeff2Side[:, i]) / (ChebDeg + 1)
            ChebCoeffGrid[:, i] = coeffs
            polynomials[i] = Chebyshev(coeffs)
        
        return polynomials, ChebCoeffGrid, ChebDeg, TrigDeg
    
    def __getitem__(self, index):
        
        return self.polynomials[index]

    def __len__(self):
        
        return len(self.polynomials)
    
    def calcCoeffGrid(self, polynomials):
        ChebCoeffGrid = np.zeros([self.ChebDeg + 1, self.TrigDeg + 1])
        for i in range(self.TrigDeg + 1):
            ChebCoeffGrid[:, i] = polynomials[i].coef
        return ChebCoeffGrid
    
    def determineParity(self, polynomials):
        zeros = np.zeros(len(polynomials) // 2)
        check = polynomials[1].coef
        check2 = polynomials[2].coef
        if np.allclose(check[::2], zeros) and np.allclose(check2[1::2], zeros):
            return 'even'
        elif np.allclose(check[1::2], zeros) and np.allclose(check2[::2], zeros):
            return 'odd'
        else:
            raise ValueError('Parity is not right')
    
    def checkParity(self):
        zeros = np.zeros((self.ChebDeg + 1) // 2)
        check = self.polynomials[1].coef
        check2 = self.polynomials[2].coef
        if self.parity == 'even' and not (np.allclose(check[::2], zeros) and np.allclose(check2[1::2], zeros)):
            self.parity = self.determineParity(self.polynomials)
        elif self.parity == 'odd' and not (np.allclose(check[1::2], zeros) and np.allclose(check2[::2], zeros)):
            self.parity = self.determineParity(self.polynomials)
        return
    
    def oppositeParity(self):
        return 'odd' if self.parity == 'even' else 'even'
    
    def spatialDerivative(self, order = 1):
        newPolys = np.array([p.deriv(m = order) for p in self.polynomials])
        return ChebExpansionArray(polynomials = newPolys, parity = self.oppositeParity() if order % 2 == 1 else self.parity)

class ChebTrigExpansion():
    def __init__(self, GridVals = None, ChebExpArr = None, TrigExpArr = None, trig_type = None, parity = None, rho1D = None):
        """ Chebyshev-trigonometric expansion in polar coordinates. The angular dependence is represented by a cosine or sine series at each radial coordinate, 
            and the radial dependency of the coefficients of these series are treated with parity restricted Chebyshev polynomials. 
            Can initialize with a 2D grid of sampled points in (rho, theta), a TrigExpansionArray, or a ChebExpansionArray"""

        assert(not (TrigExpArr is None and GridVals is None and ChebExpArr is None))

        if TrigExpArr is None:
            if GridVals is not None:
                assert(trig_type is not None and rho1D is not None)
                TrigExpArr = TrigExpansionArray(GridVals = GridVals, trig_type = trig_type, parity = parity, rho1D = rho1D)
            else:
                assert(isinstance(TrigExpArr, TrigExpansionArray))

            self.N = np.copy(TrigExpArr.N)
            self.rho1D = np.copy(TrigExpArr.rho1D)
            self.trig_type = np.copy(TrigExpArr.trig_type)
            self.TrigCoeffGrid = np.copy(TrigExpArr.coeffGrid)
            self.GridVals = np.copy(TrigExpArr.GridVals)
            self.parity = np.copy(TrigExpArr.parity)
            self.TrigDeg = np.copy(TrigExpArr.deg)
            self.TrigExpArr = TrigExpArr # make sure you don't modify it

            self.ChebDeg, self.ChebCoeffGrid, self.ChebExpArr = self._initialize_with_TrigExp(TrigExpArr)
        else:
            assert(isinstance(ChebExpArr, ChebExpansionArray))
            assert(trig_type is not None and rho1D is not None)

            self.rho1D = rho1D
            self.N = len(self.rho1D)
            self.trig_type = trig_type
            self.parity = ChebExpArr.parity
            self.ChebExpArr = ChebExpArr

            self.TrigDeg, self.TrigCoeffGrid, self.TrigExpArr, self.GridVals = self._initialize_with_TrigExp(ChebExpArr)
        
    def _initialize_with_TrigExp(self, TrigExpArr):
        
        ChebExpArr = ChebExpansionArray(TrigCoeff2Side = TrigExpArr.twoSided(), parity = TrigExpArr.parity)

        return ChebExpArr.ChebDeg, ChebExpArr.coeffGrid, ChebExpArr
    
    def _initialize_with_ChebExp(self, ChebExpArr):
        
        TrigDeg = ChebExpArr.TrigDeg
        expType = cosExpansion if self.trig_type == 'cos' else sinExpansion
        TrigCoeffGrid = np.zeros([self.N, TrigDeg + 1])
        TrigExpansions = np.zeros(self.N, dtype = expType)

        for i in range(TrigDeg + 1):
            TrigCoeffGrid[:, i] = ChebExpArr.polynomials[i](self.rho1D)

        for i in range(self.N):
            TrigExpansions[i] = expType(coeffs = TrigCoeffGrid[i])

        TrigExpArr = TrigExpansionArray(expansions = TrigExpansions, trig_type = self.trig_type, parity = self.parity, rho1D = self.rho1D)

        return TrigDeg, TrigCoeffGrid, TrigExpArr, TrigExpArr.GridVals
        
    def spatialDerivative(self, order = 1):

        newChebExpArr = self.ChebExpArr.spatialDerivative(order = order)
        
        return ChebTrigExpansion(ChebExpArr = newChebExpArr, trig_type = self.trig_type, rho1D = self.rho1D)
    
    def angDerivative(self):

        newTrigExpArr = self.TrigExpArr.thetaDerivative()

        newParity = 'even' if self.parity == 'odd' else 'odd'

        return ChebTrigExpansion(TrigExpArr = newTrigExpArr, parity = newParity)