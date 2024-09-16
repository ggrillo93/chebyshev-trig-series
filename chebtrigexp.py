from trigexp import *
from findiff import FinDiff
from chebpy.core.algorithms import vals2coeffs2, chebpts2
from chebpy.core.chebtech import Chebtech2

class ChebTrigExpansion():
    def __init__(self, GridVals = None, ChebCoeffGrid = None, TrigExpArr = None, trig_type = None, parity = None, rho1D = None):
        """ Chebyshev-trigonometric expansion in polar coordinates. The angular dependence is represented by a cosine or sine series at each radial coordinate, 
            and the radial dependency of the coefficients of these series are treated with parity restricted Chebyshev polynomials. 
            Can initialize with a 2D grid of sampled points in (rho, theta), or a TrigExpansionArray"""

        assert(not (TrigExpArr is None and GridVals is None and ChebCoeffs is None))

        if TrigExpArr is None:
            if GridVals is not None:
                assert(trig_type is not None and rho1D is not None)
                TrigExpArr = TrigExpansionArray(GridVals = GridVals, trig_type = trig_type, parity = parity, rho1D = rho1D)
            else:
                assert(isinstance(TrigExpArr, TrigExpansionArray))

            self.N = np.copy(TrigExpArr.N)
            self.nCheb = 2 * self.N # half of these will be zero
            self.rho1D = np.copy(TrigExpArr.rho1D)
            self.trig_type = np.copy(TrigExpArr.trig_type)
            self.TrigCoeffGrid = np.copy(TrigExpArr.coeffGrid)
            self.GridVals = np.copy(TrigExpArr.GridVals)
            self.parity = np.copy(TrigExpArr.parity)
            self.TrigDeg = np.copy(TrigExpArr.deg)
            self.TrigExpArr = TrigExpArr # makes sure you don't modify it

            self.ChebCoeffGrid, self.ChebExpArr = self._initialize_with_TrigExp(TrigExpArr)
        else:
            self.
        
    def _initialize_with_TrigExp(self, TrigExpArr):

        TrigCoeff2Side = TrigExpArr.twoSided()
        
        ChebCoeffGrid = np.copy(TrigCoeff2Side) * 0
        ChebExpArr = np.zeros(self.TrigDeg + 1, dtype = 'object')
        for i in range(self.TrigDeg + 1):
            coeffs = vals2coeffs2(TrigCoeff2Side[:, i])
            ChebCoeffGrid[:, i] = coeffs
            ChebExpArr[i] = Chebtech2(coeffs)
        
        return ChebCoeffGrid, ChebExpArr
    
    def _initialize_with_ChebCoeff(self, ChebCoeffGrid):
        pass
    
    def spatialDerivative(self):

        newChebExpArr = np.zeros(self.nFourier, dtype = 'object')
        for i in range(self.nFourier):
            newChebExpArr[i] = self.ChebExpArr[i].diff()
        
        return ChebTrigExpansion(ChebExpArr = newChebExpArr, trig_type = self.trig_type)
    
    def angDerivative(self):

        newTrigExpArr = np.zeros(self.nCheb // 2, dtype = 'object')
        for i in range(self.nCheb // 2):
            newTrigExpArr[i] = self.TrigExpArr[i + self.nCheb // 2].derivative()

        newParity = 'even' if self.parity == 'odd' else 'odd'
        return ChebTrigExpansion(TrigExpArr = newTrigExpArr, parity = newParity)
    
    def eval(self):

        values = np.zeros([self.nCheb // 2, 2 * self.nFourier])

        for i in range(self.nCheb // 2):
            values[i] = self.TrigExpArr[i + self.nCheb // 2].evalGridFFT()

        return values