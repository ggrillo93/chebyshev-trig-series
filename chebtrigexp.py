from trigexp import *
from findiff import FinDiff
from chebpy.core.algorithms import vals2coeffs2, chebpts2
from chebpy.core.chebtech import Chebtech

class ChebTrigExpansion():
    def __init__(self, GridVals = None, TrigExpArr = None, ChebExpArr = None, parity = None, trig_type = None, rho1D = None):
        """ Chebyshev-trigonometric expansion in polar coordinates. The angular dependence is represented by a cosine or sine series at each radial coordinate, 
            and the radial dependency of the coefficients of these series are treated with parity restricted Chebyshev polynomials. 
            Can initialize with a 2D grid of sampled points in (rho, theta), an array of TrigExpansions, one for each rho, or an array of Chebyshev expansions, 
            one for each Fourier coefficient. """

        assert(not (type(TrigExpArr) == type(None) and (type(ChebExpArr) == type(None)) and (type(GridVals) != None)))

        if type(TrigExpArr) != None:

            assert(all(isinstance(item, self.trig_type) for item in TrigExpArr)) # make sure all trigonometric expansions are of the same type
            assert(all(item.deg == self.deg for item in TrigExpArr)) # make sure all expansion have the same degree (could pad with zeros though)
            assert(len(TrigExpArr) % 2 == 0) # make sure the number of spatial samples is even
            assert(type(parity) != type(None) or type(rho1D) != type(None))

            if type(parity) != type(None):
                self.parity = parity
            else:
                self.calcParity(rho1D, TrigExpArr)

            self.nFourier = TrigExpArr[0].deg + 1
            self.nCheb = 2 * len(TrigExpArr) # half of these will be zero
            self.trig_type = type(TrigExpArr[0])
            self.TrigExpArr, self.TrigCoeffGrid, self.ChebCoeffGrid, self.ChebExpArr = self._initialize_with_TrigExp(TrigExpArr) # the new TrigExpArr will be 2 sided
            
        elif type(ChebExpArr) != type(None):
            self.nCheb = len(ChebExpArr[0].coeffs)

            assert(self.nCheb % 2 == 0)
            assert(trig_type == cosExpansion or trig_type == sinExpansion)

            self.trig_type = trig_type
            self.ChebExpArr = ChebExpArr
            self.nFourier = len(ChebExpArr)
            self.ChebCoeffGrid, self.TrigCoeffGrid, self.TrigExpArr = self._initialize_with_ChebExp()
        
        else:
            assert(type(trig_type) != type(None) and type)

            self.nCheb = len(GridVals)
            pass

    def _initialize_with_TrigExp(self, TrigExpArr):

        TrigExp2Side = np.zeros(self.nCheb, dtype = self.trig_type)
        TrigExp2Side[self.nCheb // 2:] = np.copy(TrigExpArr)
        TrigExp2Side[:self.nCheb // 2] = np.flip(TrigExpArr)

        mult = np.ones(self.nFourier)
        if self.parity == 'even':
            mult[1::2] = -1
        else:
            mult[::2] = -1

        if self.trig_type == cosExpansion:
            for i in range(self.nCheb // 2):
                TrigExp2Side[i] = cosExpansion(coeffs = TrigExp2Side[i].coeffs * mult)
        else:
            for i in range(self.nCheb // 2):
                TrigExp2Side[i] = sinExpansion(coeffs = TrigExp2Side[i].coeffs * mult)
        
        TrigCoeffGrid = np.zeros([self.nCheb, self.nFourier])
        for i in range(self.nCheb):
            TrigCoeffGrid[i] = TrigExp2Side.coeffs
        
        ChebCoeffGrid = np.zeros([self.nCheb, self.nFourier])
        ChebExpArr = np.zeros(self.nFourier, dtype = 'object')
        for i in range(self.nFourier):
            coeffs = vals2coeffs2(TrigCoeffGrid[:, i])
            if (i % 2 == 0 and self.parity == 'even') or (i % 2 != 0 and self.parity == 'odd'):
                coeffs[1::2] = 0
            else:
                coeffs[::2] = 0
            ChebCoeffGrid[:, i] = coeffs
            ChebExpArr[i] = Chebtech(coeffs)
        
        return TrigExp2Side, TrigCoeffGrid, ChebCoeffGrid, ChebExpArr
        
    def _initialize_with_ChebExp(self):

        ChebCoeffGrid = np.zeros([self.nCheb, self.nFourier])

        for i in range(self.nCheb):
            ChebCoeffGrid[:, i] = self.ChebExpArr[i].coeffs

        if np.allclose(ChebCoeffGrid[:, 0][1::2], 0):
            self.parity = 'even'
        else:
            self.parity = 'odd'
        
        nodes = chebpts2(self.nCheb)
        
        TrigCoeffGrid = np.zeros([self.nCheb, self.nFourier])
        for i in range(self.nFourier):
            TrigCoeffGrid[:, i] = self.ChebExpArr[i](nodes)
        
        TrigExp2Side = np.zeros(self.nCheb, dtype = self.trig_type)
        if self.trig_type == cosExpansion:
            for i in range(self.nCheb):
                TrigExp2Side[i] = cosExpansion(TrigCoeffGrid[i])
        else:
            for i in range(self.nCheb):
                TrigExp2Side[i] = sinExpansion(TrigCoeffGrid[i])
        
        return ChebCoeffGrid, TrigCoeffGrid, TrigExp2Side
    
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