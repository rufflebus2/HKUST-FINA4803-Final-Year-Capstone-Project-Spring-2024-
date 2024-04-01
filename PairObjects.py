#region imports
from AlgorithmImports import *
import statsmodels.formula.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import numpy as np
#endregion

class Pairs(object):

    def __init__(self, a, b):
        # Symbols (no change)
        self.a = a
        self.b = b

        # Cointegration
        self.coint_test = -1

        # Kalman-related
        self.kalmanFilter = None
        self.trading_weight = np.array([0.0,0.0])
        self.threshold = 0
        self.currentMean = 0
        self.currentCov = 0
        self.w_volAdj = np.array([1.0, 1.0])

        # Trading
        self.state = 0
        self.trading_pair = False
        self.qty_a = 0
        self.qty_b = 0
        self.ordertime = None
        self.ticket_a = None
        self.ticket_b = None

    @property
    def DataFrame(self):
        df = pd.concat([self.a.DataFrame.droplevel([0]), self.b.DataFrame.droplevel([0])], axis=1).dropna()
        df.columns = [self.a.Symbol.Value, self.b.Symbol.Value]
        return df
    
    # @property
    # def Correlation(self):
    #     return self.DataFrame.corr().iloc[0][1]

    def resetParams(self, algorithm):
        #algorithm.Debug(f'Params reset for {self.a.symbol} and {self.b.symbol}')
        # Cointegration p-value
        self.coint_test = -1

        # Kalman-related
        self.kalmanFilter = None
        self.trading_weight = np.array([0.0,0.0])
        self.threshold = 0
        self.currentMean = 0
        self.currentCov = 0
        self.w_volAdj = np.array([1.0, 1.0])

        # Trading
        self.state = 0
        self.trading_pair = False
        self.qty_a = 0
        self.qty_b = 0
        self.ordertime = None
        self.ticket_a = None
        self.ticket_b = None

    def cointegration_test(self, algorithm, price_a, price_b):
        # index [1] is for the cointegration p-value
        if not price_a.index.equals(price_b.index):
            algorithm.Debug(f'{self.a.symbol} and {self.b.symbol} Invalid indexes')
            return False

        self.coint_test = coint(price_a, price_b, trend="n", maxlag=0)[1]
        if self.coint_test >= 0.05:
            return False
        return True

    # def __str__(self):
    #     # print out pair name, correlation and cointegration p-value
    #     return f"{self.Name} \nCorrelation: {round(self.Correlation, 3)} \nCointegration: {round(self.coint_test, 3)}"
