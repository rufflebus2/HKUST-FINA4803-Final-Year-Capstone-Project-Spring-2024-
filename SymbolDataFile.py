#region imports
from AlgorithmImports import *
#endregion

class SymbolData:

    def __init__(self, algorithm, symbol, lookback):
        self.algorithm = algorithm
        self.symbol = symbol
        self.lookback = lookback
        self.updated = False

        # To store the historical daily log return
        self.window = RollingWindow[IndicatorDataPoint](lookback)

        # Use daily log return to predict cointegrating vector
        self.consolidator = QuoteBarConsolidator(timedelta(hours=1))
        self.price = Identity(f"{symbol} Price")
        self.price.Updated += self.OnUpdate

        # Subscribe the consolidator and indicator to data for automatic update
        algorithm.RegisterIndicator(symbol, self.price, self.consolidator)
        algorithm.SubscriptionManager.AddConsolidator(symbol, self.consolidator)

        # historical warm-up on the log return indicator
        history = algorithm.History[QuoteBar](self.symbol, self.lookback, Resolution.Hour)
        for bar in history:
            self.consolidator.Update(bar)

    def OnUpdate(self, sender, updated):
        self.window.Add(IndicatorDataPoint(updated.EndTime, updated.Value))
        self.updated = True

    def Dispose(self):
        self.price.Updated -= self.OnUpdate
        self.price.Reset()
        self.window.Reset()
        self.algorithm.SubscriptionManager.RemoveConsolidator(self.symbol, self.consolidator)

    @property
    def IsReady(self):
        return self.window.IsReady

    @property
    def Price(self):
        return pd.Series(
            data = [x.Value for x in self.window],
            index = [x.EndTime for x in self.window])[::-1]