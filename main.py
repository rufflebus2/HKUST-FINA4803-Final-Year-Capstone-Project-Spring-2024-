#region imports
from AlgorithmImports import *
from arch.unitroot.cointegration import engle_granger
from pykalman import KalmanFilter
from itertools import combinations
from PairObjects import *
from SymbolDataFile import *
from datetime import *
from scipy.optimize import minimize
from statsmodels.tsa.vector_ar.vecm import VECM
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
#endregion

class PCADemo(QCAlgorithm):
    
    def Initialize(self): # Initialization function
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023,12,31)
        self.SetBrokerageModel(BrokerageName.Binance, AccountType.Margin)
        self.DefaultOrderProperties = BinanceOrderProperties()
        self.SetAccountCurrency("USDT")
        self.SetCash(1_000_000)
    
        self.assets = [
            'BTCUSDT', 
            'ETHUSDT',
            'ADAUSDT',
            'LTCUSDT',
            'BCHUSDT',
            'SOLUSDT',
            'XRPUSDT',
            'BNBUSDT',
            'DOTUSDT',
            'LINKUSDT'
        ] # 10 coins

        # For different modes
        self.equal_hedge = False # If False, then run cointegration test for Hedge ratio
        self.optimization = True # Must be False if equal_hedge is True
        self.includeSLTP = False
        self.volAdj = False
        if self.equal_hedge:
            self.optimization = False
        
        self.maxPairs = 5
        self.maxDrawdown = 0.05
        self.takeProfit = 0.2
        self.sdThresh = 1.01 / 100

        self.pair_list = [] # List of all 45 pairs
        self.symbol_data = {} # Symbol data of 45 pairs

        # Parameters for time checks
        self.coint_time = datetime.min
        self.kalman_time = datetime.min
        self.liq_time = datetime.min
        self.coint_interval = timedelta(days = 7) # Cointegration test every 7 days
        self.kalman_interval = timedelta(days = 7) # Run Kalman filter every day
        self.liq_interval = timedelta(days = 1) # Check potential liquidation pairs every day
        self.kalman_data = 14 # days
        self.coint_data = 50 # days
        self.time_string = None # For current time

        # For Trade and trade logging/debugging
        self.position_list = [] 
        self.numPairsInTrade = 0 # Number of pairs to trade (not necessarily in position)
        self.numPairsInPosition = 0 # Number of pairs in position
        # self.coin_percentage = {} 
        # for coin in self.assets:
        #     self.coin_percentage[coin] = [0]
        
        for coin in self.assets:
            symbol = self.AddCrypto(
                ticker = coin, 
                resolution = Resolution.Hour, 
                market = Market.Binance
            ).Symbol
            self.symbol_data[symbol] = SymbolData(
                algorithm = self,   
                symbol = symbol, 
                lookback = 91 * 24, # Past 91 days (hourly) worth of data, some data gets filtered out cuz of mismatching indexes later
            )
        
        for pair in combinations(self.symbol_data.items(), 2):
            self.pair_list.append(Pairs(pair[0][1], pair[1][1]))
    
    ########## Main functions ###########

    def OnData(self, data):
        # for symbol, symbolData in self.symbol_data.items():
        #     if data.Bars.ContainsKey(symbol):
        #         symbolData.Update(data.Bars[symbol])
        cur_time = self.Time
        time_format = '%Y-%m-%d %H:%M:%S'
        self.time_string = cur_time.strftime(time_format)

        #self.Debug(f'--------------------------- {time_string} ---------------------------')
        if self.Time >= self.liq_time:
            self.liq_time = self.Time + self.liq_interval
            self.LiquidateCheck()

        if self.Time >= self.coint_time:
            self.coint_time = self.Time + self.coint_interval
            self.CointegrationTest()

        if self.Time >= self.kalman_time:
            self.kalman_time = self.Time + self.kalman_interval
            self.RunKalman()
    
    def LiquidateCheck(self): # Liquidate pair after 90 days 
        if self.numPairsInTrade == 0:
            return

        for pair in self.pair_list:
            if pair.trading_pair:
                if pair.state == 1 or pair.state == -1:
                    days_in_position = self.Time - pair.ordertime
                    if days_in_position > timedelta(days=90):
                        self.Debug(f'Liquidate pair after 90 days: {pair.a.symbol} and {pair.b.symbol} ({self.time_string})')
                        self.LiquidatePair(pair, False)
                        self.CointegrationTest()

    def CointegrationTest(self): # Check Cointegration for all 45 pairs
        pairs_pass = 0
        pass_list = []
        for pair in self.pair_list:
            if self.symbol_data[pair.a.symbol].IsReady and self.symbol_data[pair.b.symbol].IsReady:
                # Clean prices to make sure they have the same timestamps
                price_a = self.symbol_data[pair.a.symbol].Price
                price_b = self.symbol_data[pair.b.symbol].Price
                price_a, price_b = self.clean_prices(price_a, price_b)
                price_a = price_a[-self.coint_data * 24:]
                price_b = price_b[-self.coint_data * 24:]
                if pair.cointegration_test(self, price_a, price_b) and not pair.trading_pair:
                    pairs_pass += 1
                    pass_list.append(pair)
        
        # Rank the pair's cointegration p-value from low to high
        pass_list.sort(key = lambda x: x.coint_test, reverse = False)

        # If all trading pairs are in open positions, no need to add any new pairs
        free_spots = self.maxPairs - self.numPairsInTrade
        if free_spots > 0:
            pass_list = pass_list[:free_spots]
        else:
            return
        
        for pair in pass_list:
            pair.trading_pair = True
            self.Debug(f'Added {pair.a.symbol} and {pair.b.symbol} ({self.time_string})')
        
        self.numPairsInTrade += len(pass_list)
        # self.Debug(f'{self.numPairsInTrade} current trading pairs for strategy: ')
        # for pair in self.pair_list:
        #     if pair.trading_pair:
        #         self.Debug(f'{pair.a.symbol} and {pair.b.symbol}')

    def RunKalman(self): # Not actually the filter
        for pair in self.pair_list:
            if pair.trading_pair:
                self.kfilter(pair)
                self.PositionEntryExit(pair)

    def kfilter(self, pair): # Kalman Filter function for each pair
        # Sets hedge ratio (Trading weights), mean, cov and kalman filter for pair
        # Clean prices of past 91 days of hourly data (91*24 bars)
        price_a = self.symbol_data[pair.a.symbol].Price
        price_b = self.symbol_data[pair.b.symbol].Price
        price_a, price_b = self.clean_prices(price_a, price_b)
        log_price = np.log(
            pd.DataFrame({
                pair.a.symbol: price_a,
                pair.b.symbol: price_b
            })
        )
        log_price.dropna(inplace=True)

        if log_price.empty: return

        # Pass last self.kalman_data days (hourly) to kalman (except the last hour)
        # Will use last hour data to check whether to enter or exit
        log_price = log_price[-self.kalman_data * 24:-1]
        
        # Get the weighted spread across different cointegration subspaces
        if self.equal_hedge and not self.optimization: # Use 50/50 hedge ratio
            spread, beta = self.GetSpreads(log_price)
            used_spread = spread
        elif not self.equal_hedge and self.optimization: # Use cointegration for hedge ratio and optimize
            weighted_spread, weights, beta = self.GetSpreads(log_price) # ORIGINAL IS HERE
            used_spread = weighted_spread
        elif not self.equal_hedge and not self.optimization:
            spread, beta = self.GetSpreads(log_price)
            used_spread = spread
        
        # Set up the Kalman Filter with the weighted spread series, and obtain the adjusted mean series
        # mean_series = self.SetKalmanFilter(weighted_spread, pair) # ORIGINAL IS HERE
        # mean_series = self.SetKalmanFilter1(spread, pair) # ORIGINAL IS HERE
        mean_series = self.SetKalmanFilter(used_spread, pair)

        # Obtain the normalized spread series, the first 20 in-sample will be discarded.
        # normalized_spread = (weighted_spread.iloc[20:] - mean_series) # ORIGINAL IS HERE
        normalized_spread = (used_spread.iloc[20:] - mean_series)

        # Set the threshold of price divergence to optimize profit
        self.SetTradingThreshold(normalized_spread, pair)

        # Set the normalize trading weight/Hedge ratio
        # weights = self.GetTradingWeight(beta, weights) # ORIGINAL IS HERE
        if not self.optimization:
            weights = beta / np.sum(abs(beta))
        else:
            weights = self.GetTradingWeight(beta, weights)

        # if (weights[0] > 0 and weights[1] > 0) or (weights[0] < 0 and weights[1] < 0):
        #     self.Error('Weights are the same sign')

        # If pair is in position already, no need to change hedge ratio
        if pair.state == 0:
            # pair.trading_weight = np.abs(weights) # ORIGINAL IS np.abs(weights)
            pair.trading_weight[0] = np.abs(weights[0])
            pair.trading_weight[1] = -np.abs(weights[1])
            print(1)

    def PositionEntryExit(self, pair): # Checks entry/exit positions
        price_a = self.symbol_data[pair.a.symbol].Price
        price_b = self.symbol_data[pair.b.symbol].Price
        price_a, price_b = self.clean_prices(price_a, price_b)
        print(1)
        log_price = np.log(
            pd.DataFrame({
                pair.a.symbol: price_a,
                pair.b.symbol: price_b
            })
        )
        if log_price.empty: return
        log_price = log_price.iloc[-1]
            
        # Get the spread
        spread = np.sum(log_price * pair.trading_weight)
        
        # Update the Kalman Filter with the Series
        (pair.currentMean, pair.currentCov) = pair.kalmanFilter.filter_update(filtered_state_mean = pair.currentMean,
                                                                           filtered_state_covariance = pair.currentCov,
                                                                           observation = spread)
            
        # Obtain the normalized spread.
        normalized_spread = spread - pair.currentMean
        
        if self.volAdj:
            self.volatility_adjustment(pair)

        # Mean-reversion
        if normalized_spread < -pair.threshold and pair.state == 0:
            # long spread 
            pair.state = 1
            pair.qty_a, pair.qty_b = self.order_quantities(pair)
            if pair.qty_a == 0 or pair.qty_a == 0:
                self.Error(f'Order size is zero: {pair.qty_a} {pair.a.symbol} and {pair.qty_b} {pair.b.symbol}')
                pair.state = 0
                pair.qty_a = pair.qty_b = 0
                return
            elif not self.is_valid_order_size(pair.a.symbol, pair.qty_a) or not self.is_valid_order_size(pair.b.symbol, pair.qty_b):
                self.Error(f'Invalid order size: {pair.qty_a} {pair.a.symbol} and {pair.qty_b} {pair.b.symbol}')
                pair.state = 0
                pair.qty_a = pair.qty_b = 0
                return
            elif (pair.qty_a > 0 and pair.qty_b > 0) or (pair.qty_a < 0 and pair.qty_b < 0):
                self.Error(f'Double long/short: {pair.qty_a} {pair.a.symbol} and {pair.qty_b} {pair.b.symbol}')
                pair.state = 0
                pair.qty_a = pair.qty_b = 0
                return
            self.position_list.append(f'({pair.trading_weight[0]}) {pair.qty_a} {pair.a.symbol} and ({pair.trading_weight[1]}) {pair.qty_b} {pair.b.symbol}')
            self.EnterPair(pair)
                
        elif normalized_spread > pair.threshold and pair.state == 0:
            # short spread 
            pair.state = -1
            pair.qty_a, pair.qty_b = self.order_quantities(pair)
            if pair.qty_a == 0 or pair.qty_a == 0:
                self.Error(f'Order size is zero: {pair.qty_a} {pair.a.symbol} and {pair.qty_b} {pair.b.symbol}')
                pair.state = 0
                pair.qty_a = pair.qty_b = 0
                return
            elif not self.is_valid_order_size(pair.a.symbol, pair.qty_a) or not self.is_valid_order_size(pair.b.symbol, pair.qty_b):
                self.Error(f'Invalid order size: {pair.qty_a} {pair.a.symbol} and {pair.qty_b} {pair.b.symbol}')
                pair.state = 0
                pair.qty_a = pair.qty_b = 0
                return
            elif (pair.qty_a > 0 and pair.qty_b > 0) or (pair.qty_a < 0 and pair.qty_b < 0):
                self.Error(f'Double long/short: {pair.qty_a} {pair.a.symbol} and {pair.qty_b} {pair.b.symbol}')
                pair.state = 0
                pair.qty_a = pair.qty_b = 0
                return
            self.position_list.append(f'({pair.trading_weight[0]}) {pair.qty_a} {pair.a.symbol} and ({pair.trading_weight[1]}) {pair.qty_b} {pair.b.symbol}')
            self.EnterPair(pair)
                
        # Out of position if spread recovered
        elif (pair.state == 1 and normalized_spread > -pair.threshold) or (pair.state == -1 and normalized_spread < pair.threshold):
            # out of position so liquidate pair
            self.LiquidatePair(pair, False)
            self.CointegrationTest()
        elif self.includeSLTP and (pair.state == 1 or pair.state == -1):
            tp, sl = self.runSLTP(pair)
            if tp:
                self.Debug('Take profit activated')
                self.LiquidatePair(pair, False) 
                self.CointegrationTest()
            elif sl:
                self.Debug('Stop loss activated')
                self.LiquidatePair(pair, False) 
                self.CointegrationTest()

    def EnterPair(self, pair): # Function to Enter a pair
        if not self.IsWarmingUp:
            ticket_a = self.MarketOrder(pair.a.symbol.Value, pair.qty_a)
            ticket_b = self.MarketOrder(pair.b.symbol.Value, pair.qty_b)
            pair.ticket_a = ticket_a
            pair.ticket_b = ticket_b
            self.numPairsInPosition += 1
            if pair.qty_a > 0:
                self.Debug(f'Long {pair.qty_a} {pair.a.symbol} @ ${ticket_a.AverageFillPrice}, Short {pair.qty_b} {pair.b.symbol} @ ${ticket_b.AverageFillPrice} ({self.time_string})')
            elif pair.qty_a < 0:
                self.Debug(f'Short {pair.qty_a} {pair.a.symbol} @ ${ticket_a.AverageFillPrice}, Long {pair.qty_b} {pair.b.symbol} @ ${ticket_b.AverageFillPrice} ({self.time_string})')
            pair.ordertime = self.Time

    def LiquidatePair(self, pair, trade): # Function to Liquidate a pair
        if not self.IsWarmingUp:
            ticket_a = self.MarketOrder(pair.a.symbol.Value, -pair.qty_a)
            ticket_b = self.MarketOrder(pair.b.symbol.Value, -pair.qty_b)
            self.numPairsInPosition -= 1
            #self.coin_percentage[pair.a.symbol.Value].append(-pair.state * pair.trading_weight[0])
            #self.coin_percentage[pair.b.symbol.Value].append(pair.state * pair.trading_weight[1])
            self.Debug(f'Exit Position: {pair.qty_a} {pair.a.symbol} @ ${ticket_a.AverageFillPrice} and {pair.qty_b} {pair.b.symbol} @ ${ticket_b.AverageFillPrice} ({self.time_string})')
        pair.resetParams(self)
        if trade:
            pair.trading_pair = True # In case we still trade it during the week
        else:
            pair.trading_pair = False
            self.numPairsInTrade -= 1

    ########## KALMAN STUFF BELOW (used in kfilter function) ###########

    def GetSpreads(self, logPriceDf):
        # Initialize a VECM model following the unit test parameters, then fit to our data.
        
        ####### USING VECM (SVD DOES NOT CONVERGE) #########
        # # We allow 3 AR difference, and no deterministic term.
        # vecm_result = VECM(logPriceDf, k_ar_diff=3, coint_rank=1, deterministic='n').fit()
        # # Obtain the Beta attribute. This is the cointegration subspaces' unit vectors.
        # beta = vecm_result.beta
        # # get the spread of different cointegration subspaces.
        # spread = logPriceDf @ beta

        ####### USING JOHANSEN COINT (SVD DOES NOT CONVERGE) #########
        # # Perform Johansen Cointegration test
        # result = coint_johansen(logPriceDf, det_order=0, k_ar_diff=3)
        # # Obtain the cointegration vectors
        # beta = result.evec[:, 0]  # Assuming you want the first cointegration vector
        # beta = beta.reshape(2,1)
        # # Calculate the spread
        # spread = logPriceDf @ beta

        ####### USING ENGLE-GRANGER COINTEGRATION TEST #########
        coint_result = engle_granger(logPriceDf.iloc[:, 0], logPriceDf.iloc[:, 1], trend="c", lags=0)
        beta = coint_result.cointegrating_vector[:2] # without optimization
        if self.equal_hedge:
            beta = np.array([1, -1]) # equal hedge
        if self.optimization:
            beta = np.array(beta).reshape(2,1) # with optimization

        # # Step 5: Specify the dependent and independent variables
        # y = logPriceDf[logPriceDf.columns[0]]
        # X = logPriceDf[logPriceDf.columns[1]]
        # # Adjust the variable names as needed

        # # Step 6: Add a constant term
        # X = sm.add_constant(X)

        # # Step 7: Fit the ARDL model
        # model = sm.OLS(y, X)
        # results = model.fit()

        # # Step 8: Get the beta (cointegration vectors)
        # beta = results.params[1:]  # Exclude the constant term

        spread = logPriceDf @ beta # (1000,2) * (2,1)

        if self.optimization:
            return self.OptimizeSpreads(spread, beta)
        else:
            return spread, beta

        # Optimize the distribution across cointegration subspaces and return the weighted spread
        # return self.OptimizeSpreads(spread, beta) # OGIRINAL IS HERE
        # return spread, beta

    def OptimizeSpreads(self, spread, beta):
        # We set the weight on each vector is between -1 and 1. While overall sum is 0.
        x0 = np.array([-1**i / beta.shape[1] for i in range(beta.shape[1])])
        bounds = tuple((-1, 1) for i in range(beta.shape[1]))
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x)}]

        numer = np.cov(spread.T, spread.shift(1).fillna(0).T)[spread.shape[1]:, :spread.shape[1]]
        denom = np.cov(spread.T).reshape(numer.shape)
        
        # Optimize the Portmanteau statistics
        opt = minimize(lambda w: ((w.T @ numer @ w)\
                                    / (w.T @ denom @ w))**2,
                        x0=x0,
                        bounds=bounds,
                        constraints=constraints,
                        method="SLSQP")
         
        # Normalize the result
        opt.x = opt.x / np.sum(abs(opt.x))
        
        # Return the weighted spread series
        return spread @ opt.x, opt.x, beta

    def SetKalmanFilter(self, weighted_spread, pair):
        #self.Debug('set kalman')
        # Initialize a Kalman Filter. Using the first 20 data points to optimize its initial state. 
        # We assume the market has no regime change so that the transitional matrix and observation matrix is [1].
        pair.kalmanFilter = KalmanFilter(transition_matrices = [1],
                            observation_matrices = [1],
                            initial_state_mean = weighted_spread.iloc[:20].mean(),
                            observation_covariance = weighted_spread.iloc[:20].var(),
                            em_vars=['transition_covariance', 'initial_state_covariance'])
        pair.kalmanFilter = pair.kalmanFilter.em(weighted_spread.iloc[:20], n_iter=5)
        (filtered_state_means, filtered_state_covariances) = pair.kalmanFilter.filter(weighted_spread.iloc[:20])
        
        # Obtain the current Mean and Covariance Matrix expectations.
        pair.currentMean = filtered_state_means[-1, :]
        pair.currentCov = filtered_state_covariances[-1, :]
        
        # Initialize a mean series for spread normalization using the Kalman Filter's results.
        mean_series = np.array([None]*(weighted_spread.shape[0]-20))
        
        # Roll over the Kalman Filter to obtain the mean series.
        for i in range(20, weighted_spread.shape[0]):
            (pair.currentMean, pair.currentCov) = pair.kalmanFilter.filter_update(filtered_state_mean = pair.currentMean,
                                                                    filtered_state_covariance = pair.currentCov,
                                                                    observation = weighted_spread.iloc[i])
            mean_series[i-20] = float(pair.currentMean)

        return mean_series

    def SetTradingThreshold(self, normalized_spread, pair):
        # Initialize 20 set levels for testing.
        s0 = np.linspace(0, max(normalized_spread), 20)
        
        # Calculate the profit levels using the 20 set levels.
        f_bar = np.array([None] * 20)
        for i in range(20):
            f_bar[i] = len(normalized_spread.values[normalized_spread.values > s0[i]]) \
                / normalized_spread.shape[0]
            
        # Set trading frequency matrix.
        D = np.zeros((19, 20))
        for i in range(D.shape[0]):
            D[i, i] = 1
            D[i, i+1] = -1
            
        # Set level of lambda.
        l = 1.0
        
        # Obtain the normalized profit level.
        f_star = np.linalg.inv(np.eye(20) + l * D.T @ D) @ f_bar.reshape(-1, 1)
        s_star = [f_star[i] * s0[i] for i in range(20)]
        pair.threshold = s0[s_star.index(max(s_star))]

    def GetTradingWeight(self, beta, weights):
        trading_weight = beta @ weights
        return trading_weight / np.sum(abs(trading_weight))

    ########## RISK MANAGEMENT ###########
    
    def is_valid_order_size(self, symbol, quantity): # Checks if order size is valid
        crypto = self.Securities[symbol]
        return abs(crypto.Price * quantity) > crypto.SymbolProperties.MinimumOrderSize
    
    def runSLTP(self, pair): # True if we should liquidate
        cryptoA = self.Securities[pair.a.symbol]
        cryptoB = self.Securities[pair.b.symbol]
        cur_priceA = cryptoA.Price
        cur_priceB = cryptoB.Price
        order_priceA = pair.ticket_a.AverageFillPrice
        order_priceB = pair.ticket_b.AverageFillPrice
        qA = pair.qty_a
        qB = pair.qty_b
        sl = False
        tp = False

        init_value = -(order_priceA * qA + order_priceB * qB) 
        cur_value = cur_priceA * qA + cur_priceB * qB
        #init_value = -(pair.state * order_priceA * qA - pair.state * order_priceB * qB)
        #cur_value =  (pair.state * cur_priceA * qA - pair.state * cur_priceB * qB)
        profit = init_value + cur_value
        profitPerc = profit / np.abs(init_value)
        # self.Debug(f'Initial value: {init_value}')
        # self.Debug(f'Current value: {cur_value}')
        # self.Debug(f'Profit: {profit}')
        # self.Debug(f'Profit %: {profitPerc}')

        if profitPerc < -self.maxDrawdown:
            sl = True
        elif profitPerc > self.takeProfit:
            tp = True
        
        return tp, sl

    def volatility_adjustment(self, pair):
        historyA = self.symbol_data[pair.a.symbol].Price
        historyB = self.symbol_data[pair.b.symbol].Price

        historyA, historyB = self.clean_prices(historyA, historyB)

        historyA = np.log(historyA)
        historyB = np.log(historyB)

        retA = historyA.pct_change()
        retB = historyB.pct_change()

        retA = retA[:-1] # Prevent any lookahead bias
        retB = retB[:-1] # Prevent any lookahead bias

        retA.dropna(inplace=True)
        retB.dropna(inplace=True)

        # historyA = historyA.values.reshape(-1, 1)
        # historyB = historyB.values.reshape(-1, 1)

        # scaler = MinMaxScaler()
        # historyA = scaler.fit_transform(historyA)
        # historyB = scaler.fit_transform(historyB)

        # std_a = np.std(historyA)
        # std_b = np.std(historyB)

        std_a = retA.std()
        std_b = retB.std()

        if std_a > self.sdThresh:
            pair.w_volAdj[0] = pair.trading_weight[0] * 0.01 / std_a
        else:
            pair.w_volAdj[0] = pair.trading_weight[0]
            
        if std_b > self.sdThresh:
            pair.w_volAdj[1] = pair.trading_weight[1] * 0.01 / std_b
        else:
            pair.w_volAdj[1] = pair.trading_weight[1]
        
        print(1)

    ########## Functions to make my life easier ###########

    def clean_prices(self, a, b):
        new_a = a.loc[a.index.intersection(b.index)]
        new_b = b.loc[b.index.intersection(a.index)]
        return new_a, new_b
        
    def free_cash(self, pair):
        cashInvested = 0
        constant_slippage_factor = 0.01
        for pair in self.pair_list:
            if pair.state == 1 or pair.state == -1:
                investA = np.abs(pair.qty_a) * self.Securities[pair.a.symbol].Price
                investB = np.abs(pair.qty_b) * self.Securities[pair.b.symbol].Price
                cashInvested += (1 + constant_slippage_factor) * (investA + investB)
        total_cash = self.Portfolio.CashBook.TotalValueInAccountCurrency
        # for coin in self.assets:
        #     coin = coin[:-4]
        #     cash_in_coins += np.abs(self.Portfolio.CashBook[coin].ValueInAccountCurrency)
        return total_cash - cashInvested

    def order_quantities(self, pair):
        cash = self.free_cash(pair)
        cash_for_pair = cash * (1/(self.maxPairs - self.numPairsInPosition))
        if self.volAdj:
            cashA = cash_for_pair * pair.w_volAdj[0] # pair.trading_weight[0] # pair.w_volAdj[0] 
            cashB = cash_for_pair * pair.w_volAdj[1] # pair.trading_weight[1] # pair.w_volAdj[1] 
        else:
            cashA = cash_for_pair * pair.trading_weight[0] # pair.w_volAdj[0] 
            cashB = cash_for_pair * pair.trading_weight[1] # pair.w_volAdj[1] 

        cryptoA = self.Securities[pair.a.symbol]
        cryptoB = self.Securities[pair.b.symbol]
        priceA = cryptoA.Price
        priceB = cryptoB.Price

        qty_a = pair.state * cashA / (1.001 * priceA) # Takes fees into account # pair.state * 
        qty_b = pair.state * cashB / (1.001 * priceB) # Takes fees into account # -pair.state * 

        if priceA == 0 or priceB == 0:
            print(1)

        lot_sizeA = cryptoA.SymbolProperties.LotSize
        qty_a = round(qty_a / lot_sizeA) * lot_sizeA
        lot_sizeB = cryptoB.SymbolProperties.LotSize
        qty_b = round(qty_b / lot_sizeB) * lot_sizeB
        if qty_a == 0 or qty_b == 0:
            print(1)
        return qty_a, qty_b