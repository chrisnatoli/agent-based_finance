# Coded for Python 3.

import numpy as np
import matplotlib.pylab as plt
import random



# Trader class for both fundamentalists and chartists.
class Trader:
    def __init__(self, strategy, capital, prob_active, lag=-1):
        self.strategy = strategy
        self.capital = capital                         # c^(i)
        self.prob_active = prob_active                 # p^(i)
        self.lag = lag                                 # theta^(i)

    def order(self, prices, eta):                      # omega_{n+1}^(i)
        if self.strategy == 'fundamentalist':
            # omega_{n+1}^(i) = -c^(i) * (r_n - eta_n)
            return - self.capital * (prices[-1] - prices[-2] - eta)
        elif self.strategy == 'chartist':
            # omega_{n+1}^(j) = c^(j) * (r_n - r_{n-theta})
            return self.capital * ((prices[-1] - prices[-2])
                                   - (prices[-1 - self.lag]
                                      - prices[-2 - self.lag]))




# Set parameters. (All prices are log price.)
num_fundamentalists = 5000
num_chartists = 5000
lambdaa = 1
total_time = 5000
prices = [50]*102 # Let the first 102 prices be 50.

# The following block of parameters are specific to the example on pg 422-423
# of "Economic Complexity"
eta_always = 0.1
capital_all = 0.08
prob_active_all = 0.01
lag_min = 1
lag_max = 100



# Instantiate traders.
fundamentalists = [ Trader('fundamentalist', capital_all, prob_active_all)
                    for i in range(num_fundamentalists)]
chartists = [ Trader('chartist', capital_all, prob_active_all,
                     random.randint(lag_min, lag_max))
              for i in range(num_chartists) ]
traders = fundamentalists + chartists



for t in range(total_time):
    # Compute the new price by
    # p_{n+1} = p_n + 1/lambda * sum orders
    orders = 0
    for trader in traders:
        if trader.prob_active > random.random():
            orders = orders + trader.order(prices, eta_always)
    new_price = prices[-1] + 1/lambdaa * orders
    prices.append(new_price)
