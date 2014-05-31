# Coded for Python 3.

import random
import numpy as np
from scipy.stats import norm
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages



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
total_time = 50000
num_trials = 1

# The following block of parameters are specific to the example on pg 422-423
# of "Economic Complexity"
capital_all = 0.08
prob_active_all = 0.01
lag_min = 1
lag_max = 100
len_past = lag_max + 2 # Number of extra periods needed in the beginning.





pdf_pages = PdfPages('plots_of_prices_and_returns.pdf')

for i in range(num_trials):
    # Let the first len_past prices be 0.
    prices = [0]*len_past

    # Instantiate traders.
    fundamentalists = [ Trader('fundamentalist', capital_all, prob_active_all)
                        for i in range(num_fundamentalists) ]
    chartists = [ Trader('chartist', capital_all, prob_active_all,
                         random.randint(lag_min, lag_max))
                  for i in range(num_chartists) ]
    traders = fundamentalists + chartists

    # Run the system for the total amount of time.
    fundamental_price = 0
    for t in range(total_time):
        # Let the fundamental price be a random walk with step size 0.1.
        eta = 0.1 * random.randint(-1,1)
        fundamental_price = fundamental_price + eta

        # Compute the new price by Farmer's model:
        # p_{n+1} = p_n + 1/lambda * sum orders
        orders = 0
        for trader in traders:
            if trader.prob_active > random.random():
                orders = orders + trader.order(prices, eta)
        new_price = prices[-1] + 1/lambdaa * orders
        prices.append(new_price)



    # Plot some stuff.
    L = [ trader.lag for trader in chartists ]
    for i in range(1,102):
        if i not in L:
            print(i)
    plt.hist(L, 100)
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

    plt.plot(range(total_time), prices[len_past: ])
    plt.xlabel('Time')
    plt.ylabel('log(price)')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

    returns = [ prices[i] - prices[i-1] for i in range(1,len(prices)) ]
    plt.plot(range(total_time-1), returns[len_past: ])
    plt.xlabel('Time')
    plt.ylabel('log(returns)')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

    (mu, sigma) = norm.fit(returns)
    (n, bins, patches) = plt.hist(returns, 100, normed=1)
    y = plt.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=1.5)
    plt.title('Histogram of normalized returns')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()



pdf_pages.close()

