# Requires Python 3, numpy >=1.7.1, and scipy >=0.14

import numpy as np
from scipy.stats import norm, ks_2samp
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages


############################################
########## CLASSES AND PARAMETERS ##########
############################################

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

# The following block of parameters are specific to the example on pg 422-423
# of "Economic Complexity"
capital_all = 0.08
prob_active_all = 0.01
lag_min = 1
lag_max = 100
len_past = lag_max + 2 # Number of extra periods needed in the beginning.




#######################################
########## RUNNING THE MODEL ##########
#######################################

# Let the first len_past prices be 0.
prices = [0]*len_past

# Instantiate traders.
fundamentalists = [ Trader('fundamentalist', capital_all, prob_active_all)
                    for i in range(num_fundamentalists) ]
chartists = [ Trader('chartist', capital_all, prob_active_all,
                     np.random.randint(lag_min, lag_max+1))
              for i in range(num_chartists) ]
traders = fundamentalists + chartists

# Run the system for the total amount of time.
fundamental_price = 0
for t in range(total_time):
    # Let the fundamental price be a random walk with steps -0.1, 0, and 0.1.
    eta = 0.1 * np.random.randint(-1,2) 
    fundamental_price = fundamental_price + eta

    # Compute the new price by Farmer's model:
    # p_{n+1} = p_n + 1/lambda * sum orders
    orders = 0
    for trader in traders:
        if trader.prob_active > np.random.random():
            orders = orders + trader.order(prices, eta)
    new_price = prices[-1] + 1/lambdaa * orders
    prices.append(new_price)

returns = [ prices[i] - prices[i-1] for i in range(1,len(prices)) ]




##############################
########## ANALYSIS ##########
##############################

# Test distribution of random subintervals vs distribution of
# random sampling of returns. Record the p-values from 
# Kolmogorov-Smirnov test.
avg_pvalues = []
lens_subinterval = [ 500*i for i in range(1,51) ]
num_subintervals = 1000
for len_subinterval in lens_subinterval:
    pvalues = []
    for i in range(num_subintervals):
        left = np.random.randint(len_past, len(returns)-len_subinterval+1)
        subinterval = returns[left:(left+len_subinterval)]
        sample_size = 10000
        sampling = np.random.choice(returns, size=sample_size, replace=False)
        (D, p) = ks_2samp(sampling, subinterval)
        pvalues.append(p)
    avg_pvalues.append(sum(pvalues) / len(pvalues))


# Plot some stuff into a single pdf.
pdf_pages = PdfPages('plots.pdf')

# Line graph of prices.
plt.plot(range(total_time), prices[len_past: ])
plt.title('Prices')
plt.xlabel('Time')
plt.ylabel('log(price)')
plt.savefig(pdf_pages, format='pdf')
plt.close()

# Scatterplot of returns.
plt.plot(range(total_time-1), returns[len_past: ], '.', markersize=3)
plt.xlim(0, total_time)
plt.title('Returns')
plt.xlabel('Time')
plt.ylabel('log(returns)')
plt.savefig(pdf_pages, format='pdf')
plt.close()

# Histogram of returns.
(mu, sigma) = norm.fit(returns)
(n, bins, patches) = plt.hist(returns, 250, normed=1)
y = plt.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=1.5)
plt.xlim(-50, 50)
plt.title('Histogram of normalized returns')
plt.savefig(pdf_pages, format='pdf')
plt.close()

# Kolmogorov-Smirnov p-values.
plt.plot(lens_subinterval, avg_pvalues)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title('Kolmogorov-Smirnov test for subintervals vs\nbackground distribution of returns')
plt.xlabel('Length of subinterval')
plt.ylabel('p-value')
plt.savefig(pdf_pages, format='pdf')
plt.close()

pdf_pages.close()
