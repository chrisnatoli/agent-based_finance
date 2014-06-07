# Requires Python 3, numpy >=1.7.1, and scipy >=0.14

import numpy as np
from scipy.stats import norm, ks_2samp
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import multiprocessing as mp


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
total_time = 1000 #40000
ensemble_size = 64 #512

# The following block of parameters are specific to an example in Carvalho.
capital_all = 0.08
prob_active_all = 0.01
lag_min = 1
lag_max = 100
len_past = lag_max + 2 # Number of extra periods needed in the beginning.





#######################################
########## RUNNING THE MODEL ##########
#######################################

print('Beginning model')

# Compute multiple time series (ensemble members) using the same traders and
# same fundamental price series. The zeroth ensemble member is picked
# to be "truth". Each ensemble member is run in its own process.

# Instantiate separate RNGs.
lag_RNG = np.random.RandomState(111)
fundamental_RNG = np.random.RandomState(112)

# Instantiate traders.
fundamentalists = [ Trader('fundamentalist', capital_all, prob_active_all)
                    for i in range(num_fundamentalists) ]
chartists = [ Trader('chartist', capital_all, prob_active_all,
                     lag_RNG.randint(lag_min, lag_max+1))
              for i in range(num_chartists) ]
traders = fundamentalists + chartists

# The fundamental log(price) follows a random walk with steps -0.1, 0, and 0.1.
fundamental_price = 0
fundamental_prices = []
etas = []
for t in range(total_time):
    eta = 0.1 * fundamental_RNG.randint(-1,2) 
    fundamental_price = fundamental_price + eta
    fundamental_prices.append(fundamental_price)
    etas.append(eta)

# Create shared lists of price series (i.e., ensembles) for processes
# to record data to.
manager = mp.Manager()
prices_ensemble = manager.list()
returns_ensemble = manager.list()

# Put the construction of the price series in a subroutine so that it
# can be passed to the processes.
def run_ensemble_member(seed, prices_ensemble, returns_ensemble):
    activity_RNG = np.random.RandomState(seed) # Each process gets its own RNG.
    prices = [0] * len_past # Let the first len_past prices be 0.
    for t in range(total_time):
        # Market maker collect orders for each period and then computes
        # the new price according to Farmer's model:
        # p_{n+1} = p_n + 1/lambda * sum orders
        orders = 0
        for trader in traders:
            if trader.prob_active > activity_RNG.rand():
                orders = orders + trader.order(prices, etas[t])
        new_price = prices[-1] + 1/lambdaa * orders
        prices.append(new_price)
    returns = [ prices[i] - prices[i-1] for i in range(1,len(prices)) ]
    prices_ensemble.append(prices)
    returns_ensemble.append(returns)

# First run truth.
run_ensemble_member(113, prices_ensemble, returns_ensemble)
# Then run the ensemble, one ensemble member per process.
processes = []
for e in range(ensemble_size):
    seed = e
    process = mp.Process(target=run_ensemble_member,
                         args=(seed, prices_ensemble, returns_ensemble))
    processes.append(process)
    process.start()
for process in processes: # Wait till they all finish.
    process.join()
returns = returns_ensemble[0] # Denote the "true" returns by returns.





##############################
########## ANALYSIS ##########
##############################

print('Beginning isopleths plot')

# Plot some stuff into a single pdf.
pdf_pages = PdfPages('plots.pdf')


##############
# Line graph of prices and isopleths.
(fig, ax) = plt.subplots()

# Plot truth in black.
ax.plot(range(total_time), prices_ensemble[0][len_past: ],
        color='k', linewidth=1)

# Plot fundamental price in purple.
ax.plot(range(total_time), fundamental_prices, color='purple',
        linewidth=1, linestyle='--')

# Shade in the regions between 1% and 99% isopleths, between 10% and
# 90% isopleths, etc.
percents = (.01, .1, .2, .3, .4)

# Isopleth points is a 2d array of isopleths on one axis and time on the
# other axis, but flattened into a 1d array since multiprocessing in Python
# has problems with lists of lists.
manager = mp.Manager()
isopleth_points = manager.list([None] * (2*len(percents)*total_time))

# For a single period in time, figure out the 1% and the 99%, etc.,
# points on the respective isopleth. Put this in a subroutine
# so it can be split among processes.
def get_isopleth_points(t, isopleth_points):
    prices = [ price_series[len_past + t]
               for price_series in prices_ensemble[1: ] ]
    prices.sort()
    for i in range(len(percents)):
        lower_bound = int( len(prices) * percents[i] )
        upper_bound = int( len(prices) * (1-percents[i]) )
        minimum = prices[lower_bound]
        maximum = prices[upper_bound]
        isopleth_points[t * 2*len(percents) + i] = minimum
        isopleth_points[(t+1) * 2*len(percents) - (i+1)] = maximum

# Create one process for each point in time to compute
# the 2*len(percents) isopleths at that point.
processes = []
for t in range(total_time):
    process = mp.Process(target=get_isopleth_points,
                         args=(t, isopleth_points))
    processes.append(process)
    process.start()
for process in processes: # Wait till they all finish.
    process.join()

# Cut isopleth_points into 2d array.
isopleths = [ [ isopleth_points[t * 2*len(percents) + i]
                for t in range(total_time) ]
              for i in range(2*len(percents)) ]

# Color in the regions between 1% and 99% isopleths, etc.
for i in range(len(percents)):
    ax.fill_between(range(total_time), isopleths[i], isopleths[-i-1],
                    facecolor='green', linewidth=0, alpha=0.2)

plt.title('Prices')
plt.xlabel('Time')
plt.ylabel('log(price)')
plt.savefig(pdf_pages, format='pdf')
plt.close()



##############
# Scatterplot of returns.
plt.plot(range(total_time-1), returns[len_past: ], '.', markersize=3, color='k')
plt.xlim(0, total_time)
plt.title('Returns')
plt.xlabel('Time')
plt.ylabel('log(returns)')
plt.savefig(pdf_pages, format='pdf')
plt.close()



##############
# Histogram of returns.
(mu, sigma) = norm.fit(returns)
(n, bins, patches) = plt.hist(returns, 200, normed=1, color='k')
y = plt.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=1.5)
plt.xlim(-50, 50)
plt.title('Histogram of normalized returns')
plt.savefig(pdf_pages, format='pdf')
plt.close()



print('Beginning relative entropy plot')

##############
# Relative entropy (aka Kullback-Leibler).
# First compute the background distribution.
num_bins = 20
num_periods = int(total_time / 8)
prices = [ prices_series[len_past + t] for prices_series in prices_ensemble
           for t in range(total_time - num_periods, total_time) ]
prices = prices[(len(prices) % num_bins): ] # Make it divisible by num_bins.
prices.sort()
bin_size = len(prices) / num_bins
bin_size = int(bin_size)
bin_edges = [ (prices[i*bin_size - 1] + prices[i*bin_size]) / 2
              for i in range(1, num_bins) ]

# At each point in time, compute the relative entropy of the ensemble
# with respect to the background distribution.
relative_entropies = []
for t in range(0, total_time, 100):
    prices = [ price_series[len_past + t] for price_series in prices_ensemble ]
    probabilities = []
    for i in range(num_bins - 1):
        if i == 0:
            num_in_bin = len([ price for price in prices
                               if price <= bin_edges[i] ])
        elif i == num_bins - 2:
            num_in_bin = len([ price for price in prices
                               if price > bin_edges[i] ])
        else:
            num_in_bin = len([ price for price in prices
                               if price > bin_edges[i-1]
                               and price <= bin_edges[i] ])
        probability = num_in_bin / ensemble_size
        probabilities.append(probability)
    q = 1 / num_bins
    relative_entropy = sum([ np.log(p / q) * p for p in probabilities
                             if p > 0 ])
    relative_entropies.append(relative_entropy)

# Plot the relative entropy vs time.
plt.plot(range(0, total_time, 100), relative_entropies, color='g')
plt.title('Relative entropy of ensemble at time $t$\nw.r.t. background distribution')
plt.xlabel('Time')
plt.ylabel('Relative entropy')
plt.savefig(pdf_pages, format='pdf')
plt.close()





'''
# Test distribution of random subintervals vs distribution of random
# sampling of returns. Record the p-values from Kolmogorov-Smirnov
# test.
KStest_RNG = np.random.RandomState()
sample_size = int(total_time / 5)
avg_pvalues = []
lens_subinterval = np.arange(500, 10500, 500)
num_subintervals = 500
for len_subinterval in lens_subinterval:
    pvalues = []
    for i in range(num_subintervals):
        left = KStest_RNG.randint(len_past, len(returns)-len_subinterval+1)
        subinterval = returns[left:(left+len_subinterval)]
        sampling = KStest_RNG.choice(returns, size=sample_size, replace=False)
        (D, p) = ks_2samp(sampling, subinterval)
        pvalues.append(p)
    avg_pvalues.append(sum(pvalues) / len(pvalues))

# Kolmogorov-Smirnov p-values.
plt.plot(lens_subinterval, avg_pvalues)
plt.axhline(y=0.05, color='r', linestyle='--')
plt.title('Kolmogorov-Smirnov test for subintervals vs\nbackground distribution of returns')
plt.xlabel('Length of subinterval')
plt.ylabel('p-value')
plt.savefig(pdf_pages, format='pdf')
plt.close()
'''

pdf_pages.close()
