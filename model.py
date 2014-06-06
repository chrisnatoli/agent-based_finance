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
total_time = 1000 #35000
ensemble_size = 32 #512

# The following block of parameters are specific to an example in Carvalho.
capital_all = 0.08
prob_active_all = 0.01
lag_min = 1
lag_max = 100
len_past = lag_max + 2 # Number of extra periods needed in the beginning.




#######################################
########## RUNNING THE MODEL ##########
#######################################

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
etas = []
for t in range(total_time):
    eta = 0.1 * fundamental_RNG.randint(-1,2) 
    fundamental_price = fundamental_price + eta
    etas.append(eta)    

# Compute multiple time series (ensemble members) with the same traders and
# same fundamental price series. The zeroth ensemble member is picked
# to be "truth".
manager = mp.Manager()
prices_ensemble = manager.list() # Shared lists.
returns_ensemble = manager.list()

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

run_ensemble_member(113, prices_ensemble, returns_ensemble) # First run truth.
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

'''
# Test distribution of random subintervals vs distribution of
# random sampling of returns. Record the p-values from 
# Kolmogorov-Smirnov test.
KStest_RNG = np.random.RandomState()
sample_size = int(total_time / 5)
avg_pvalues = []pp
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
'''

# Plot some stuff into a single pdf.
pdf_pages = PdfPages('plots.pdf')

# Line graph of prices.
(fig, ax) = plt.subplots()
# Plot truth in black.
ax.plot(range(total_time), prices_ensemble[0][len_past: ],
        color='k', linewidth=1)
# Compute the mean price and std_dev across ensemble for each period.
means = []
std_devs = []
for t in range(total_time):
    prices = [ price_series[len_past + t]
               for price_series in prices_ensemble[1: ] ]
    means.append(np.mean(prices))
    std_devs.append(np.std(prices))
# Compute lines for +/- 0.5, 1, 1.5, 2, 2.5, 3 stddevs.
ks = (0.5, 1, 1.5, 2, 2.5, 3)
means_plus_ks = [ [ means[t] + k*std_devs[t] for t in range(total_time) ]
                  for k in ks ]
means_minus_ks = [ [ means[t] - k*std_devs[t] for t in range(total_time) ]
                   for k in ks ]
# Plot "histogram from above" of mean prices.
for k in range(len(ks)):
    ax.fill_between(range(total_time), means_plus_ks[k], means_minus_ks[k],
                    facecolor='green', linewidth=0, alpha=0.2)
'''
# The following plots all price series in the ensemble. It was
# scrapped in favor of the above "histogram from above" approach.
for prices in prices_ensemble[1: ]:
    ax.plot(range(total_time), prices[len_past: ], alpha=0.3,
            color='g', linewidth=1)
'''
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
(n, bins, patches) = plt.hist(returns, 50, normed=1)
y = plt.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--', linewidth=1.5)
plt.xlim(-50, 50)
plt.title('Histogram of normalized returns')
plt.savefig(pdf_pages, format='pdf')
plt.close()

'''
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
