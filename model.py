# Requires Python 3, numpy >=1.7.1, scipy >=0.14,
# matplotlib, and multiprocessing.

import numpy as np
from scipy.stats import norm, ks_2samp
import matplotlib.pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection
import multiprocessing as mp
import time




#############################################################
#################### GENERAL SUBROUTINES ####################
#############################################################

# Subroutine to cut a list into batches.
def batchify_list(liszt):
    if len(liszt) % batch_size == 0:
        return [ liszt[(batch_size*i) : (batch_size*(i+1))]
                 for i in range(int(len(liszt) / batch_size)) ]
    else:
        batches = [ liszt[(batch_size*i) : (batch_size*(i+1))]
                    for i in range(int(len(liszt) / batch_size)) ]
        leftover = liszt[-(len(liszt) % batch_size): ]
        batches.append(leftover)
        return batches



#################################################
#################### TRADERS ####################
#################################################

# Trader class for both fundamentalists and chartists.
class Trader:
    def __init__(self, strategy, capital, prob_active, lag=-1):
        self.strategy = strategy
        self.capital = capital                         # c^(i)
        self.prob_active = prob_active                 # p^(i)
        self.lag = lag                                 # theta^(i)

    def order(self, prices, fundamental_prices):       # omega_{n+1}^(i)
        if self.strategy == 'fundamentalist':
            '''
            # omega_{n+1}^(i) = -c^(i) * (r_n - eta_n)
            order = - self.capital * ((prices[-1] - prices[-2])
                                      - (fundamental_prices[-1]
                                         - fundamental_prices[-2]))
            '''
            # omega_{n+1}^(i) = c^(i) * (nu_n - p_n)
            order = self.capital * (fundamental_prices[-1] - prices[-1])

        elif self.strategy == 'chartist':
            # omega_{n+1}^(j) = c^(j) * (r_n - r_{n-theta})
            order = self.capital * ((prices[-1] - prices[-2])
                                    - (prices[-1 - self.lag]
                                       - prices[-2 - self.lag]))

        # Bound order by 1.
        order = np.sign(order) * (1 - np.exp(-np.abs(order))) 
        #print('{}\t{}'.format(self.strategy,order))
        return order

lag_RNG = np.random.RandomState(111)

def instantiate_traders():
    fundamentalists = [ Trader('fundamentalist', capital_all, prob_active_all)
                        for i in range(num_fundamentalists) ]
    chartists = [ Trader('chartist', capital_all, prob_active_all,
                         lag_RNG.randint(lag_min, lag_max+1))
                  for i in range(num_chartists) ]
    traders = fundamentalists + chartists
    return traders





###################################################
#################### THE MODEL ####################
###################################################

def construct_fundamental_prices():
    # The fundamental log(price) follows a random walk with normally
    # distributed steps with mean 0 and sigma 0.1.
    fundamental_RNG = np.random.RandomState(112)

    fundamental_prices = [0]
    for t in range(total_time):
        eta = 0.1 * fundamental_RNG.randn() 
        fundamental_price = fundamental_prices[-1] + eta
        fundamental_prices.append(fundamental_price)

    return fundamental_prices

def construct_price_series(seed, traders, fundamental_prices,
                           starting_price, starting_time,
                           is_truth,
                           prices_ensemble=None, returns_ensemble=None):
    activity_RNG = np.random.RandomState(seed) 
    # Let the first (len_past+starting_time) prices have the starting_price.
    prices = [starting_price] * (len_past + starting_time)
    for t in range(series_length):
        # Market maker collect orders for each period and then computes
        # the new price according to Farmer's model:
        # p_{n+1} = p_n + 1/lambda * sum orders
        orders = 0
        for trader in traders:
            if trader.prob_active > activity_RNG.rand():
                orders = orders + trader.order(prices,
                                               fundamental_prices[:t+2])
        new_price = prices[-1] + 1/lambdaa * orders
        prices.append(new_price)
    returns = [ prices[i] - prices[i-1] for i in range(1, len(prices)) ]

    if is_truth:
        return (prices, returns)
    else:
        prices_ensemble.append(prices)
        returns_ensemble.append(returns)

# Compute an ensemble of different time series.
def run_ensemble(traders, fundamental_prices,
                 starting_price, starting_time):
    beginning = time.time()
    
    # Create shared lists of price series (i.e., ensembles) for processes
    # to record data to.
    manager = mp.Manager()
    prices_ensemble = manager.list()
    returns_ensemble = manager.list()

    # Run the processes in batches.
    batches = batchify_list(list(range(ensemble_size)))
    for batch in batches:
        processes = []
        for e in batch:
            seed = e
            # Run one ensemble member per process.
            process = mp.Process(target=construct_price_series,
                                 args=(seed, traders, fundamental_prices,
                                       starting_price, starting_time,
                                       False, prices_ensemble,
                                       returns_ensemble))

            processes.append(process)
            process.start()
        for process in processes: # Wait till the batch finish.
            process.join()

    delta = time.time() - beginning
    print('Ensemble took [{} hrs {} min].'.format(int(delta/3600),
                                                  int((delta%3600)/60)))

    return (prices_ensemble, returns_ensemble)





##################################################
#################### ANALYSIS ####################
##################################################

def isopleths_plot(true_price_series, fundamental_prices, prices_ensemble,
                   starting_time):
    beginning = time.time()

    (fig, ax) = plt.subplots()

    # Plot truth in black.
    ax.plot(range(total_time), true_price_series[len_past: ],
            color='k', linewidth=1)

    # Plot fundamental price in orange.
    ax.plot(range(total_time), fundamental_prices[1:], color='orange',
            linewidth=1, linestyle='--')

    # Shade in the regions between 1% and 99% isopleths, between 10% and
    # 90% isopleths, etc.
    percents = (.01, .1, .2, .3, .4)

    # isopleth_points is a 2d array of isopleths on one axis and time on the
    # other axis, but flattened into a 1d array since multiprocessing in Python
    # has problems with lists of lists.
    manager = mp.Manager()
    isopleth_points = manager.list([None] * (2*len(percents) * series_length))

    # For a single period in time, figure out the 1% and the 99%, etc.,
    # points on the respective isopleth. Put this in a subroutine
    # so it can be split among processes.
    def get_isopleth_points(t, isopleth_points):
        prices = [ price_series[len_past + starting_time + t]
                   for price_series in prices_ensemble ]
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
    batches = batchify_list(list(range(series_length)))
    for batch in batches:
        processes = []
        for t in batch:
            process = mp.Process(target=get_isopleth_points,
                                 args=(t, isopleth_points))
            processes.append(process)
            process.start()
        for process in processes: # Wait till they all finish.
            process.join()

    # Cut isopleth_points into 2d array.
    isopleths = [ [ isopleth_points[t * 2*len(percents) + i]
                    for t in range(series_length) ]
                  for i in range(2*len(percents)) ]

    # Color in the regions between 1% and 99% isopleths, etc.
    for i in range(len(percents)):
        ax.fill_between(range(starting_time, total_time),
                        isopleths[i], isopleths[-i-1],
                        facecolor='green', edgecolor='none',
                        linewidth=0, alpha=0.2)

    plt.xlim(0, total_time)
    plt.title('Prices')
    plt.xlabel('Time')
    plt.ylabel('log(price)')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

    delta = time.time() - beginning
    print('Isopleths plot took [{} hrs {} min].'.format(int(delta/3600),
                                                        int((delta%3600)/60)))

def returns_lineplot(returns):
    plt.plot(range(total_time-1), returns[len_past: ], color='k')
    plt.xlim(0, total_time)
    plt.title('Returns')
    plt.xlabel('Time')
    plt.ylabel('log(returns)')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

def returns_histogram(returns):
    (mu, sigma) = norm.fit(returns)
    (n, bins, patches) = plt.hist(returns, 400, normed=1, color='k')
    y = plt.normpdf(bins, mu, sigma)
    plt.plot(bins, y, 'r--', linewidth=1.5)
    plt.xlim(-50, 50)
    plt.title('Histogram of normalized returns')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

# Take the last series_length/8 periods of an enesemble and make num_bins
# equally likely bins to estimate the background distribution.
def compute_background_distribution(prices_ensemble):
    num_periods = int(series_length / 8)
    prices = [ prices_series[len_past + t]
               for prices_series in prices_ensemble
               for t in range(series_length - num_periods, series_length) ]
    # Make it divisible by num_bins by cutting off leftover
    # in the beginning.
    prices = prices[(len(prices) % num_bins): ] 
    prices.sort()
    bin_size = len(prices) / num_bins
    bin_size = int(bin_size)
    bin_edges = [ (prices[i*bin_size - 1] + prices[i*bin_size]) / 2
                         for i in range(1, num_bins) ]
    return bin_edges

# At each 100th point in time, compute the relative entropy of the
# ensemble with respect to the background distribution.
def relative_entropy_plot(prices_ensemble, bin_edges, starting_time):
    beginning = time.time()

    step_size = 100
    manager = mp.Manager()
    relative_entropies = manager.list([None] * int(series_length / step_size))

    # Again, put the computation in a subroutine to pass to a process.
    def compute_relative_entropy(t, relative_entropies):
        prices = [ price_series[len_past + starting_time + t]
                   for price_series in prices_ensemble ]
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
        relative_entropies[int(t/step_size)] = relative_entropy

    # Run processes in batches.
    batches = batchify_list(list(range(0, series_length, step_size)))
    for batch in batches:
        processes = []
        for t in batch:
            process = mp.Process(target=compute_relative_entropy,
                                 args=(t, relative_entropies))
            processes.append(process)
            process.start()
        for process in processes:
            process.join()

    # Plot the relative entropy vs time.
    plt.plot(range(starting_time, total_time, step_size), relative_entropies, color='g')
    plt.xlim(0, total_time)
    plt.ylim(0, plt.ylim()[1])
    plt.title('Relative entropy of ensemble at time $t$\nw.r.t. background distribution')
    plt.xlabel('Time')
    plt.ylabel('Relative entropy')
    plt.savefig(pdf_pages, format='pdf')
    plt.close()

    delta = time.time() - beginning
    print('Relative entropy plot took [{} hrs {} min].'
          .format(int(delta/3600), int((delta%3600)/60)))





##############################################
#################### MAIN ####################
##############################################

time_script_begins = time.time()
pdf_pages = PdfPages('plots.pdf')

total_time = 20000#30000
ensemble_size = 200#200
batch_size = 25
num_bins = 20

num_fundamentalists = 5000
num_chartists = 5000

# The following block of parameters are specific to an example in Carvalho.
lambdaa = 1
capital_all = 0.08
prob_active_all = 0.01
lag_min = 1
lag_max = 100
len_past = lag_max + 2 # Number of extra periods needed in the beginning.

starting_time = 0
series_length = total_time - starting_time

# Run the main system.
traders = instantiate_traders()
fundamental_prices = construct_fundamental_prices()
(true_price_series,
 true_returns_series) = construct_price_series(114, traders,
                                               fundamental_prices,
                                               0, starting_time, True)
(prices_ensemble,
 returns_ensemble) = run_ensemble(traders, fundamental_prices, 0, 0)

# Analyze and plot the results.
returns_lineplot(true_returns_series)
returns_histogram(true_returns_series)
isopleths_plot(true_price_series, fundamental_prices, prices_ensemble,
               starting_time)
bin_edges = compute_background_distribution(prices_ensemble)
relative_entropy_plot(prices_ensemble, bin_edges, starting_time)

# Run a separate ensemble with a different starting time and price.
starting_price = 500
starting_time = 5000
series_length = total_time - starting_time
(late_prices_ensemble,
 late_returns_ensemble) = run_ensemble(traders, fundamental_prices,
                                       starting_price, starting_time)
isopleths_plot(true_price_series, fundamental_prices, late_prices_ensemble,
               starting_time)
relative_entropy_plot(late_prices_ensemble, bin_edges, starting_time)



pdf_pages.close()

delta = time.time() - time_script_begins
print('Entire script took [{} hrs {} min].'.format(int(delta/3600),
                                                   int((delta%3600)/60)))

#t=1000, p=400
