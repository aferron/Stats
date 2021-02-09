# practice problems for stat 551

import math
import stemgraphic
import statistics
import scipy.stats
import pandas as pd
import numpy as np
#import _tkinter
import plotly.graph_objects as go
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
from scipy.stats import hypergeom

#mpl.use('TkAgg')

# simple test to see if plotting works
# not working
# more info here: https://matplotlib.org/3.3.3/api/_as_gen/matplotlib.pyplot.plot.html
#def check_plot():
    #fig, ax = plt.plot(x,range(10))
    #fig.savefig("check_plot.pdf")


# doesn't work how I want it to
# This is the example from plotly
# https://en.wikipedia.org/wiki/Dot_plot_(statistics)
# https://plotly.com/python/dot-plots/#basic-dot-plot
def dot_plot():
    schools = ["Brown", "NYU", "Notre Dame", "Cornell", "Tufts", "Yale",
               "Dartmouth", "Chicago", "Columbia", "Duke", "Georgetown",
               "Princeton", "U.Penn", "Stanford", "MIT", "Harvard"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[72, 67, 73, 80, 76, 79, 84, 78, 86, 93, 94, 90, 92, 96, 94, 112],
        y=schools,
        marker=dict(color="crimson", size=12),
        mode="markers",
        name="Women",
    ))

    fig.add_trace(go.Scatter(
        x=[92, 94, 100, 107, 112, 114, 114, 118, 119, 124, 131, 137, 141, 151, 152, 165],
        y=schools,
        marker=dict(color="gold", size=12),
        mode="markers",
        name="Men",
    ))

    fig.update_layout(title="Gender Earnings Disparity",
                      xaxis_title="Annual Salary (in thousands)",
                      yaxis_title="School")

    fig.show()




# create a stem and leaf plot and save as a pdf
def stem_leaf(data):
    #y = pd.Series(data)
    fig, ax = stemgraphic.stem_graphic(data, scale = 10)
    fig.savefig("stem_leaf.pdf")



# create histogram
# enter the data as a list, bins could be for example:
# bins = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]
def histogram(data, bins):
    plt.hist(data, bins)

    # in general this could be used
    #plt.plot(data)

    canvas = plt.get_current_fig_manager().canvas
    agg = canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    s, (width, height) = agg.print_to_buffer()

    # convert to a numpy array
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    # pass off to PIL
    im = Image.frombytes("RGBA", (width, height), s)

    # display the image using imagemagick's display tool
    im.show()



# get the range of a list of data
def get_range(data):
    maximum = max(data)
    minimum = min(data)
    ran = maximum - minimum
    print("max:", maximum, "min:", minimum)
    print(maximum, " - ", minimum, " = ", ran)
    return ran



# get the numerator for sample variance
def sample_variance_numerator(data):
    size = len(data)
    mean = statistics.mean(data)
    sigma = 0
    for i in range(0, size):
        sigma += (data[i] - mean) ** 2
    return sigma



# get the sample variance
def sample_variance(data):
    size = len(data)
    return sample_variance_numerator(data) / (size - 1)




def boxplot(data):
    np.random.seed(1234)
    df = pd.DataFrame(data)
    boxplot = df.boxplot()

    canvas = plt.get_current_fig_manager().canvas
    agg = canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    s, (width, height) = agg.print_to_buffer()

    # convert to a numpy array
    X = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    # pass off to PIL
    im = Image.frombytes("RGBA", (width, height), s)

    # display the image using imagemagick's display tool
    im.show()




# enter the data as a list and the percentile you'd like to find
def percentile(data, p):
    n = len(data)
    c = float((n * p) / 100)
    value = 0

    if(c != math.floor(c)):
        c = math.ceil(c) - 1
        value = data[c]
    else:
        c = int(c)
        if c != (n + 1):
            value = (data[c - 1] + data[c]) / 2
        else:
            value = data[c - 1]

    return value



# enter the data to get quartiles
def quartiles(data):
    return [percentile(data, 25), percentile(data, 50), percentile(data, 75)]




def basic_stats(data):
    print("range:", get_range(data))
    print("variance:", statistics.variance(data))
    print("standard deviation:", statistics.stdev(data))
    print("mean:", statistics.mean(data))
    print("median (check this):", statistics.median(data))
    print("low median:", statistics.median_low(data))
    print("high median:", statistics.median_high(data))
    print("mode:", statistics.mode(data))
    print("quantiles:", quantiles(data))
    print("sample variance numerator:", sample_variance_numerator(data))
    print("sample variance:", sample_variance(data))



def box_data(data):
    quarts = quartiles(data)
    iqr = quarts[2] - quarts[0]
    high_fence = quarts[2] + 1.5 * iqr
    low_fence = quarts[0] - 1.5 * iqr
    outliers = []
    non_outliers = []

    for i in data:
        if i < low_fence or i > high_fence:
            outliers.append(i)
        else:
            non_outliers.append(i)

    non_o_size = len(non_outliers)

    print("quartiles:", quarts)
    print("IQR:", iqr)
    print("fences:", low_fence, ", ", high_fence)
    print("outliers:", outliers)
    print("smallest non-outlier observation: ", non_outliers[0])
    print("largest non-outlier observation: ", non_outliers[non_o_size - 1])

    boxplot(data)



# n different items are available
# select k of the n items without replacement
# order doesn't matter (as in a committee)
def combination(n, k):
    numerator = math.factorial(n)
    denominator = math.factorial(k) * math.factorial(n - k)
    return numerator / denominator



# n different items available
# select k of the n items without replacement
# order matters (as in a photo)
def permutation(n, k):
    return math.factorial(n)/math.factorial(n - k)



# binomial distribution probability for an exact value
# n: # of trials
# x: # of successes
# p: probability of success
def pmf_binomial_dist(n, p, x):
    p_x = scipy.stats.binom.pmf(x, n, p)
    print("in a binomial distribution, the probability of getting exactly", x, " successes ",
            " in ", n, " trials where the probability of success is ", p, " is: ", p_x)
    return p_x



# binomial distribution probability
# calculates the binomial dist for P(X <= x)
# n: # of trials
# x: # of successes
# p: probability of success
def cdf_binomial_dist(n, p, x):
    p_x = 0

    for i in range(0, x + 1):
        p_x += scipy.stats.binom.pmf(i, n, p)

    print("in a binomial distribution, the probability of getting less than or equal to", x, " successes ",
            " in ", n, " trials where the probability of success is ", p, " is: ", p_x)
    return p_x



# binomial distribution stats
# n: # of trials
# p: probability of success
def binomial_dist_stats(n, p):
    e_x = n * p
    v_x = e_x * (1 - p)
    s_x = math.sqrt(v_x)

    print("\nbinomial dist stats:")
    print("expected value/mean: ", e_x)
    print("variance: ", v_x)
    print("standard deviation: ", s_x)
    print()



# probability with negative binomial distribution
# x: # of failures before the rth success
# r: # of successes
# p: probability of success
def pmf_neg_binom_dist(x, r, p):
    p_x = scipy.stats.nbinom.pmf(x, r, p)
    print("in a negative binomial distribution, the probability of getting ", x, " failures before getting ",
            r, " successes  where the probability of success is ", p, " is: ", p_x)
    return p_x



# probability with negative binomial distribution
# x: # of failures before the rth success
# r: # of successes
# p: probability of success
def cdf_neg_binom_dist(x, r, p):
    p_x = scipy.stats.nbinom.cdf(x, r, p)
    print("in a negative binomial distribution, the probability of getting at least", x, " failures before getting ",
            r, " successes  where the probability of success is ", p, " is: ", p_x)
    return p_x



# stats for negative binomial distribution
# r: # of successes
# p: probability of success
def neg_binom_dist_stats(r, p):
    print("\nnegative binom dist stats: ")
    mean, var = scipy.stats.nbinom.stats(4, .6, moments='mv')
    print("expected value/mean: ", mean, "\nvariance: ", var, "\n")



# new mean
# om: old mean
# nu: new units
# ou: old units
def new_mean(om, nu, ou):
    return om * (nu / ou)



# poisson pmf
# mu: old mean
# nu: new units
# ou: old units
# x: number of occurrences
# x being 6, where the question is
# what is the probability that exactly 6 cars arrive between 12 - 12:05pm?
# if there's no need for a new mu, enter 1 for nu and ou
def poisson_pmf(mu, nu, ou, x):
    new_mu = new_mean(mu, nu, ou)
    p_x = scipy.stats.poisson.pmf(x, new_mu)
    print("probability of having ", x, " occurrences over a time of ", nu, " time units ",
            "at a rate of ", mu, " per ", ou, " time units is: ", p_x)
    return p_x


# poisson cdf
# mu: old mean
# nu: new units
# ou: old units
# x: number of occurrences
# x being 6, where the question is
# what is the probability that exactly 6 cars arrive between 12 - 12:05pm?
# if there's no need for a new mu, enter 1 for nu and ou
def poisson_cdf(mu, nu, ou, x):
    new_mu = new_mean(mu, nu, ou)
    p_x = scipy.stats.poisson.cdf(x, new_mu)
    print("probability of having *at least* ", x, " occurrences over a time of ", nu, " time units ",
            "at a rate of ", mu, " per ", ou, " time units is: ", p_x)
    return p_x



# poisson stats
# mu: mean
# x: number of occurrences
def poisson_stats(mu, x):
    mean, var = scipy.stats.poisson.stats(mu, moments='mv')
    print("\npoisson stats:")
    print("mean", mean, "\nvariance", var)



# hypergeometric distribution for a specific x value
# x: number of successes in the sample that you want to find the prob. for
# N: population size
# n: sample size
# M: # of successes in the population
def hypergeometric_dist(N, M, n, x):
    numerator_1 = combination(M, x)
    numerator_2 = combination(N - M, n - x)
    denominator = combination(N, n)
    p_x = (numerator_1 * numerator_2) / denominator
    print("Probability of seeing ", x, " items in a subset of ", n, " items, where the ",
            "population is ", N, " and the number of successes is ", M, " is: ", p_x)
    return p_x



# hypergeometric mean
# N: population size
# n: sample size
# M: # of successes in the population
def hypergeometric_mean(N, M, n):
    e_x = (n * M) / N
    print("mean of hypergeometric dist in a subset of size ", n, " where the population size is ", N,
            " and the number of successes is ", M, " is: ", e_x)
    return e_x



# hypergeometric variance
# N: population size
# n: sample size
# M: # of successes in the population
def hypergeometric_variance(N, M, n):
    A = (N - n) / (N - 1)
    B = n
    C = M / N
    D = 1 - (M / N)
    v_x = A * B * C * D
    print("variance of hypergeometric distribution in a subset of size ", n, " where the population size is ", N,
            " and the number of successes is ", M, " is: ", v_x)



# hypergeometric distribution using scipy.stats
# N: population size
# n: sample size
# M: # of successes in the population
# not working
# Error: Matplotlib is currently using agg, which is a non-GUI backend, so cannot show the figure.
def hypergeometric_dist_plot(N, M, n):
   rv = hypergeom(M, n, N)
   x = np.arange(0, n+1)
   pmf_dogs = rv.pmf(x)

   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.plot(x, pmf_dogs, 'bo')
   ax.vlines(x, 0, pmf_dogs, lw=2)
   ax.set_xlabel('# of dogs in our group of chosen animals')
   ax.set_ylabel('hypergeom PMF')
   plt.show()






poisson_pmf(30, .3333, 1, 5)

