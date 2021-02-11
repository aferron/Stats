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
    data.sort()
    n = len(data)
    c = float((n * p) / 100) - 1
    value = 0

    if c != math.floor(c):
        c = int(math.ceil(c))
        value = data[c]
    elif c == 0:
        value = data[c]
    else:
        c = int(c)
        if c != (n):
            value = (data[c] + data[c + 1]) / 2
        else:
            value = data[c]

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
    quarts = quartiles(data)
    print("quartiles:", quarts)
    print("IQR:", quarts[2] - quarts[0])
    print("sample variance check:", sample_variance(data))



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
    print("in a negative binomial distribution, the probability of getting *up to* (<=) ", x,
            " failures before getting ", r, " successes  where the probability of success is ", p, " is: ", p_x)
    return p_x



# stats for negative binomial distribution
# r: # of successes
# p: probability of success
def neg_binom_dist_stats(r, p):
    print("\nnegative binom dist stats: ")
    mean, var = scipy.stats.nbinom.stats(r, p, moments='mv')
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
    print("probability of having *at most* ", x, " occurrences over a time of ", nu, " time units ",
            "at a rate of ", mu, " per ", ou, " time units is: ", p_x)
    return p_x



# poisson stats
# mu: mean
def poisson_stats(mu):
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
    return v_x



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





#3
# x: number of successes in the sample that you want to find the prob. for
# N: population size
# n: sample size
# M: # of successes in the population
#a = hypergeometric_dist(1000, 103, 10, 0)
#b = hypergeometric_dist(1000, 103, 10, 1)
#c = hypergeometric_dist(1000, 103, 10, 2)

#print("P(X <=2) =  ", a + b + c)

#hypergeometric_mean(1000, 103, 10)
#print(math.sqrt(hypergeometric_variance(1000, 103, 10)))

#correct:
# calculates the binomial dist for P(X <= x)
# n: # of trials
# x: # of successes
# p: probability of success
#cdf_binomial_dist(10, .103, 2)
#binomial_dist_stats(10, .103)

#4
#print(combination(6,3) / combination(12,3))

#5
#poisson_pmf(20, 1, 100, 2)
#poisson_cdf(20, 1, 100, 2)
#poisson_stats(.2 * 30)
#print(math.sqrt(6))

#6
#b
#print(combination(5,1))
#print(combination(5,1) * .04 * pow(0.8,4))
#pmf_neg_binom_dist(4, 2, .2)
#correct:
#print(combination(3,1) * .04 * pow(0.8,2))
#pmf_neg_binom_dist(2, 2, .2)

#c
#cdf = cdf_neg_binom_dist(4, 2, .2)
#print("P(X<=4) = ", 1 - cdf)
#correct
#cdf = cdf_neg_binom_dist(2, 2, .2)

#d
#neg_binom_dist_stats(2, .2)

#7
#a
#cdf_binomial_dist(25, .25, 6)
#b
#pmf_binomial_dist(25, .25, 6)
#c
#print(1 - cdf_binomial_dist(25, .25, 5))
#d
#print(1 - cdf_binomial_dist(25, .25, 6))

#8
#a
#poisson_pmf(4, 2, 1, 10)
#b
#print(scipy.stats.poisson.pmf(0, 2))
#c
#poisson_stats(2)

#9
#a
#data = [72, 74, 75, 78, 78, 85, 88, 93, 98, 114]
#stem_leaf(data)
#b
#basic_stats(data)

#12
#print(combination(20,5))

#13
#print(permutation(10, 3))

#14
#print(combination(9,3) * combination(8,2))

#practice
#poisson_pmf(1, 1, 1, 2)

#poisson_stats(20)


#1
#data = [425, 251, 510, 395, 430, 285, 375, 415, 445, 389]
#stem_leaf(data)
#basic_stats(data)
#print("Q2:", percentile(data, 50))

#4
#a
# mu: old mean
# nu: new units
# ou: old units
# x: number of occurrences
# x being 6, where the question is
# what is the probability that exactly 6 cars arrive between 12 - 12:05pm?
# if there's no need for a new mu, enter 1 for nu and ou
#poisson_cdf(14, .2, 1, 0)
#print(1 - scipy.stats.poisson.cdf(0, 2.8))
#b
#poisson_stats(2.8)

#5
# x: # of failures before the rth success
# r: # of successes
# p: probability of success
#pmf_neg_binom_dist(4, 12, .65)

#8
# n: # of trials
# p: probability of success
#binomial_dist_stats(24, .20)




