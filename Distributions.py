##Distributions
import math
import numpy as np

def dbinom(x, n, p):
	''' Provided a x, n, and p of a binomial distribution, this will return the probability
	of x occurring (pmf)
	'''
	return (math.factorial(n)/(math.factorial(x)*math.factorial(n-x)))*(p**x)*((1-p)**(n-x))

def pbinom(x, n, p):
	''' Provided a x, n, and p of a binomial distribution, this will return the probability
	of x or a value less than x occurring (cdf)'''
	y = 0
	for i in xrange(x+1):
		y += dbinom(i, n, p)
	return y

def dpois(x, lam):
	'''Provided an x and lambda, give back the probability of x occurring.'''
	return (math.exp(-lam)*(lam**x))/math.factorial(x)

def ppois(x, lam):
	'''Provided an x and lambda, give back the probability of x occurring.'''
	y = 0
	for i in xrange(x+1):
		y += dpois(i, lam)
	return y


'''List of Distributions: 
http://docs.scipy.org/doc/scipy/reference/stats.html'''




"""
file     rstats.py
desc     renaming of Python Scipy functions to match with R.
author   Dr. Ernesto P. Adorio
         UPDEPP at Clark Field
         Pampanga, the Philippines
email    ernesto.adorio@gmail.com
version  mar 29, 2009 initial release. (basic probability distributions)
"""
 
import scipy
import scipy.stats as stat #In the future use scs
 
#t     = scipy.transpose     # transpose of a matrix.
#mean  = stat.mean
#var   = stat.var
#sd    = stat.std
 
'''Give x, loc, scale for pnorm, 
give q, loc, scale for qnorm, 
give loc, scale, size for rnorm'''

dnorm = stat.norm.pdf
pnorm = stat.norm.cdf
qnorm = stat.norm.ppf
rnorm = stat.norm.rvs

'''Give x, df for pt, 
give q, df for qt, 
give df, size for rt''' 

dt = stat.t.pdf
pt = stat.t.cdf
qt = stat.t.ppf
rt = stat.t.rvs

'''Give x, ndf, ddf for pf, 
give q, ndf, ddf for qf, 
give ndf, ddf, size, for rf''' 
 
df = stat.f.pdf
pf = stat.f.cdf
qf = stat.f.ppf
rf = stat.f.rvs

'''Remember chisq is a gamma with beta = 2, and alpha = df/2
The chisq is also the sum of squared N(0,1) R.V.s where df = n
(the number of rvs you are summing together)
Give x, df, for pchisq 
give q, df for qchisq, 
give df, size  rchisq''' 
 
dchisq = stat.chi2.pdf
pchisq = stat.chi2.cdf
qchisq = stat.chi2.ppf
rchisq = stat.chi2.rvs

'''This is the continuous uniform
Give x, min, max for punif, 
give q, min, max for qunif, 
give min, max size for runif''' 

dunif = stat.uniform.pdf
punif = stat.uniform.cdf
qunif = stat.uniform.ppf
runif = stat.uniform.rvs

'''Give x, n, p for pbinom, 
give q, n, p for qbinom, 
give n, p, size for rbinom''' 

dbinom = stat.binom.pmf
pbinom = stat.binom.cdf
qbinom = stat.binom.ppf
rbinom = stat.binom.rvs

'''Give x, mean for ppois, 
give q, mean for qpois, 
give mean, size for rpois''' 
 
dpois = stat.poisson.pmf
ppois = stat.poisson.cdf
qpois = stat.poisson.ppf
rpois = stat.poisson.rvs

'''Give x, a = alpha, scale = 1/beta for pgamma  
give q, a and scale for qgamma, 
give size, a and scale for rgamma''' 

#Need to specify a = alpha and scale = 1/beta 
dgamma = stat.gamma.pdf
pgamma = stat.gamma.cdf
qgamma = stat.gamma.ppf
rgamma = stat.gamma.rvs

'''Give x, p for pgeom, 
give q, p for qgeom, 
give p, size for rt''' 
 
#Defined as the number of failures until your first success - different than definition in R
dgeom = stat.geom.pmf
pgeom = stat.geom.cdf
qgeom = stat.geom.ppf
rgeom = stat.geom.rvs

'''Give x, scale for pexp, 
give q, scale for qexp, 
give scale, size for rexp''' 
 
#Exponential with scale = mean of exponential
dexp = stat.expon.pdf
pexp = stat.expon.cdf
qexp = stat.expon.ppf
rexp = stat.expon.rvs

'''Give x, alpha, beta for pbeta, 
give q, alpha, beta for qbeta, 
give alpha, beta, and size for rbeta''' 
 
dbeta = stat.beta.pdf
pbeta = stat.beta.cdf
qbeta = stat.beta.ppf
rbeta = stat.beta.rvs

'''We might plot the data as a histogram, or plot the kde.
We can do that with the following:'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

x = np.array(rbeta(1, 1, size = 10000))
plt.hist(x)
plt.xlabel('Random Variable Value')
plt.ylabel('Count')
plt.title('Beta 1,1 Random Variable')
plt.show()

y = np.array(rbeta(500,500, size = 10000))
plt.hist(y)
plt.xlabel('Random Variable Value')
plt.ylabel('Count')
plt.title('Beta 500,500 Random Variable')
plt.show()

y1 = np.array(rexp(scale = 2, size = 10000))
plt.hist(y1)
plt.xlabel('Random Variable Value')
plt.ylabel('Count')
plt.title('Exponential Random Variable')
plt.show()



#We could take a look at the distributions in a similar way
'''Because we have the data stored in an np.array(), 
We can look at the probability of different events 
happening based on our random sample in the following way:'''

len(x[x>.5])/float(len(x))
len(y[y>.5])/float(len(y))

'''Plot the KDEs for Each Beta Distribution'''


###This stuff still doesnt work yet....
beta11 = dbeta(x_vals, 1, 1)
beta51 = dbeta(x_vals, 500, 500)
plt.plot(x_vals, beta11)
plt.plot(x_vals, beta51)
plt.show()


def Test():
  X = range(1,10)
  print X
  print mean(X)
  print sd(X)
  print var(X)
 
if __name__ == "__main__":
  Test()