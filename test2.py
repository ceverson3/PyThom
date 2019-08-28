import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import integration as inn
import matplotlib.pyplot as plt


y = [8/3,52/3,54,368/3]
mean_error = np.zeros(len(y))
C = [[0.1,0,0,0],[0.,0.1,0,0],[0.,0,0.1,0],[0.,0,0,0.1]] 
error = np.random.multivariate_normal(mean_error, C)
y_obs = y + error

xx = np.array([1.,2.,3.,4.])
start = theano.shared(0.)
stop = [theano.shared(x) for x in xx]


with pm.Model() as basic_model:
    a = pm.Uniform('a', 3., 8.)
    b = pm.Uniform('b', 0., 3.)

    t = tt.dscalar('t')
    t.tag.test_value = np.zeros(())

    a_ = tt.dscalar('a_')
    a_.tag.test_value = np.ones(())*5

    b_ = tt.dscalar('b_')
    b_.tag.test_value = np.ones(())*2

    func = a_*t**2 + b_*t

    integrate = inn.Integrate(func,t,a_,b_)
    mu = integrate(start,stop[-4],a,b)


    #step = pm.Metropolis()
    step = None
    #step=pm.SMC()
    #step=pm.HamiltonianMC()

    y = pm.Normal('y', mu=mu, sd=0.1, observed=y_obs)
    trace = pm.sample(2000,tune=1500, cores=2,chains=2,step=step)

pm.traceplot(trace)
# plt.savefig('res.eps')
print(pm.summary(trace))