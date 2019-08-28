import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
import integration as inn
import matplotlib.pyplot as plt
from MDSplus import Connection, Tree, Data


# y = [8/3,52/3,54,368/3]
# mean_error = np.zeros(len(y))
# C = [[0.1,0,0,0],[0.,0.1,0,0],[0.,0,0.1,0],[0.,0,0,0.1]] 
# error = np.random.multivariate_normal(mean_error, C)
# y_obs = y + error

N_in = [-31164.3, 3040.38, 1613.666, 460.4418, 219.852]
var_in = [3202870.7274, 17794.926, 8707.380, 4142.40, 2409.076]
analysis_tree = Tree('analysis3', 190516030)

tau = {}

for channel in np.arange(2,6):
    pass

xx = np.array([1.,2.,3.,4.])
start = theano.shared(0.)
stop = [theano.shared(x) for x in xx]

with pm.Model() as basic_model:
    # a = pm.Uniform('a', 3., 8.)
    # b = pm.Uniform('b', 0., 3.)
    
    t_e = pm.Uniform('t_e', lower=0.025, upper=100)
    n_e = pm.Uniform('n_e', lower=0, upper=10**22)
    c_geom = pm.Uniform('c_geom', lower=0, upper=np.Inf)

    # Initializing theano variables with guess values?
    l = tt.dscalar('l')
    l.tag.test_value = np.zeros(())

    a_ = tt.dscalar('a_')
    a_.tag.test_value = np.ones(())*5

    b_ = tt.dscalar('b_')
    b_.tag.test_value = np.ones(())*2

    func = 

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