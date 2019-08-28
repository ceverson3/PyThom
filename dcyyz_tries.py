#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:24:11 2019

@author: everson
"""


import theano.tensor as tt
from theano.compile.ops import as_op

# @as_op(itypes=[tt.lscalar], otypes=[tt.lscalar])
# def myFunc(X,V):
#     smod[0,2:] = X
#     smod[1,2:] = V
#     d1 = myBlackBox.calcD(smod)
#     d0 = myBlackBox.getObserved()
#     return (d1-d0)**2.sum()

# with pm.Model as model:
#    X = pm.Normal('X', mu=X_pr, sd=100, shape=len(X_pr))
#    V = pm.Normal('V', mu=V_pr, sd=50, shape=len(V_pr))
#    like = myFunc(X,V)
#    obs = pm.Normal('obs', mu=like, observed=d0)
#    trace = pm.sample(10000)



### Log likelihood class with gradient ####
class LogLikelihood(theano.tensor.Op):
    #itypes = [tt.dvector, tt.dvector, tt.dvector]
    itypes = [tt.scalar, tt.dscalar, tt.dscalar]
    otypes = [tt.fscalar]

    def __init__(self, fm):
        self.fm = fm
        self.grad = LoglikeGrad(self.fm)
        super(LogLikelihood, self).__init__()

    def perform(self,node,inputs,outputs):

        A, B, C = inputs
        model = np.copy(self.fm.model)
        model = np.array([A,B,C]).T
        self.fm.setModel(model)
        likelihood = self.fm.getOF()
        outputs[0][0] = np.array(likelihood)

    def grad(self, inputs, g):
        A, B, C = inputs
        return [g[0]*self.grad(A,B,C)]

### Gradient class  ###
class LoglikeGrad(theano.tensor.Op):
    #itypes = [tt.dvector, tt.dvector, tt.dvector]
    itypes = [tt.scalar, tt.dscalar, tt.dscalar]
    otypes = [tt.fvector]
    
    def __init__(self, fm):
        self.fm = fm
        super(LoglikeGrad, self).__init__()

    def perform(self,node,inputs,outputs):
        A, B, C = inputs
        model = np.copy(self.fm.model)
        model = np.array([A,B,C]).T
        self.fm.setModel(model)
        self.fm.buildFrechet()
        grads = self.fm.einSum()
        outputs[0][0] = grads

### Instantiate logLike tt.Op class
fm = myOwnClass(data,ABC_init)
logP = LogLikelihood(fm)

## Set up probablistic model
with pm.Model():
    
    A = pm.Normal('A', mu=ABC_init[-1,1], sd=200.0)
    B = pm.Normal('B', mu=ABC_init[-1,2], sd=100.0)
    C = pm.Normal('C', mu=ABC_init[-1,3], sd=0.1)
    
    # Custom likelihood
    likelihood = pm.DensityDist('likelihood', lambda A,B,C: logP(A,B,C), observed={'A':A,'B':B,'C':C})
    
    # Sample from posterior
    trace = pm.sample(draws=5000)