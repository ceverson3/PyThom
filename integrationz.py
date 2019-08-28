#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:01:16 2019

@author: everson
"""


from scipy.integrate import trapz
import theano
import theano.tensor as tt
import numpy as np


class Integrate(theano.Op):
    def __init__(self, expr, dom, *extra_vars):
        super().__init__()
        self._expr = expr
        self._dom = dom
        self._extra_vars = extra_vars
        self._func = theano.function(
            [dom] + list(extra_vars),
            self._expr,
            on_unused_input='ignore')
    
    def make_node(self, x, *extra_vars):
        self._extra_vars_node = extra_vars
        assert len(self._extra_vars) == len(extra_vars)
        self._x = x
        vars = [x] + list(extra_vars)
        return theano.Apply(self, vars, [tt.dscalar().type()])
    
    def perform(self, node, inputs, out):
        x = inputs
        val = trapz(self._func, x)
        out[0][0] = np.array(val)
        
    def grad(self, inputs, grads):
        x, *args = inputs
        out, = grads
        replace = dict(zip(self._extra_vars, args))
        
        replace_ = replace.copy()
        replace_[self._var] = x
        dx = out * theano.clone(-self._expr, replace=replace_)

        grads = tt.grad(self._expr, self._extra_vars)
        dargs = []
        for grad in grads:
            integrate = Integrate(grad, self._var, *self._extra_vars)
            darg = out * integrate(x, *args)
            dargs.append(darg)
   
        return [dx] + dargs

