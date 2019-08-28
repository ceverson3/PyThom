from scipy.integrate import quad
import theano
import theano.tensor as tt
import numpy as np


class Integrate(theano.Op):
    def __init__(self, expr, var, *extra_vars):
        super().__init__()
        self._expr = expr
        self._var = var
        self._extra_vars = extra_vars
        self._func = theano.function(
            [var] + list(extra_vars),
            self._expr,
            on_unused_input='ignore')
    
    def make_node(self, start, stop, *extra_vars):
        self._extra_vars_node = extra_vars
        assert len(self._extra_vars) == len(extra_vars)
        self._start = start
        self._stop = stop
        vars = [start, stop] + list(extra_vars)
        return theano.Apply(self, vars, [tt.dscalar().type()])
    
    def perform(self, node, inputs, out):
        start, stop, *args = inputs
        val = quad(self._func, start, stop, args=tuple(args))[0]
        out[0][0] = np.array(val)
        
    def grad(self, inputs, grads):
        start, stop, *args = inputs
        out, = grads
        replace = dict(zip(self._extra_vars, args))
        
        replace_ = replace.copy()
        replace_[self._var] = start
        dstart = out * theano.clone(-self._expr, replace=replace_)
        
        replace_ = replace.copy()
        replace_[self._var] = stop
        dstop = out * theano.clone(self._expr, replace=replace_)

        grads = tt.grad(self._expr, self._extra_vars) 
        dargs = []
        for grad in grads:
            integrate = Integrate(grad, self._var, *self._extra_vars)
            darg = out * integrate(start, stop, *args)
            dargs.append(darg)
   
        return [dstart, dstop] + dargs


