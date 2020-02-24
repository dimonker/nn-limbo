import numpy as np

import logs


def check_gradient(f, x, delta=1e-5, tol = 1e-4):
    '''
    Checks the implementation of analytical gradient by comparing
    it to numerical gradient using two-point formula

    Arguments:
      f: function that receives x and computes value and gradient
      x: np array, initial point where gradient is checked
      delta: step to compute numerical gradient
      tol: tolerance for comparing numerical and analytical gradient

    Return:
      bool indicating whether gradients match or not
    '''
    logs.m('check_gradient')
    
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float
    
    orig_x = x.copy()
    fx, analytic_grad = f(x)
    assert np.all(np.isclose(orig_x, x, tol)), "Functions shouldn't modify input variables"

    assert analytic_grad.shape == x.shape
    analytic_grad = analytic_grad.copy()
    
    logs.pr('analytic_grad', analytic_grad)
    logs.pr('x', x)

    # We will go through every dimension of x and compute numeric
    # derivative for it
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        logs.pr('ix', ix)
        
        if x.ndim == 1:
            cur_x_plus_delta = x.copy()
            cur_x_plus_delta[ix] += delta
            logs.pr('cur_x_plus_delta', cur_x_plus_delta)

            cur_x_minus_delta = x.copy()
            cur_x_minus_delta[ix] -= delta
            logs.pr('cur_x_minus_delta', cur_x_minus_delta)
        else:
            cur_x_plus_delta = x[ix[0]].copy()
            cur_x_plus_delta[ix[1]] += delta
            logs.pr('cur_x_plus_delta', cur_x_plus_delta)

            cur_x_minus_delta = x[ix[0]].copy()
            cur_x_minus_delta[ix[1]] -= delta
            logs.pr('cur_x_minus_delta', cur_x_minus_delta)

        f_val_x_plus_delta, dummy = f(cur_x_plus_delta)
        f_val_x_minus_delta, dummy = f(cur_x_minus_delta)
        logs.pr('f_val_x_plus_delta', f_val_x_plus_delta)
        logs.pr('f_val_x_minus_delta', f_val_x_minus_delta)

        numeric_grad_at_ix = (f_val_x_plus_delta - f_val_x_minus_delta) / (2 * delta)
        analytic_grad_at_ix = analytic_grad[ix]

        logs.pr('analytic_grad_at_ix', analytic_grad_at_ix)
        logs.pr('numeric_grad_at_ix', numeric_grad_at_ix)

        if not np.any(np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol)):
            print("Gradients are different at %s. Analytic: %2.5f, Numeric: %2.5f" % (ix, analytic_grad_at_ix, numeric_grad_at_ix))
            logs.me('check_gradient')
            return False

        it.iternext()

    print("Gradient check passed!")
    logs.me('check_gradient')
    return True
