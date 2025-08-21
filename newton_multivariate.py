import numpy as np
from autograd import grad, hessian
from scipy.linalg import solve

def optimize(x0, f):
    if not callable(f):
       raise TypeError(f"Argument `f` is not a function, it is of type {type(f)}")
    # if not np.isreal(x0):
    #    raise TypeError(f"Argument `x0` is not numeric")
    if not isinstance(x0, (list, tuple, np.array)):
        raise AttributeError("'x0' must be a vector input!")
    x = np.array(x0, dtype=float)
    if x.ndim != 1:
        raise TypeError("`x0` must be a 1D vector")
        
    grad_f = grad(f)
    hess_f = hessian(f)
    for i in np.arange(100):
        g = grad_f(x)
        H = hess_f(x)
        new_x = solve(H, g)
        x = new_x
    
    return {"x": new_x, "value": f(new_x)}

