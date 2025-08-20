import numpy as np

def first_derivative(f, x, small_h= 1e-5):
    return (f(x + small_h) - f(x) ) / small_h

def second_derivative(f, x, small_h=1e-5):
    return (first_derivative(f, x + small_h, small_h) - first_derivative(f, x, small_h)) / small_h

def optimize(f, starting_value, small_h=1e-5):
    max_iterations = 10
    history = [starting_value]
    for i in np.arange(max_iterations):

        if (history[-1] - starting_value) < 0.5 and (history[-1] != starting_value):
            break
        new_x_t = starting_value - ((first_derivative(f, starting_value, small_h)) /  second_derivative(f, starting_value, small_h))
        history.append(new_x_t)
        starting_value = new_x_t
    return starting_value, history

