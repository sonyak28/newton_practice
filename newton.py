import numpy as np

def first_derivative(x, f, small_h= 10**-5):
    """ Takes the first derivative 

    Parameters:
    f - the same function that is passed into optimize()
    x - the same integer that is passed in as starting_value into optimize()
    small_h - the same optional parameter that is passed into optimize(); default number is 10**-5
    """
    return (f(x + small_h) - f(x) ) / small_h

def second_derivative(x, f, small_h=10**-5):
    """ Takes the second derivative 

    Parameters:
    f - the same function that is passed into optimize()
    x - the same integer that is passed in as starting_value into optimize()
    small_h - the same optional parameter that is passed into optimize(); default number is 10**-5
    """    
    return (first_derivative(x + small_h, f, small_h) - first_derivative(x, f, small_h)) / small_h

def optimize(starting_value, f, small_h=10**-5):
    """ Function for Newton’s method for optimization

    Parameters:
    f - function to be passed into (ex: np.cos)
    starting values - an int (ex: 10)
    small_h - optional parameter; default number is 10**-5
    """
    # max_iterations = 10
    
    if not isinstance(starting_value, (int, float, np.number)):
        raise TypeError("`x0` must be numeric")

    new_x_t = starting_value - ((first_derivative(starting_value, f, small_h)) /  second_derivative(starting_value, f, small_h))
    x= starting_value
    # starting_value = new_x_t
    while abs(x - new_x_t) > small_h:
        x = new_x_t
        new_x_t = x - ((first_derivative(x, f, small_h)) /  second_derivative(x, f, small_h))
    return {'x': new_x_t, 'value': f(new_x_t)}

