import numpy as np

def newtons_method(x, error, max_iter):
    """ Inputs:
            x: the number at which to start iteration
            error: the error tolerance
            max_iter: the maximum number of iterations the function will run
        Returns
            x: the solution to the equation--f(x) == 0
            f(x): check that the algorithm converges to a solution
            count: the number of iterations needed to reach the solution
        ** f(x) and f_prime(x) must be defined elsewhere **

        This function performs newton's method starting at x until f(x)
        is within the given error of 0 or the maximum number of iterations
        are reached."""

    count = 0
    while (f(x) < (-1*error) or f(x) > error) and count < max_iter:
        x = x - (f(x)/f_prime(x))
        count += 1
    return x, f(x), count

def secant_method(a, b, error, max_iter):
    """ Inputs:
            a, b: the numbers at which to start iteration. They should be close together.
            error: the error tolerance
            max_iter: the maximum number of iterations the function will run
        Returns
            x: the solution to the equation--f(x) == 0
            f(x): check that the algorithm converges to a solution
            count: the number of iterations needed to reach the solution
        ** f(x) must be defined elsewhere **

        This function performs secant method starting at a and b until f(a)
        is within the given error of 0 or the maximum number of iterations
        are reached."""

    count = 1
    while (f(a) < (-1*error) or f(a) > error) and count < max_iter:
        temp = a
        a = a - ((f(a)*(a-b))/(f(a)-f(b)))
        b = temp
        count += 1
    return a, f(a), count

def tolerance_bisection_method(a, b, tolerance):
    """ Inputs:
            a: the positive number at which to start iteration
            b: the negative number at which to start iteration
            tolerance: the error tolerance
        Returns
            p: the solution to the equation--f(p) == 0
            n: the number of iterations needed to reach the solution
        ** f(x) must be defined elsewhere **

    This function performs the bisection method until a solution to f(x)
    is within the given tolerance of zero to solve for the root of
    the function f(x) defined above."""

    n = 1
    tolerance = abs(tolerance)
    # ensure that the inputs are valid
    if f(a)/abs(f(a)) == f(b)/abs(f(b)):
        return np.nan, n

    # check the upper and lower bounds are not zero
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b



    # perform the loop
    p = (a + b)/2
    while p <= tolerance or p >= tolerance:
        if f(p) == 0:
            return p
        if f(a)/abs(f(a)) == f(p)/abs(f(p)):
            a = p
        else:
            b = p
        p = (a + b)/2
        n += 1

    return p, n

def iteration_bisection_method(a, b, n):
    """ Inputs:
            a: the positive number at which to start iteration
            b: the negative number at which to start iteration
            n: the number of iterations to perform
        Returns
            p: the approximation of the solution to the equation--f(p) == 0
        ** f(x) must be defined elsewhere **

    This function performs the bisection method with n iterations
    to solve for the root of the function f(x) defined above."""



    # ensure that the inputs are valid
    if f(a)/abs(f(a)) == f(b)/abs(f(b)):
        return np.nan

    # check the upper and lower bounds are not zero
    if f(a) == 0:
        return a
    if f(b) == 0:
        return b

    # perform the loop
    p = (a + b)/2
    for n in range(n):
        if f(p) == 0:
            return p
        if f(a)/abs(f(a)) == f(p)/abs(f(p)):
            a = p
        else:
            b = p
        p = (a + b)/2

    return p

def fixed_point_iteration(p, tolerance):
    """ Inputs:
            p: the number at which to start iteration
            tolerance: the error tolerance to solve f(p) == 0 within
        Returns
            p: the solution to the equation--f(p) == p
            counter: the number of iterations needed to reach the solution
        ** g(x) must be defined elsewhere **

        This function performs newton's method starting at x until f(x)
        is within the given error of 0 or the maximum number of iterations
        are reached."""

    past = 1
    if past == p:
        past *= -1
    counter = 1
    while abs(p-past) >= tolerance:
        temp = p
        p = g(p)
        past = temp
        counter += 1
    return p, counter
