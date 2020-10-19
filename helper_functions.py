import numpy as np
import pandas as pd
from scipy import interpolate

def jacobi(A, b, x0, tol, n_iterations=300):
    """jacobi(A, b, x0, tol, n_iterations=300)
    Performs Jacobi iterations to solve the line system of
    equations, Ax=b, starting from an initial guess, ``x0``.

    Returns:
    x, the estimated solution
    """

    n = A.shape[0]
    x = x0.copy()
    x_prev = x0.copy()
    counter = 0
    x_diff = tol+1

    while (x_diff > tol) and (counter < n_iterations): #iteration level
        for i in range(0, n): #element wise level for x
            s = 0
            for j in range(0,n): #summation for i !=j
                if i != j:
                    s += A[i,j] * x_prev[j]

            x[i] = (b[i] - s) / A[i,i]
        #update values
        counter += 1
        x_diff = (np.sum((x-x_prev)**2))**0.5
        x_prev = x.copy() #use new x for next iteration


    # print("Number of Iterations: ", counter)
    # print("Norm of Difference: ", x_diff)
    return x

def cubic_spline(x, y, tol = 1e-100):
    """cubic_spline(x, y, tol = 1e-100)
    Interpolate using natural cubic splines.

    Generates a strictly diagonal dominant matrix then applies Jacobi's method.

    Returns coefficients:
    b, coefficient of x of degree 1
    c, coefficient of x of degree 2
    d, coefficient of x of degree 3
    """
    x = np.array(x)
    y = np.array(y)
    ### check if sorted
    if np.any(np.diff(x) < 0):
        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

    size = len(x)
    delta_x = np.diff(x)
    delta_y = np.diff(y)

    ### Get matrix A
    A = np.zeros(shape = (size,size))
    b = np.zeros(shape=(size,1))
    A[0,0] = 1
    A[-1,-1] = 1

    for i in range(1,size-1):
        A[i, i-1] = delta_x[i-1]
        A[i, i+1] = delta_x[i]
        A[i,i] = 2*(delta_x[i-1]+delta_x[i])
    ### Get matrix b
        b[i,0] = 3*(delta_y[i]/delta_x[i] - delta_y[i-1]/delta_x[i-1])

    ### Solves for c in Ac = b
    # print('Jacobi Method Output:')
    c = jacobi(A, b, np.zeros(len(A)), tol = tol, n_iterations=1000)

    ### Solves for d and b
    d = np.zeros(shape = (size-1,1))
    b = np.zeros(shape = (size-1,1))
    for i in range(0,len(d)):
        d[i] = (c[i+1] - c[i]) / (3*delta_x[i])
        b[i] = (delta_y[i]/delta_x[i]) - (delta_x[i]/3)*(2*c[i] + c[i+1])

    return b.squeeze(), c.squeeze(), d.squeeze()

def make_hermite_table(input_table):
    """ make_hermite_table(input_table)
    input a dataframe of with columns 'x', 'f(x)' and 'f_prime(x)'
    return a completed hermite table"""
    data = pd.DataFrame({'z_vals' : [np.NaN]*(2*input_table['x'].size),
                        'f_1' : [np.NaN]*(2*input_table['x'].size),
                        'f_2' : [np.NaN]*(2*input_table['x'].size)})
    for z in range(data['z_vals'].size):
        data.loc[z, 'z_vals'] = input_table['x'][int(z/2)]
        data.loc[z, 'f_1'] = input_table['f(x)'][int(z/2)]
        if z % 2 == 0:
            data.loc[z, 'f_2'] = input_table['f_prime(x)'][z/2]
    for z in range(1, data['z_vals'].size - 1, 2):
        data.loc[z, 'f_2'] = (data['f_1'][z+1] - data['f_1'][z])/(data['z_vals'][z+1] - data['z_vals'][z])
    for x in range(3, len(list(data.index)) + 1):
        col_name = 'f_' + str(x)
        data[col_name] = [np.NaN]*len(list(data.index))
    for x in range(3, len(list(data.columns))):
        cur_col = 'f_' + str(x)
        prev_col = 'f_' + str(x-1)
        for y in range(data[cur_col].size - x + 1):
            data.loc[y, cur_col] = (data[prev_col][y + 1] - data[prev_col][y])/(data['z_vals'][y + x - 1] - data['z_vals'][y])
    return data

def make_lagrange_table(input_table):
    """ make_lagrange_table(input_table)
        input a dataframe with columns 'x' and 'f(x)'
        returns a completed dataframe for lagrange polynomials
    """
    data = input_table.rename(columns = {'x' : 'x', 'f(x)' : 'f_1'})
    for x in range(2, len(list(data.index)) + 1):
        col_name = 'f_' + str(x)
        data[col_name] = [np.NaN]*len(list(data.index))
    for x in range(2, len(list(data.columns))):
        cur_col = 'f_' + str(x)
        prev_col = 'f_' + str(x-1)
        for y in range(data[cur_col].size - x + 1):
            data.loc[y, cur_col] = (data[prev_col][y + 1] - data[prev_col][y])/(data['x'][y + x - 1] - data['x'][y])
    return data

def endpoint_3(data, i, h=1):
    """endpoint_3(data, i, h=1)
        input a dataframe with columns 'x' and 'f(x)', an index in the
        dataframe, and an increment h. H is assumed to be 1, but can
        also be -1
        Returns the float value of the derivative at the index
    """
    try:
        a = -3*data['f(x)'][i]
        b = 4*data['f(x)'][i+h]
        c = -1*data['f(x)'][i+(2*h)]
        gap = data['x'][i+h] - data['x'][i]
        return (1/(2*gap))*(a+b+c)
    except:
        return np.NaN

def midpoint_3(data, i, h=1):
    """midpoint_3(data, i, h=1)
        input a dataframe with columns 'x' and 'f(x)', an index in the
        dataframe, and an increment h. H is assumed to be 1, but can
        also be -1
        Returns the float value of the derivative at the index
    """
    try:
        a = data['f(x)'][i+h]
        b = -1*data['f(x)'][i-h]
        gap = data['x'][i] - data['x'][i+h]
        return (1/(2*gap))*(a+b)
    except:
        return np.NaN

def endpoint_5(data, i, h=1):
    """endpoint_5(data, i, h=1)
        input a dataframe with columns 'x' and 'f(x)', an index in the
        dataframe, and an increment h. H is assumed to be 1, but can
        also be -1
        Returns the float value of the derivative at the index
    """
    try:
        a = -25*data['f(x)'][i]
        b = 48*data['f(x)'][i+h]
        c = -36*data['f(x)'][i+(2*h)]
        d = 16*data['f(x)'][i+(3*h)]
        e = -3*data['f(x)'][i+(4*h)]
        gap = data['x'][i+h] - data['x'][i]
        return (1/(12*gap))*(a+b+c+d+e)
    except:
        return np.NaN

def midpoint_5(data, i, h=1):
    """ midpoint_5(data, i, h=1)
        input a dataframe with columns 'x' and 'f(x)', an index in the
        dataframe, and an increment h. H is assumed to be 1, but can
        also be -1
        Returns the float value of the derivative at the index
    """
    try:
        a = 1*data['f(x)'][i-(2*h)]
        b = -8*data['f(x)'][i-(h)]
        c = 8*data['f(x)'][i+(h)]
        d = -1*data['f(x)'][i+(2*h)]
        gap = data['x'][i+h] - data['x'][i]
        return (1/(12*gap))*(a+b+c+d)
    except:
        return np.NaN

def prep_hermite_table(input_df):
    """ prep_hermite_table(input_df)
        input a dataframe with columns 'x' and 'f(x)'
        return a table that calculates 'f_prime(x) using
        midpoint and endpoint approximations'
    """
    hermite_df = input_df.copy()
    hermite_df['f_prime(x)'] = [np.NaN]*hermite_df.index.size
    hermite_df['method'] = [np.NaN]*hermite_df.index.size
    for n in hermite_df.index:
        if not pd.isna(midpoint_5(hermite_df, n, 1)) :
            hermite_df['f_prime(x)'][n] = midpoint_5(hermite_df, n, 1)
            hermite_df['method'][n] = 'midpoint5'
        elif not pd.isna(endpoint_5(hermite_df, n, 1)):
            hermite_df['f_prime(x)'][n] = endpoint_5(hermite_df, n, 1)
            hermite_df['method'][n] = 'forward endpoint 5'
        elif not pd.isna(endpoint_5(hermite_df, n, -1)):
            hermite_df['f_prime(x)'][n] = endpoint_5(hermite_df, n, -1)
            hermite_df['method'][n] = 'backward endpoint 5'
        elif not pd.isna(midpoint_3(hermite_df, n, 1)):
            hermite_df['f_prime(x)'][n] = midpoint_3(hermite_df, n, 1)
            hermite_df['method'][n] = 'midpoint3'
        elif not pd.isna(endpoint_3(hermite_df, n, 1)):
            hermite_df['f_prime(x)'][n] = endpoint_3(hermite_df, n, 1)
            hermite_df['method'][n] = 'forward endpoint 3'
        elif not pd.isna(endpoint_3(hermite_df, n, -1)):
            hermite_df['f_prime(x)'][n] = endpoint_3(hermite_df, n, -1)
            hermite_df['method'][n] = 'backward endpoint 3'
        else:
            hermite_df['f_prime(x)'][n] = np.NaN
    return hermite_df

def get_forward_lagrange_approximation(data, n, x):
    """get_forward_lagrange_approximation(data, n, x)
       Input a dataframe holding a divided difference table, a degree n,
       and a value x at which to calculate the approximation.
       The columns in the df are 'x' and 'f_n'
       It returns the value of the forward lagrange approximation
       starting at the 0th index, up to degree n
    """
    result = 0
    for i in range(1, n+2):
        cur_col = 'f_' + str(i)
        cur_mult = data[cur_col][0]
        for j in range(i-1):
            cur_mult *= x-data['x'][j]
        result += cur_mult
    return result

def get_backward_lagrange_approximation(data, n, x):
    """get_backward_lagrange_approximation(data, n, x)
       Input a dataframe holding a divided difference table, a degree n,
       and a value x at which to calculate the approximation.
       The columns in the df are 'x' and 'f_n'
       It returns the value of the backward lagrange approximation
       starting at the last index, up to degree n
    """
    result = 0
    cur_max = data['f_1'].size - 1
    for i in range(1, n+2):
        cur_col = 'f_' + str(i)
        cur_mult = data[cur_col][cur_max]
        for j in range(3, cur_max, -1):
            cur_mult *= x-data['x'][j]
        result += cur_mult
        cur_max -= 1
    return result

def get_forward_hermite_approximation(data, n, x):
    """Input a dataframe holding a hermite divided difference table, a degree n,
       and a value x at which to calculate the approximation.
       The columns in the df are 'z_vals' and 'f_n'
       It returns the value of the forward hermite approximation
       starting at the last index, up to degree n
    """
    result = 0
    for i in range(1, n+2):
        cur_col = 'f_' + str(i)
        cur_mult = data[cur_col][0]
        for j in range(i-1):
            cur_mult *= x-data['z_vals'][j]
        result += cur_mult
    return result

def get_backward_hermite_approximation(data, n, x):
    """get_backward_hermite_approximation(data, n, x)
       Input a dataframe holding a hermite divided difference table, a degree n,
       and a value x at which to calculate the approximation.
       The columns in the df are 'z_vals' and 'f_n'
       It returns the value of the backward hermite approximation
       starting at the last index, up to degree n
    """
    result = 0
    cur_max = data['f_1'].size - 1
    for i in range(1, n+2):
        cur_col = 'f_' + str(i)
        cur_mult = data[cur_col][cur_max]
        for j in range(3, cur_max, -1):
            cur_mult *= x-data['z_vals'][j]
        result += cur_mult
        cur_max -= 1
    return result

def make_spline_table(df):
    """make_spline_table(df)
    input a dataframe with with columns 'x' and 'f(x)'
    returns a dataframe with a, b, c, d at each index
    These constants are the coefficients of a spline function
    note that this is dubious. Use scipy when possible.
    """
    mat = cubic_spline(list(df['x']), list(df['f(x)']))
    spline_df = df.rename(columns = {'f(x)' : 'a'})
    spline_df['b'] = [np.NaN]* spline_df.index.size
    spline_df['c'] = [np.NaN]* spline_df.index.size
    spline_df['d'] = [np.NaN]* spline_df.index.size
    for n in range(spline_df.index.size - 1):
        spline_df['b'][n] = mat[0][n]
        spline_df['c'][n] = mat[1][n]
        spline_df['d'][n] = mat[2][n]
    return spline_df

def spline_function(df, i, x):
    """spline_function(df, i, x)
    Input the dataframe resulting from make_spline_table, the index to use to approximate,
    and the point x to approximate
    """
    z = df['x'][i]
    return df['a'][i] + (df['b'][i] * pow(x - z, 1)) + (df['c'][i] * pow(x - z, 2))  + (df['c'][i] * pow(x - z, 3))

def get_df(data, n):
    points = []
    size = data.index.size - 1
    step = size/(n-1)
    i = 0
    while i <= size:
        points.append(int(i))
        i += step
    cur_df = data[data.index.isin(points)]
    cur_df.reset_index(drop = True, inplace = True)
    return cur_df

def trapezoidal(f, a, b, n):
    """trapezoidal(f, a, b, n)
    uses the trapezoidal rule to approximate the function f from a to b using n intervals
    """
    h = (b-a)/n

    total = (f(a)+f(b))/2
    x = a + h

    for i in range(n-1):
        total += f(x)
        x += h

    return total * h

def simpsons(f, a, b, n):
    """simpsons(f, a, b, n)
    uses simpsons rule to approximate the function f from a to b using n intervals
    """
    if n % 2 == 1:
        n += 1
    h = (b-a)/n

    total = f(a) + f(b)
    x = a+h

    for i in range(n-1):
        coeff = 2 if (i & 1) else 4
        total += coeff * f(x)
        x += h

    return total*h/3

def midpoint(f, a, b, n):
    """midpoint(f, a, b, n)
    uses midpoint rule to approximate the function f from a to b using n intervals
    """
    h = (b-a)/(n+2)
    total = 0
    j = 0
    while j <= n/2:
        x = a + (((2*j)+1)*h)
        total += f(x)
        j += 1
    return 2*h*total

def itp(x_list,y_list, x):
    """itp(x_list,y_list, x)
    uses a cubic spline to approximate the value of a function at x.
    x_list and y_list are the known values in the function
    """
    tck = interpolate.splrep(x_list, y_list)
    return interpolate.splev(x, tck)
