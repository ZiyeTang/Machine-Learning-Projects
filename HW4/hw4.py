import torch
import hw4_utils as utils

'''
    Important
    ========================================
    The autograder evaluates your code using FloatTensors for all computations.
    If you use DoubleTensors, your results will not match those of the autograder
    due to the higher precision.

    PyTorch constructs FloatTensors by default, so simply don't explicitly
    convert your tensors to DoubleTensors or change the default tensor.

    Be sure to modify your input matrix X in exactly the way specified. That is,
    make sure to prepend the column of ones to X and not put the column anywhere
    else, and make sure your feature-expanded matrix in Problem 3 is in the
    specified order (otherwise, your w will be ordered differently than the
    reference solution's in the autograder).
'''

# Problem 2
def linear_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        num_iter (int): iterations of gradient descent to perform

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n = X.shape[0]
    d = X.shape[1]
    X_new = torch.cat([torch.ones(n,1), X], dim = 1)
    w = torch.zeros(d+1, 1)
    for i in range(num_iter):
        w -= lrate / n * X_new.T @ (X_new @ w - Y)
    
    return w
        

def linear_normal(X, Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    n = X.shape[0]
    X_new = torch.cat([torch.ones(n,1), X], dim = 1)

    X_plus = torch.pinverse(X_new)

    return X_plus @ Y

def plot_linear():
    '''
        Returns:
            Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    n = X.shape[0]
    w = linear_normal(X, Y)
    X_new = torch.cat([torch.ones(n,1), X], dim = 1)

    utils.plt.plot(X, Y, '.')
    utils.plt.plot(X, X_new @ w)
    utils.plt.xlabel('Y')
    utils.plt.ylabel('X')

# Problem 3
def poly_gd(X, Y, lrate=0.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels
        lrate (float): the learning rate
        num_iter (int): number of iterations of gradient descent to perform

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''

    n = X.shape[0]
    d = X.shape[1]
    X = torch.cat([torch.ones(n,1), X], dim = 1)
    for i in range(1,d+1):
        for j in range(i, d+1):
            X = torch.cat([X, (X[:,i]*X[:,j]).reshape(n,1)], dim = 1)
    
    w = torch.zeros(int(1 + d + d * (d + 1) / 2), 1)
    for i in range(num_iter):
        w -= lrate / n * X.T @ (X @ w - Y)
    
    return w

def poly_normal(X,Y):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (1 + d + d * (d + 1) / 2) x 1 FloatTensor: the parameters w
    '''
    n = X.shape[0]
    d = X.shape[1]
    X = torch.cat([torch.ones(n,1), X], dim = 1)
    for i in range(1,d+1):
        for j in range(i, d+1):
            X = torch.cat([X, (X[:,i]*X[:,j]).reshape(n,1)], dim = 1)
    
    X_plus = torch.pinverse(X)

    return X_plus @ Y

def plot_poly():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_reg_data()
    n = X.shape[0]
    d = X.shape[1]

    w = poly_normal(X, Y)
    X_new = torch.cat([torch.ones(n,1), X], dim = 1)
    for i in range(1,d+1):
        for j in range(i, d+1):
            X_new = torch.cat([X_new, (X_new[:,i]*X_new[:,j]).reshape(n,1)], dim = 1)

    utils.plt.plot(X, Y, '.')
    utils.plt.plot(X, X_new @ w)
    utils.plt.xlabel('Y')
    utils.plt.ylabel('X')




def poly_xor():
    '''
    Returns:
        n x 1 FloatTensor: the linear model's predictions on the XOR dataset
        n x 1 FloatTensor: the polynomial model's predictions on the XOR dataset
    '''
    X, Y = utils.load_xor_data()
    n = X.shape[0]
    d = X.shape[1]


    w1 = linear_normal(X,Y)
    w2 = poly_normal(X, Y)
    
    def linear_predict(X):
        n = X.shape[0]
        X_new = torch.cat([torch.ones(n,1), X], dim = 1)

        return X_new @ w1

    def poly_predict(X):
        n = X.shape[0]
        d = X.shape[1]
        
        X_new = torch.cat([torch.ones(n,1), X], dim = 1)
        for i in range(1,d+1):
            for j in range(i, d+1):
                X_new = torch.cat([X_new, (X_new[:,i]*X_new[:,j]).reshape(n,1)], dim = 1)

        return X_new @ w2

    utils.contour_plot(-1, 1, -1, 1, linear_predict)
    utils.contour_plot(-1, 1, -1, 1, poly_predict)

    X_new = torch.cat([torch.ones(n,1), X], dim = 1)
    for i in range(1,d+1):
        for j in range(i, d+1):
            X_new = torch.cat([X_new, (X_new[:,i]*X_new[:,j]).reshape(n,1)], dim = 1)
    
    return X_new[:,0:d+1] @ w1, X_new @ w2


