import hw5_utils as utils
import numpy as np
import torch
import matplotlib.pyplot as plt

def svm_solver(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    '''
    Computes an SVM given a training set, training labels, the number of
    iterations to perform projected gradient descent, a kernel, and a trade-off
    parameter for soft-margin SVM.

    Arguments:
        x_train: 2d tensor with shape (n, d).
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        lr: The learning rate.
        num_iters: The number of gradient descent steps.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.
        c: The trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Returns:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step.
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    '''
    def grad_f(alp):
        res = torch.zeros_like(alp)
        l = alp.shape[0]
        for i in range(l):
            for j in range(l):
                res[i] += alp[j] * y_train[i] * y_train[j] * kernel(x_train[i], x_train[j])
            res[i] -= 1.0
        return res 
    
    alpha = torch.zeros_like(y_train)
    for i in range(num_iters):
        exp = alpha - lr * grad_f(alpha)
        if c == None:
            alpha = torch.clamp(exp, min=0)
        else:
            alpha = torch.clamp(exp, 0, c)
        
    return alpha.detach()


def svm_solver2(x_train, y_train, lr, num_iters,
               kernel=utils.poly(degree=1), c=None):
    
    def f(alp):
        res = 0.0
        l = alp.shape[0]
        for i in range(l):
            for j in range(l):
                res += alp[i] * alp[j] * y_train[i] * y_train[j] * kernel(x_train[i], x_train[j])
        return 0.5 * res - torch.sum(alp)    
    

    alpha = torch.zeros_like(y_train, requires_grad=True)
    for i in range(num_iters):
        torch.autograd.backward(f(alpha))
        
        exp = alpha - lr * alpha.grad
        alpha.grad.zero_()
        with torch.no_grad():
            if c == None:
                alpha = torch.clamp(exp, min=0)
            else:
                alpha = torch.clamp(exp, 0, c)
        alpha.requires_grad_(True)
        
    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=utils.poly(degree=1)):
    '''
    Returns the kernel SVM's predictions for x_test using the SVM trained on
    x_train, y_train with computed dual variables alpha.

    Arguments:
        alpha: 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: 2d tensor with shape (n, d), denoting the training set.
        y_train: 1d tensor with shape (n,), whose elements are +1 or -1.
        x_test: 2d tensor with shape (m, d), denoting the test set.
        kernel: The kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    '''
    m = x_test.shape[0]
    n = x_train.shape[0]
    wx = torch.zeros(m)

    for i in range(m):
        for j in range(n):
            wx[i] += alpha[j] * y_train[j] * kernel(x_test[i], x_train[j])

    return wx

def logistic(X, Y, lrate=.01, num_iter=1000):
    '''
    Arguments:
        X (n x d FloatTensor): the feature matrix
        Y (n x 1 FloatTensor): the labels

    Returns:
        (d + 1) x 1 FloatTensor: the parameters w
    '''
    d = X.shape[1]
    n = X.shape[0]
    w = torch.zeros(d+1,1)
    X_new = torch.cat([torch.ones(n,1), X], dim = 1)
    
    for i in range(num_iter):
        # R = 0
        # for j in range(n):
        #     R += Y[j] * X_new[j] * (1 - 1 / (1 + torch.exp(-Y[j] * X_new[j] @ w)))
        R = torch.sum(Y.view(-1, 1) * X_new * (1 - 1 / (1 + torch.exp(-Y.view(-1, 1) * X_new @ w))), dim=0)
        
        w += lrate * R.reshape(d+1,1) / n
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


def logistic_vs_ols():
    '''
    Returns:
        Figure: the figure plotted with matplotlib
    '''
    X, Y = utils.load_logistic_data()
    wl = logistic(X,Y,num_iter=100000000)
    wo = linear_normal(X,Y)
    
    x1,idx = torch.sort(X[:,0])
    x2 = X[:,1][idx]

    utils.plt.plot(x1,x2, '.')
    utils.plt.plot(x1, (-wo[1]*x1-wo[0])/wo[2], label = "OLS")
    utils.plt.plot(x1, (-wl[1]*x1-wl[0])/wl[2], label = "logistic")
    utils.plt.xlabel('X')
    utils.plt.ylabel('Y')
    plt.title('Logistic v.s. OLS')
    plt.legend()

    
