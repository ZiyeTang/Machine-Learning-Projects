import torch
import numpy as np
import matplotlib.pyplot as plt

''' Start SVM helpers '''
def svm_contour(pred_fxn, xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33):
    '''
    Produces a contour plot for the prediction function.

    Arguments:
        pred_fxn: Prediction function that takes an n x d tensor of test examples
        and returns your SVM's predictions.
        xmin: Minimum x-value to plot.
        xmax: Maximum x-value to plot.
        ymin: Minimum y-value to plot.
        ymax: Maximum y-value to plot.
        ngrid: Number of points to be plotted between max and min (granularity).
    '''
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)),
            dim = 2).view(-1, 2)
        zz = pred_fxn(x_test)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                         cmap = 'coolwarm')
        plt.clabel(cs)
        plt.show()

def poly_implementation(x, y, degree):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (1 + (x * y).sum()).pow(degree)

def poly(degree):
    return lambda x, y: poly_implementation(x, y, degree)

def rbf_implementation(x, y, sigma):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (-(x - y).norm().pow(2) / 2 / sigma / sigma).exp()

def rbf(sigma):
    return lambda x, y: rbf_implementation(x, y, sigma)

def xor_data():
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y

''' End SVM Helpers '''

''' Start Logistic Helpers '''
def load_logistic_data():
    torch.manual_seed(0) # reset seed
    return linear_problem(torch.tensor([-1., 2.]), margin=1.5, size=200)

def linear_problem(w, margin, size, bounds=[-5., 5.], trans=0.0):
    in_margin = lambda x: torch.abs(w.flatten().dot(x.flatten())) / torch.norm(w) \
                          < margin
    half_margin = lambda x: 0.6*margin < w.flatten().dot(x.flatten()) / torch.norm(w) < 0.65*margin
    X = []
    Y = []
    for i in range(size):
        x = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        while in_margin(x):
            x.uniform_(bounds[0], bounds[1]) + trans
        if w.flatten().dot(x.flatten()) + trans > 0:
            Y.append(torch.tensor(1.))
        else:
            Y.append(torch.tensor(-1.))
        X.append(x)
    for j in range(1):
        x_out = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        while not half_margin(x_out):
            x_out = torch.zeros(2).uniform_(bounds[0], bounds[1]) + trans
        X.append(x_out)
        Y.append(torch.tensor(-1.))
    X = torch.stack(X)
    Y = torch.stack(Y).reshape(-1, 1)

    return X, Y
''' End Logistic Helpers '''
