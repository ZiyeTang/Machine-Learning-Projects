import numpy as np
import torch
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_reg_data():
    # load the regression synthetic data
    torch.manual_seed(0) # force seed so same data is generated every time

    X = torch.linspace(0, 4, 100).reshape(-1, 1)
    noise = torch.normal(0, .4, size=X.shape)
    w = 0.5
    b = 1.
    Y = w * X**2 + b + noise

    return X, Y

def load_xor_data():
    X = torch.tensor([[-1,1],[1,-1],[-1,-1],[1,1]]).float()
    Y = torch.prod(X,axis=1)

    return X, Y

def load_logistic_data():
    torch.manual_seed(0) # reset seed
    return linear_problem(torch.tensor([-1., 2.]), margin=1.5, size=200)

def contour_plot(xmin, xmax, ymin, ymax, pred_fxn, ngrid = 33):
    """
    make a contour plot
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param pred_fxn: prediction function that takes an (n x d) tensor as input
                     and returns an (n x 1) tensor of predictions as output
    @param ngrid: number of points to use in contour plot per axis
    """
    # Build grid
    xgrid = torch.linspace(xmin, xmax, ngrid)
    ygrid = torch.linspace(ymin, ymax, ngrid)
    (xx, yy) = torch.meshgrid(xgrid, ygrid)

    # Get predictions
    features = torch.dstack((xx, yy)).reshape(-1, 2)
    predictions = pred_fxn(features)

    # Arrange predictions into grid and plot
    zz = predictions.reshape(xx.shape)
    C = plt.contour(xx, yy, zz,
                    cmap = 'coolwarm')
    plt.clabel(C)
    plt.show()

    return plt.gcf()

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
