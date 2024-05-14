import torch
import hw1_utils
import numpy as np
import torch


def k_means(X=None, init_c=None, n_iters=50):
    """K-Means.

    Argument:
        X: 2D data points, shape [2, N].
        init_c: initial centroids, shape [2, 2]. Each column is a centroid.
    
    Return:
        c: shape [2, 2]. Each column is a centroid.
    """

    if X is None:
        X, init_c = hw1_utils.load_data()
    
    cur_c = init_c
    N = X.shape[1]

    r = torch.zeros((N,2))
    change = False
    for i in range(N):
        dif0 = X[:,i] - cur_c[:,0]
        dif1 = X[:,i] - cur_c[:,1]
        if dif0@dif0 < dif1@dif1:
            if r[i][0] == 0:
                change = True
                r[i][0] = 1
                r[i][1] = 0
        else:
            if r[i][1] == 0:
                change = True
                r[i][0] = 0
                r[i][1] = 1

    for i in range(n_iters):
        x1 = []
        x2 = []
        for i in range(N):
            if r[i,0]:
                x1.append(X[:,i].tolist())
            else:
                x2.append(X[:,i].tolist())

        x1 = torch.tensor(x1).T
        x2 = torch.tensor(x2).T
        cur_c[:,0] = torch.sum(x1, axis=1) / torch.sum(r[:,0])
        cur_c[:,1] = torch.sum(x2, axis=1) / torch.sum(r[:,1])

        hw1_utils.vis_cluster(cur_c[:,0].reshape(2,1), x1, cur_c[:,1].reshape(2,1), x2)
        print("center:", cur_c)
        print("cost:", 0.5*torch.sum((x1-cur_c[:,0].reshape(2,1))**2) + 0.5*torch.sum((x2-cur_c[:,1].reshape(2,1))**2),"\n\n")

        change = False
        for i in range(N):
            dif0 = X[:,i] - cur_c[:,0]
            dif1 = X[:,i] - cur_c[:,1]
            if dif0@dif0 < dif1@dif1:
                if r[i][0] == 0:
                    change = True
                    r[i][0] = 1
                    r[i][1] = 0
            else:
                if r[i][1] == 0:
                    change = True
                    r[i][0] = 0
                    r[i][1] = 1
        if not change:
            break
        
    return cur_c

