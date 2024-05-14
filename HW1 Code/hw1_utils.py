import torch
import numpy as np
import matplotlib.pyplot as plt


def load_data():
    N = 10
    std = 0.5
    torch.manual_seed(1)
    x = torch.cat(
        (
            std * torch.randn(2, N) + torch.Tensor([[2], [-2]]),
            std * torch.randn(2, N) + torch.Tensor([[-2], [2]]),
        ),
        1,
    )
    init_c = torch.Tensor([[2, -2], [2, -2]])
    return x, init_c


def vis_cluster(c1, x1, c2, x2):
    # c1, c2: [2, 1]
    # x1, x2: [2, #cluster_points]
    assert c1.ndim == 2, "please keep centroid with dimension [2, 1]"

    c = torch.cat((c1, c2), dim=1)

    plt.plot(x1[0, :].numpy(), x1[1, :].numpy(), "ro")
    plt.plot(x2[0, :].numpy(), x2[1, :].numpy(), "bo")
    l = plt.plot(c[0, :].numpy(), c[1, :].numpy(), "kx")
    plt.setp(l, markersize=10)
    plt.show()
