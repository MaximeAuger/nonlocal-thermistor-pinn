"""
Neural network ansatz shared by the PINN and the VPINN.

    u_theta(x) = x (1 - x) * N_theta(2x - 1)

N_theta is a tanh MLP with Xavier initialisation (same design as the
steady-state-heat-conduction benchmark).  The factor x(1-x) imposes the
homogeneous Dirichlet conditions u(0) = u(1) = 0 exactly ("hard" BCs), so
both losses contain a single term and no penalty weight has to be tuned.

References
----------
    - Raissi, M., Perdikaris, P. & Karniadakis, G.E. (2019). Physics-informed
      neural networks: A deep learning framework for solving forward and
      inverse problems involving nonlinear partial differential equations.
      J. Comput. Phys., 378, 686-707.
    - Lagaris, I.E., Likas, A. & Fotiadis, D.I. (1998). Artificial neural
      networks for solving ordinary and partial differential equations.
      IEEE Trans. Neural Netw., 9(5), 987-1000.   (hard-constraint ansatz)

Author: Maxime Auger, Dept. of Applied Mechanics, FEMTO-ST Institute, ENSMM
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """x (R^1) -> [Linear -> Tanh] x n_layers -> Linear -> R^1."""

    def __init__(self, n_input=1, n_output=1, n_hidden=32, n_layers=4):
        super().__init__()
        layers = [nn.Linear(n_input, n_hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(n_hidden, n_hidden), nn.Tanh()]
        layers.append(nn.Linear(n_hidden, n_output))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class HardDirichletNet(nn.Module):
    """u_theta(x) = x (1 - x) N_theta(2x - 1)  ->  u(0) = u(1) = 0 exactly."""

    def __init__(self, n_hidden=32, n_layers=4):
        super().__init__()
        self.mlp = MLP(1, 1, n_hidden, n_layers)

    def forward(self, x):
        return x * (1.0 - x) * self.mlp(2.0 * x - 1.0)
