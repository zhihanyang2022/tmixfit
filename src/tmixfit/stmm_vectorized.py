import numpy as np
import torch
import torch.distributions as dist
import pyro.distributions as dist2

from .stmm_abstract import STMMAbstract
from .utils import batch_mahalanobis, Param

torch.set_default_dtype(torch.float64)  # helps with precision


class STMMVectorized(STMMAbstract):

    """Vectorized implementation of STMM and EM in PyTorch."""

    def __init__(
        self,
        p: int,
        g: int,
        v: int = 2,
        pi_init: torch.tensor = None,
        mus_init: torch.tensor = None,
        Sigmas_init: torch.tensor = None
    ):

        self.p = p  # data dimension
        self.g = g  # number of components to fit
        self.v = v  # degree of freedom

        assert self.v <= 10000, "For safety, don't set dof too large."

        self.pi = pi_init if pi_init else torch.ones(self.g) / self.g
        self.mus = mus_init if mus_init else (torch.rand(self.g, self.p) - 0.5) * 2
        self.Sigmas = Sigmas_init if Sigmas_init else torch.eye(self.p).unsqueeze(0).repeat(self.g, 1, 1)
        self.scale_trils = torch.linalg.cholesky(self.Sigmas)

        assert self.pi.size() == (self.g, )
        assert self.mus.size() == (self.g, self.p)
        assert self.Sigmas.size() == (self.g, self.p, self.p)

        self.mix, self.comp, self.stmm = None, None, None

        self.update_distributions()

    def update_distributions(self) -> None:

        self.mix = dist.Categorical(self.pi)
        self.comp = dist2.MultivariateStudentT(
            loc=self.mus,
            scale_tril=self.scale_trils,  # lower triangular cholesky decomposition of cov matrix
            df=self.v
        )
        self.stmm = dist.MixtureSameFamily(self.mix, self.comp)  # Student-t mixture model

    def loglik(self, data: torch.tensor) -> float:
        
        return float(torch.sum(
            self.stmm.log_prob(data)  # this returns one log prob per data point
        ))

    def fit_one_iter(self, data: torch.tensor, debug: bool = False) -> tuple:
        """Execute one E step and one M step using data."""
        
        n = data.shape[0]

        # XXXXXXXXXXXXXXXXXXXX E step XXXXXXXXXXXXXXXXXXXX

        # (1) Estimating the tau matrix of shape (g, n)
        # (The following lines of code simply do vectorized Bayes rule.)

        p_zij_equals_1 = self.pi.reshape(self.g, 1)  # (g, 1); first dim will be broadcasted from 1 to n
        p_yj_given_zij_equals_1 = torch.exp(self.comp.log_prob(data.unsqueeze(1))).T  # (g, n); more on this in my blog

        p_yj_and_yzij_equals_1 = p_zij_equals_1 * p_yj_given_zij_equals_1  # (g, n); the joint

        p_yj = p_yj_and_yzij_equals_1.sum(dim=0).reshape(1, n)  # (1, n); summed over the component dim (indexed by i),
        # zeroth dim will be broadcasted from 1 to g

        tau_matrix = p_yj_and_yzij_equals_1 / p_yj  # (g, n); or p_zij_equals_1_given_yj

        # (2) Estimating the u matrix of shape (g, n)

        # data has shape (1, n, p)
        # mus has shape (g, 1, p)

        demeaned_data = data.view(1, n, self.p) - self.mus.view(self.g, 1, self.p)  # (g, n, p); relied on broadcasting
        mahalanobis_distances = batch_mahalanobis(
            bL=self.scale_trils.view(self.g, 1, self.p, self.p),
            bx=demeaned_data
        )  # (g, n); for each of the g covariance matrices, evaluated mahalanobis distance for n data vectors
        u_matrix = (self.v + self.p) / (self.v + mahalanobis_distances)

        # XXXXXXXXXXXXXXXXXXXX M step XXXXXXXXXXXXXXXXXXXX

        # (1) Getting updated pi of shape (g, )

        self.pis = tau_matrix.sum(dim=1) / n  # sum over the first dim of shape (g, n), and get (g, )

        # (2) Getting updated mus of shape (g, p)

        # weight = tau_matrix * u_matrix and has shape (g, n).
        # We can think of this matrix as the weights assigned to each of the n data vectors by each of the g components.

        # data has shape (n, p).

        # weight @ data has shape (g, p).
        # It contains weighted sums of all data vectors for each of the g components.

        # weight.sum(dim=1) has shape (g, ).
        # It sums over all weights for each of the g components.

        # (weights @ data) / weights.sum(dim=1).view(self.g, 1) has shape (g, p).
        # For the weighted sum of all data vectors associated with a component, this operation divides it by
        # the sum of all weights - this acts like an averaging operation and creates g prototypical data vectors.

        weights = tau_matrix * u_matrix
        self.mus = (weights @ data) / weights.sum(dim=1).view(self.g, 1)

        # (2) Getting updated Sigmas of shape (g, p, p)

        # This was the most unintuitive operation to vectorize, here were my steps.

        # First, for Eq 31's numerator, if there weren't \tau_{ij} and u_{ij}, then it would be calculating n * the
        # empirical covariance matrix. Computing this thing is something that I know how to vectorized, as I've done it
        # a few times before:

        # (Y with shape n, p - \mu_j with shape p) ^ T @  (Y with shape n, p - \mu_j with shape p)
        # Hint: By the rules of broadcasting, \mu_j gets turned into (1, p) and then (n, p).

        # Therefore, calculating demeaned_data_new and using torch.bmm is simply extending this operation to g
        # components. In other words, without considering \tau_{ij} and u_{ij}, the numerator would be:

        # demeaned_data_new = data.view(1, n, self.p) - self.mus.view(self.g, 1, self.p)  # (g, n, p)
        # self.Sigmas = torch.bmm(
        #             demeaned_data_new.transpose(2, 1),  # (g, p, n)
        #             demeaned_data_new  # (g, n, p)
        #         ) / (tau_matrix * u_matrix).sum(dim=1).view(self.g, 1, 1)

        # Now, we need to include \tau_{ij} and u_{ij} in the calculations of numerator. The crucial step is to think
        # about them as a part of either (Y - \mu_j)^T or (Y - \mu_j)

        # Since tau_matrix * u_matrix has shape (g, n), it was easier for me to just pick Y since it has shape (g, n, p)
        # and hence broadcasting can be done easily by appending a dimension to tau_matrix * u_matrix.

        demeaned_data_new = data.view(1, n, self.p) - self.mus.view(self.g, 1, self.p)  # (g, n, p)
        self.Sigmas = torch.bmm(
            demeaned_data_new.transpose(2, 1),  # (g, p, n)
            demeaned_data_new * tau_matrix.view(self.g, n, 1) * u_matrix.view(self.g, n, 1)  # (g, n, p)
        ) / (tau_matrix * u_matrix).sum(dim=1).view(self.g, 1, 1)
        self.scale_trils = torch.linalg.cholesky(self.Sigmas)

        self.update_distributions()

        if debug:
            return self.loglik(data), Param(self.pi, self.mus, self.Sigmas)
        else:
            return self.loglik(data)

    def fit(self, data: torch.tensor, num_iters: int) -> float:
        logliks = []
        prev_loglik = - np.inf
        for _ in range(num_iters):
            loglik = self.fit_one_iter(data)
            assert loglik >= prev_loglik or np.allclose(loglik, prev_loglik), "EM should be monotonically improving the log-likelihood"
            prev_loglik = loglik
            logliks.append(loglik)
        return logliks

    def pdf(self, data: torch.tensor) -> torch.tensor:
        n = data.shape[0]
        return torch.exp(self.stmm.log_prob(data.reshape(n, 1, -1)))
