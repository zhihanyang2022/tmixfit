import numpy as np
from scipy.stats import multivariate_t as mt
from scipy.spatial.distance import mahalanobis

import torch
torch.set_default_dtype(torch.float64)

from .stmm_abstract import STMMAbstract


class STMMLoop(STMMAbstract):

    """
    Loop implementation of STMM and EM in Numpy.
    Some part of this is definitely vectorize-able, but I'll keep it this way.
    """

    def __init__(
        self,
        p: int,
        g: int,
        v: int = 2,
        pi_init: np.array = None,
        mus_init: np.array = None,
        Sigmas_init: np.array = None
    ):

        self.p = p
        self.g = g
        self.v = v

        self.tau_matrix, self.u_matrix = None, None

        self.pi = pi_init if pi_init is not None else np.ones((self.g, )) / self.g
        self.mus = mus_init if mus_init is not None else (np.random.uniform(size=(self.g, self.p)) - 0.5) * 2
        self.Sigmas = Sigmas_init if Sigmas_init is not None else np.tile(np.expand_dims(np.eye(self.p), axis=0), reps=(self.g, 1, 1))

    def loglik(self, data: np.array) -> float:

        n = data.shape[0]

        sum_ = 0

        for j in range(n):
            sum_ += np.log(
                np.sum(
                    [self.pi[ip] * np.exp(mt(self.mus[ip], self.Sigmas[ip], df=self.v).logpdf(data[j]))for ip in range(self.g)]
                )
            )

        return sum_

    def fit_one_iter(self, data: np.array) -> None:

        n = data.shape[0]

        # XXXXXXXXXXXXXXXXXXXX E step XXXXXXXXXXXXXXXXXXXX

        # (1) Estimating the tau matrix of shape (g, n)

        self.tau_matrix = np.zeros((self.g, n))

        for i in range(self.g):
            for j in range(n):
                p_zij_equals_1 = self.pi[i]
                p_yj_given_zij_equals_1 = np.exp(mt(self.mus[i], self.Sigmas[i], df=self.v).logpdf(data[j]))
                p_yj = np.sum(
                    [self.pi[ip] * np.exp(mt(self.mus[ip], self.Sigmas[ip], df=self.v).logpdf(data[j])) for ip in range(self.g)]
                )
                p_zij_equals_1_given_yj = p_yj_given_zij_equals_1 * p_zij_equals_1 / p_yj
                self.tau_matrix[i][j] = p_zij_equals_1_given_yj

        # (2) Estimating the u matrix of shape (g, n)

        self.u_matrix = np.zeros((self.g, n))
        for i in range(self.g):
            for j in range(n):
                mahalanobis_distance = mahalanobis(
                    data[j], self.mus[i], np.linalg.inv(self.Sigmas[i])
                ) ** 2  # I'm passing in the inverse and squaring the result because Scipy's implementation is weird...
                self.u_matrix[i][j] = (self.v + self.p) / (self.v + mahalanobis_distance)

        # XXXXXXXXXXXXXXXXXXXX M step XXXXXXXXXXXXXXXXXXXX

        # (1) Getting updated pi
        # Equation 29 in paper

        for i in range(self.g):
            self.pi[i] = np.sum(self.tau_matrix[i]) / n

        # (2) Getting updated mus
        # Equation 30 in paper

        for i in range(self.g):
            numerator = np.sum([self.tau_matrix[i][j] * self.u_matrix[i][j] * data[j] for j in range(n)], axis=0)  # a vector
            denominator = np.sum([self.tau_matrix[i][j] * self.u_matrix[i][j] for j in range(n)])  # a scalar
            self.mus[i] =  numerator / denominator  # a vector

        # (3) Getting updated Sigmas
        # Equation 31 in paper + small modification for faster convergence at the top of page 343

        for i in range(self.g):
            numerator = np.sum(
                [
                    self.tau_matrix[i][j] * self.u_matrix[i][j] * (data[j] - self.mus[i]).reshape(self.p, 1) @ (data[j] - self.mus[i]).reshape(self.p, 1).T
                for j in range(n)],
                axis=0
            )  # a matrix
            denominator = np.sum([self.tau_matrix[i][j] * self.u_matrix[i][j] for j in range(n)])  # a scalar
            self.Sigmas[i] = numerator / denominator

    def pdf(self, data: np.array) -> np.array:

        n = data.shape[0]

        densities = []
        for j in range(n):
            densities.append(
                np.sum([
                    self.pi[ip] * mt(self.mus[ip], self.Sigmas[ip], df=self.v).pdf(data[j]) for ip in range(self.g)
                ])
            )

        return np.array(densities)
