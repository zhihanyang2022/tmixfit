from abc import ABC, abstractmethod
import numpy as np
import torch


class STMMAbstract(ABC):

    @abstractmethod
    def loglik(self, data):
        pass

    @abstractmethod
    def fit_one_iter(self, data, debug):
        pass

    def fit(self, data: torch.tensor, num_iters: int) -> float:
        logliks = []
        prev_loglik = - np.inf
        for _ in range(num_iters):
            loglik = self.fit_one_iter(data)
            assert loglik >= prev_loglik or np.allclose(loglik, prev_loglik), "EM should be monotonically improving the log-likelihood"
            prev_loglik = loglik
            logliks.append(loglik)
        return logliks

    @abstractmethod
    def pdf(self, data):
        pass
