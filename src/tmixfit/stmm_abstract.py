from abc import ABC, abstractmethod
from typing import Union, NamedTuple, Tuple, Any
import numpy as np
import torch
from tqdm import tqdm


class STMMAbstract(ABC):

    @abstractmethod
    def loglik(self, data: Union[np.array, torch.tensor]) -> float:
        """Compute the log-likelihood of the data under the current parameters. Use for debugging."""
        pass

    @abstractmethod
    def fit_one_iter(self, data: Union[np.array, torch.tensor]) -> None:
        pass

    def fit(self, data: Union[np.array, torch.tensor], num_iters: int) -> list:
        logliks = []
        prev_loglik = - np.inf
        for _ in tqdm(range(num_iters)):
            self.fit_one_iter(data)
            loglik = self.loglik(data)
            assert loglik >= prev_loglik or \
                   np.allclose(loglik, prev_loglik), "EM should be monotonically improving the log-likelihood"
            prev_loglik = loglik
            logliks.append(loglik)
        return logliks

    @abstractmethod
    def pdf(self, data: Union[np.array, torch.tensor]) -> Union[np.array, torch.tensor]:
        """Compute the densities for each data vector under the current parmaeters. Use for plotting."""
        pass
