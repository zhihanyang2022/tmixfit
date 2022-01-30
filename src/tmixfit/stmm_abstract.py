from abc import ABC, abstractmethod


class STMMAbstract(ABC):

    @abstractmethod
    def loglik(self, data):
        pass

    @abstractmethod
    def fit_one_iter(self, data, debug):
        pass

    @abstractmethod
    def fit(self, data, num_iters):
        pass

    @abstractmethod
    def pdf(self, data):
        pass
