import numpy as np

from ...EA import Individual
from . import AbstractMutation
import time

class PolynomialMutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, nm = 15, pm = None, *arg, **kwargs):
        '''
        nm: parameters of PolynomialMutation
        pm: prob mutate of PolynomialMutation
        '''
        super().__init__(*arg, **kwargs)
        self.nm = nm
        self.pm = pm
        self.count_time = 0

    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        s_time = time.time()
        idx_mutation = np.where(np.random.rand(self.dim_uss) <= self.pm)[0]

        u = np.zeros((self.dim_uss,)) + 0.5
        u[idx_mutation] = np.random.rand(len(idx_mutation))

        delta = np.where(u < 0.5,
            # delta_l
            (2*u)**(1/(self.nm + 1)) - 1,
            # delta_r
            1 - (2*(1-u))**(1/(self.nm + 1))
        )

        if return_newInd:
            newInd = self.IndClass(
                genes= np.where(delta < 0,
                    # delta_l: ind -> 0
                    ind.genes + delta * ind.genes,
                    # delta_r: ind -> 1
                    ind.genes + delta * (1 - ind.genes)
                ),
                parent= ind
            )
            newInd.skill_factor = ind.skill_factor
            self.count_time += (time.time() - s_time)
            return newInd
        else:
            ind.genes = np.where(delta < 0,
                    # delta_l: ind -> 0
                    ind.genes + delta * ind.genes,
                    # delta_r: ind -> 1
                    ind.genes + delta * (1 - ind.genes)
            )
            ind.fcost = None

            self.count_time += (time.time() - s_time)
            return ind
    