import numpy as np
from typing import Tuple

from ...EA import Individual
from . import AbstractCrossover

class SBX_Crossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 15, *args, **kwargs):
        self.nc = nc

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
        u = np.random.rand(self.dim_uss)

        # ~1 TODO
        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (1 + self.nc)))

        #like pa
        oa = self.IndClass(np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1), parent = pa)
        #like pb
        ob = self.IndClass(np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1), parent = pb)

        if pa.skill_factor == pb.skill_factor:
            idx_swap = np.where(np.random.rand(self.dim_uss) < 0.5)[0]
            oa.genes[idx_swap], ob.genes[idx_swap] = ob.genes[idx_swap], oa.genes[idx_swap]

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob