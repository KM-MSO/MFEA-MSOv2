import numpy as np
from typing import Tuple, Type, List

from ...tasks.task import AbstractTask
from ...EA import Individual, Population
from . import AbstractCrossover

class KL_SBXCrossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 2, k = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nc = nc
        self.k = k
    
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        # self.prob = 1 - KL_divergence
        self.prob = [[np.ones((self.dim_uss, )) for i in range(self.nb_tasks)] for j in range(self.nb_tasks)]

    def update(self, population: Population, **kwargs) -> None:
        mean: list = np.empty((self.nb_tasks, )).tolist()
        std: list = np.empty((self.nb_tasks, )).tolist()
        for idx_subPop in range(self.nb_tasks):
            mean[idx_subPop] = np.mean(population[idx_subPop].ls_inds, axis = 0)
            std[idx_subPop] = np.std(population[idx_subPop].ls_inds, axis = 0)

        for i in range(self.nb_tasks):
            for j in range(self.nb_tasks):
                kl = np.log((std[i] + 1e-50)/(std[j] + 1e-50)) + (std[j] ** 2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                self.prob[i][j] = 1/(1 + kl/self.k)

        self.prob = np.clip(self.prob, 1/self.dim_uss, 1).tolist()

    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        if skf_oa == pa.skill_factor:
            p_of_oa = pa
        elif skf_oa == pb.skill_factor:
            p_of_oa = pb
        else:
            raise ValueError()
        if skf_ob == pb.skill_factor:
            p_of_ob = pb
        elif skf_ob == pa.skill_factor:
            p_of_ob = pa
        else:
            raise ValueError()
        
        u = np.random.rand(self.dim_uss)

        beta = np.where(u < 0.5, (2*u)**(1/(self.nc +1)), (2 * (1 - u))**(-1 / (self.nc + 1)))

        idx_crossover = np.random.rand(self.dim_uss) < self.prob[pa.skill_factor][pb.skill_factor]

        if np.all(idx_crossover == 0) or np.all(pa[idx_crossover] == pb[idx_crossover]):
            # alway crossover -> new individual
            idx_notsame = np.where(pa.genes != pb.genes)[0].tolist()
            if len(idx_notsame) == 0:
                idx_crossover = np.ones((self.dim_uss, ))
            else:
                idx_crossover[np.random.choice(idx_notsame)] = 1

        #like pa
        oa = self.IndClass(np.where(idx_crossover, np.clip(0.5*((1 + beta) * pa.genes + (1 - beta) * pb.genes), 0, 1), p_of_oa))
        #like pb
        ob = self.IndClass(np.where(idx_crossover, np.clip(0.5*((1 - beta) * pa.genes + (1 + beta) * pb.genes), 0, 1), p_of_ob))

        #swap
        if skf_ob == skf_oa:
            idx_swap = np.where(np.random.rand(self.dim_uss) < 0.5)[0]
            oa.genes[idx_swap], ob.genes[idx_swap] = ob.genes[idx_swap], oa.genes[idx_swap]
        
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob

        return oa, ob

