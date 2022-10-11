from ..EA import * 
from ..operators import Crossover, Mutation, Search, Selection 
from ..tasks.task import AbstractTask 
from . import AbstractModel 

class model(AbstractModel.model):

    class rmp_lsa21: 
        def __init__(self, default_rmp= 0.5) -> None:
            self.default_rmp = default_rmp
            self.C = 0.02 
            self.best_partner = None 


            self.rmp = np.zeros(shape=(self.nb_tasks, self.nb_tasks)) + self.default_rmp
            self.best_partner = np.zeros(shape= (self.nb_tasks), dtype= int) - 1 

            self.s_rmp = np.empty(shape= (self.nb_tasks, self.nb_tasks,0)).tolist()
            self.diff_f_inter_x = np.empty(shape=(self.nb_tasks, self.nb_tasks,0)).tolist() 
        


            
    def __init__(self, seed= None, percent_print= 2): 
        super().__init__(seed, percent_print) 
    
    def compile(self, 
        IndClass: Type[Individual], 
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_LSA21, 
        mutation: Mutation.AbstractMutation,
        search: Search.LocalSearch_DSCG, 
        selection: Selection.ElitismSelection,
        *args, **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
        self.search = search 
        self.search.getInforTasks(IndClass, tasks, seed= self.seed) 
    
    def fit(self, nb_generations: int, nb_inds_each_task: int, nb_inds_min= None,
            lr= 1,
            evaluate_initial_skillFactor = False,
            *args, **kwargs): 
            super().fit(*args, **kwargs)

            # nb_inds_min 
            if nb_inds_min is not None: 
                assert nb_inds_each_task >= nb_inds_min 
            else: 
                nb_inds_min = nb_inds_each_task 
            
            # init history of rmp 
            self.history_rmp = [] 

            # init population 
            population = Population(
                self.IndClass, 
                nb_inds_tasks= [nb_inds_each_task] * len(self.tasks), 
                dim = self.dim_uss, 
                list_tasks = self.tasks, 
                
            )

            MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks) 
            eval_k = np.zeros(len(self.tasks)) 


            gen = 0 
            stop = False # use for 

            self.history_cost.append([ind.fcost for ind in population.get_solves()])
            epoch= 0 

            while np.sum(eval_k) <= MAXEVALS and stop is False: 
                gen += 1 
                offsprings = Population(
                    self.IndClass, 
                    nb_inds_tasks=[0] * len(self.tasks), 
                    dim= self.dim_uss, 
                    list_tasks= self.tasks
                )

                for skf in range(0, len(self.tasks)): 

                    for idx_ind, ind in enumerate(population[skf]): 
                        partner_task = np.random.choice(len(self.tasks))
                        if partner_task == skf: 
                            # intra
                            off = self.search(ind = ind, population = population)
                            offsprings[skf].__addIndividual__(off)
                            eval_k[skf] += 1 
                        else: 
                            rmp = 0 
                            if self.crossover.best_partner[skf] == partner_task: 
                                rmp =1 
                            else : 
                                mu_rmp = self.crossover.rmp[skf][partner_task] 
                                
                                rmp = np.random.normal(loc= mu_rmp, scale= 0.1 )
                                while rmp <= 0 or rmp > 1: 
                                    rmp = np.random.normal(loc= mu_rmp, scale= 0.1 )

                            if np.random.rand() <= rmp: 
                                ind2 = population[partner_task].__getRandomItems__(size=1)[0]
                                oa, ob = self.crossover(ind, ind2, skf, skf)

                                oa.fcost = oa.eval(self.tasks[oa.skill_factor])
                                ob.fcost = ob.eval(self.tasks[ob.skill_factor])
                                
                                survival = oa 
                                if oa.fcost > ob.fcost : 
                                    survival = ob 
                                
                                eval_k[skf] += 2 

                                delta = ind.fcost - survival.fcost 
                                if delta == 0: 
                                    offsprings[skf].__addIndividual__(survival) 
                                elif delta > 0: 
                                    self.crossover.s_rmp[skf][partner_task].append(rmp) 
                                    self.crossover.diff_f_inter_x[skf][partner_task].append(delta)
                                    offsprings[skf].__addIndividual__(survival) 
                                else: 
                                    offsprings[skf].__addIndividual__(ind)
                            else: 
                                off = self.search(ind = ind, population = population)
                                offsprings[skf].__addIndividual__(off)
                                eval_k[skf] += 1 
                    

                    population.ls_subPop[skf] = offsprings.ls_subPop[skf]
                    # population.update_rank() 
                    population[skf].update_rank() 
                    if gen % 50 == 0: 
                        ls = Search.LocalSearch_DSCG() 
                        ls.getInforTasks(self.IndClass, self.tasks, seed = self.seed) 

                        best_ind = population[skf].getSolveInd() 
                        evals, new_ind = ls.search(best_ind, fes=2000)
                        if new_ind.fcost < best_ind.fcost: 
                            idx = int(np.where(population[skf].factorial_rank == 1)[0])
                            population[skf].ls_inds[idx].genes = new_ind.genes 
                            population[skf].ls_inds[idx].fcost = new_ind.fcost 

                # selection
                nb_inds_tasks = [int(
                    # (nb_inds_min - nb_inds_each_task) / nb_generations * (epoch - 1) + nb_inds_each_task
                    int(max((nb_inds_min - nb_inds_each_task)*(np.sum(eval_k)/MAXEVALS) + nb_inds_each_task, nb_inds_min))
                )] * len(self.tasks)

                self.selection(population, nb_inds_tasks)
                # update operators
                self.crossover.update(population = population)
                self.mutation.update(population = population)
                self.search.update(population) 

                # '''local search'''
                # if gen % 50 == 0: 
                #     # if (epoch - before_epoch) > kwargs['step_over']: 
                #     for skf in range(len(self.tasks)): 
                        
                #         ls = Search.LocalSearch_DSCG()
                #         ls.getInforTasks(self.IndClass, self.tasks, seed= self.seed)
                        
                #         ind = population[skf].getSolveInd()
                #         evals, new_ind = ls.search(ind, fes = 2000)
                #         eval_k[skf] += evals
                #         if new_ind.fcost < ind.fcost : 
                #             idx = int(np.where(population[skf].factorial_rank == 1)[0])
                #             population[skf].ls_inds[idx].genes = new_ind.genes 
                #             population[skf].ls_inds[idx].fcost = new_ind.fcost 
                #             # population[skf].ls_inds[0].genes= new_ind.genes 
                #             # population[skf].ls_inds[0].fcost = new_ind.fcost
                #             population.update_rank()  

                if np.sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history
                    self.history_cost.append([ind.fcost for ind in population.get_solves()])
                    # self.history_smp.append([M_smp[i].get_smp() for i in range(len(self.tasks))])

                    self.render_process(np.sum(eval_k)/MAXEVALS, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True, print_format_e= False)

                    # update mutation
                    self.mutation.update(population = population)

                    epoch = np.sum(eval_k)/MAXEVALS*nb_generations
                    epoch= int(epoch)
            
            
            self.last_pop = population
            self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(population)], self.history_cost[-1]], use_sys= True, print_format_e= False)
            print()
            # print(p_choose_father)
            print(eval_k)
            print('END!')
            return self.last_pop.get_solves()            
            



