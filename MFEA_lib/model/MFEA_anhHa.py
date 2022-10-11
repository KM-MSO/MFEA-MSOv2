from asyncio import tasks
from operator import itemgetter
from re import sub
import numpy as np
import operator
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
import matplotlib.pyplot as plt
import copy

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.GaussMutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
    
    def findParentSameSkill(self, subpop: SubPopulation, ind):
        ind2 = ind 
        while ind2 is ind: 
            ind2 = subpop.__getRandomItems__(size= 1)[0]
        
        return ind2 
    
    def Linear_population_size_reduction(self, evaluations, current_size_pop, max_eval_each_tasks, max_size, min_size):
        for task in range(len(self.tasks)):
            new_size = (min_size[task] - max_size[task]) * evaluations[task] / max_eval_each_tasks[task] + max_size[task] 

            new_size= int(new_size) 
            if new_size < current_size_pop[task]: 
                current_size_pop[task] = new_size 
        
        return current_size_pop 
    def get_elite(self,pop,size):
        elite_subpops = []
        for i in range(len(pop)):
            idx = np.argsort(pop[i].factorial_rank)[:size].tolist()
            elite_subpops.append(pop[i][idx])
        return elite_subpops
    def get_elite_transfer (self, pop, size) :
        elite_transfer = []
        idx = np.argsort(pop.factorial_rank)[:size].tolist()
        elite_transfer += pop[idx]
        return elite_transfer
    def distance(self,a,b):
        return np.linalg.norm(a-b)        
    def rank_distance(self,subpops,x:Individual):
        dist = []
        for i in subpops:
            dist.append(self.distance(i.genes,x.genes))
        return np.argsort(np.argsort(dist)) + 1
    def get_pop_similarity(self,subpops):
        k = len(self.tasks)
        rmp = np.ones((k,k))
        for i in range(k):
            for j in range(k):
                x =  self.rank_distance(subpops[i],subpops[i][0])
                y = self.rank_distance(subpops[i],subpops[j][0])
                rmp[i][j] = np.sum([np.abs(i+1-x[i]) for i in range(len(x))]) / np.sum([np.abs(i+1-y[i]) for i in range(len(y))])
        return rmp
    def get_pop_intersection(self,subpops):
        k = len(self.tasks)
        rmp = np.zeros([k,k])
        for i in range(k):
            DT = 0           
            for u in range(20):
                tmp = 9999999
                for v in range(20):
                    if v != u: 
                        tmp =min(self.distance(subpops[i][u],subpops[i][v]),tmp)
                DT+=tmp
            for j in range(k):
                DA = 0 
                for u in range(20):
                    tmp = 9999999
                    for v in range(20):
                        tmp =min(self.distance(subpops[i][u],subpops[j][v]),tmp)
                    DA+=tmp
                if j != i:
                    rmp[i][j] = np.float64( DT / DA)
        return rmp
    def get_pop_intersection_v2(self,subpops):
        k = len(self.tasks)
        rmp = np.ones((k,k))
        for i in range(k):
            DT = 0        
            for u in range(20):
                DT+=self.distance(subpops[i][u],subpops[i][0])
            for j in range(k):
                DA = 0 
                for u in range(20):
                    DA+=self.distance(subpops[i][0],subpops[j][u])
                if j != i:
                    rmp[i][j] = np.float64( DT / DA)
        return rmp
    def renderRMP(self,tmp, title = None, figsize = None, dpi = 200):
        if figsize is None:
            figsize = (30, 30)
        if title is None:
            title = self.__class__.__name__
        fig = plt.figure(figsize= figsize, dpi = dpi)
        fig.suptitle(title, size = 15)
        fig.set_facecolor("white")
        for i in range(len(self.tasks)):
            for j in range(len(self.tasks)):
                x=[]
                # for k in range(1000):
                for k in range(len(tmp)):
                    if i!=j:
                        x.append(tmp[k][i][j])
                        plt.subplot(int(np.ceil(len(self.tasks) / 3)), 3, i + 1)
                        plt.plot(x,label= 'task: ' +str(j + 1))
                        plt.legend()
              
            plt.title('task ' + str( i + 1))
            plt.xlabel("Epoch")
            plt.ylabel("M_rmp")
            plt.ylim(bottom = -0.1, top = 1.1)
    def RoutletWheel(self,rmp,rand):
        tmp = [0]*len(self.tasks)
        tmp[0]= rmp[0]
        for i in range(1,len(tmp)):
            tmp[i]=tmp[i-1]+rmp[i]
        index =0 
        while tmp[index] < rand:
            index+=1
            if index == len(self.tasks) - 1:
                return index
        return index 

    def distance_to_pop (self,inv, pop_elite, fRank) :
        sum = 0
        for individual in pop_elite :
            sum += self.distance(inv.genes, individual.genes)
        tmp = 1/sum * (1+1/fRank)
        return tmp
    def get_max_IM (self,IM_i) :
        b = {}
        for i in range(len(self.tasks)) :
            b[i] = IM_i[i]
        temp = sorted(b.items(), key = operator.itemgetter(1), reverse=True)
        return temp
        


    def fit(self, nb_inds_each_task: int, nb_inds_min:int,nb_generations :int ,  bound = [0, 1], evaluate_initial_skillFactor = False,LSA = False,
            *args, **kwargs): 
        super().fit(*args, **kwargs)
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )
        #history
        self.IM = []
        self.rmp_hist = []
        self.inter_task = []
        len_task  = len(self.tasks)
        inter =  [1-1/len_task]*len_task
        intra =  [1/len_task]*len_task
        rmp = np.zeros([len_task,len_task])

        #SA param
        nb_inds_tasks = [nb_inds_each_task]*len(self.tasks)
        MAXEVALS = nb_generations * nb_inds_each_task * len(self.tasks)
        epoch =0 
        eval_k = np.zeros(len(self.tasks))

        while np.sum(eval_k) <= MAXEVALS:
            offsprings = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )
            elite = self.get_elite(population.ls_subPop,20)
            # IM =self.get_pop_intersection_v2(elite)
            measurement = np.zeros((len_task,len_task))
            if np.sum(eval_k) >= epoch * nb_inds_each_task * len(self.tasks):
                    # save history 
                self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
                self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)

                # self.IM.append(np.copy(IM))
                self.rmp_hist.append(np.copy(rmp))
                epoch+=1
      
            if (epoch % 5 == 0) : 
                for i in range(len_task):
                    for j in range(len_task):
                        if i != j :
                            for inv1 in range(20):
                                measurement[i,j] += self.distance_to_pop(elite[j][inv1], elite[i], inv1 + 1)
                for i in range(len_task):
                    sum_tmp = np.sum(measurement[i])
                    for j in range(len_task):
                        if i != j:
                            rmp[i,j] = measurement[i,j] / sum_tmp * inter[i]
                    rmp[i,i]= intra[i]


            if (epoch % 20) == 19 :
                IM =self.get_pop_intersection_v2(elite)
                for i in range(len(self.tasks)) :
                    arr = self.get_max_IM(IM[i])
                    for t in arr :
                        j = int(t[0])
                        transfer = min (3, int(IM[i,j] * nb_inds_tasks[i]))
                        transfer = max(1, transfer)
                        pop_transfer = self.get_elite_transfer (population[j], transfer)
                        for inv in pop_transfer :
                            gen = np.copy(inv.genes)
                            tmp_inv = self.IndClass (gen)
                            tmp_inv.skill_factor = i
                            tmp_inv.hybrid = False
                            eval_k[i] += 1
                            offsprings.__addIndividual__(tmp_inv)
                        if len(offsprings[i]) >= nb_inds_tasks[i] / 2 :
                            break
                self.IM.append(np.copy(IM))
            # create new offspring population 
            for i in range(len_task):
                    while(len(population.ls_subPop[i])+len(offsprings.ls_subPop[i]) < 2* nb_inds_tasks[i]):
                        if np.random.rand() < 0.2:
                            pa = population.__getIndsTask__(idx_task=i,type='tournament',tournament_size=1 )
                            oa = self.mutation(pa,return_newInd=True)
                            oa.skill_factor = pa.skill_factor
                            offsprings.__addIndividual__(oa)
                            eval_k[oa.skill_factor]+=1
                        else:
                            k =self.RoutletWheel(rmp[i],np.random.rand())
                            pa = population.__getIndsTask__(idx_task=i,type='tournament',tournament_size= 2 )
                            pb = population.__getIndsTask__(idx_task=k ,type='tournament',tournament_size=1)
                            oa,ob = self.crossover(pa,pb)
                            oa.skill_factor = pa.skill_factor
                            ob.skill_factor = pa.skill_factor
                            if i!=k:
                                oa.transfer =True
                                ob.transfer =True
                            else :
                                oa.transfer = False
                                ob.transfer = False
                            offsprings.__addIndividual__(oa)
                            offsprings.__addIndividual__(ob)
                            eval_k[oa.skill_factor]+=1
                            eval_k[ob.skill_factor]+=1
            offsprings.update_rank()                
            elite_off = self.get_elite(offsprings.ls_subPop,40)
            for i in range(len(self.tasks)):
                x = 0 #inter
                y= 0 #intra
                for inv in elite_off[i] :
                    if inv.transfer == True :
                        x += 1
                    else : 
                        y += 1
                y_tmp = y/(x+y)
                y_tmp = max(0.2,min(0.8,y_tmp))
                x_tmp = 1-y_tmp

                   
                inter[i] =0.5*inter[i]+x_tmp*0.5
                intra[i] =0.5*intra[i]+y_tmp*0.5
            tmp_inter = copy.deepcopy(inter)
            self.inter_task.append(tmp_inter)
            # merge and update rank

            population = population + offsprings
            population.update_rank()

            # selection 
            if LSA is True:
                nb_inds_tasks = [int(
                    # (nb_inds_min - nb_inds_each_task) / nb_generations * (epoch - 1) + nb_inds_each_task
                    int(min((nb_inds_min - nb_inds_each_task)/(nb_generations - 1)* epoch  + nb_inds_each_task, nb_inds_each_task))
                )] * len(self.tasks)
            self.selection(population, nb_inds_tasks)
           
            # save history
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
                
            self.render_process(epoch/nb_generations, ['Pop_size', 'Cost'], [[len(u) for u in population.ls_subPop], self.history_cost[-1]], use_sys= True)

            # self.IM.append(np.copy(IM))
            self.rmp_hist.append(np.copy(rmp))
        print("End")
        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves()




