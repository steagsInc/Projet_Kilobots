import os
import random

from deap import creator, tools, base, algorithms, cma
import numpy as np

from Src.simulationController.topologyOptimizer import topologyOptimisation

print("Début du test de l'extracteur des propriétés de l'essaim sur le chemin : ", os.getcwd())
os.chdir("../..")
S = topologyOptimisation("pile",nb=150,visible=False,time=3000)



def fitnessShapeIndex(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    return S.Swarm.SumshapeIndex()

def fitnessRectanglitude(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    return np.array(S.Swarm.rectanglitude()).mean()

def fitnessAggregation(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    S.Swarm.shapeIndex()
    #print("Nombre de turing de ",-S.Swarm.nb_turing_spots)
    S.Swarm.calculerTuringSpots(seuil=4)
    #print("Précision : ",S.getPrecision())
    #print("Shape de ", -S.Swarm.SumshapeIndex())
    return (S.Swarm.nb_turing_spots*S.Swarm.SumshapeIndex())

def varianceUV(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.readDatas()
    return np.std(S.Swarm.concentrations,axis=0).sum()+np.mean(np.max(S.Swarm.concentrations,0))



def fitnessTuringSpot(w):
    S.put_genotype(w)
    S.computeSimulation()
    S.Swarm.setRange(80)
    #S.Swarm.shapeIndex()
    S.Swarm.calculerTuringSpots(seuil=4)
    return S.Swarm.nb_turing_spots

def NSGA(funcs_l, weights, var, sigma, MU=12, NGEN=50,wide_search=1.5):
    IND_SIZE = len(var)
    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)
    toolbox = base.Toolbox()

    eval_funcs = lambda x: tuple([f(x) for f in funcs_l])
    toolbox.register("evaluate", eval_funcs)
    S.model = var
    c = S.extract_genotype()

    init_func = lambda c, sigma, size: np.random.normal(c, sigma, size)
    bound_max = list(wide_search*c)
    bound_min = list(-wide_search*c)
    toolbox.register("attr_float", init_func, c, sigma, len(var))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    paretofront = tools.ParetoFront()
    #toolbox.register("mutate", tools.mutGaussian, mu=c, sigma=sigma, indpb=1.0 / IND_SIZE)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=bound_min, up=bound_max, eta=20.0, indpb=1.0 / IND_SIZE)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=bound_min, up=bound_min, eta=20.0)

    toolbox.register("select", tools.selNSGA2)
    CXPB = 0.6
    L = []
    turing_spot = tools.Statistics(lambda ind: ind.fitness.values[0])
    shape_index = tools.Statistics(lambda ind: ind.fitness.values[2])
    rectanglitude = tools.Statistics(lambda ind: ind.fitness.values[1])

    mstats = tools.MultiStatistics(Rectanglitude=rectanglitude, Shape_Index=shape_index, Turing_Spot = turing_spot)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    pop = toolbox.population(n=MU)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    pop = toolbox.select(pop, len(pop))
    record = mstats.compile(pop)
    #print("Record  =" ,record)
    logbook.record( **record)
    print(logbook.stream)
    # Begin the generational process
    for gen in range(1, NGEN):
        # Vary the population
        if(gen%5==0):
            S.Swarm.controller.withVisiblite(True)
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(ind1, ind2)
            toolbox.mutate(ind1)
            toolbox.mutate(ind2)
            del ind1.fitness.values, ind2.fitness.values
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Select the next generation population
        pop = toolbox.select(pop + offspring, MU)
        record = mstats.compile(pop)
        logbook.record(g=gen, **record)
        print(logbook.stream)

    return pop, paretofront




def cmaES(funcs_l , weights,lambd , mu, var, sigma,ngen):
    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)
    toolbox = base.Toolbox()
    eval_funcs = lambda x: tuple([f(x) for f in funcs_l])
    toolbox.register("evaluate", eval_funcs)
    S.Swarm.controller.rez_params()
    S.model = var
    c = S.extract_genotype()

    init_func = lambda  c,sigma,size : np.random.normal(c,sigma,size)

    toolbox.register("attr_float", init_func, c, sigma, len(var))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)



    strategy = cma.Strategy(centroid=c * len(var), sigma=sigma, lambda_=lambd * len(var))
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaGenerateUpdate(toolbox, ngen=ngen, stats=stats, halloffame=hof)
    return stats, hof


def CMAES_MO(var,weights,funcs_l,sigma,verbose = True, MAXITER = 100, STAGNATION_ITER =10, lambda_=3, mu =5):
    NRESTARTS = 10  # Initialization + 9 I-POP restarts

    creator.create("MaFitness", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.MaFitness)
    toolbox = base.Toolbox()
    eval_funcs = lambda x: tuple([f(x) for f in funcs_l])
    toolbox.register("evaluate", eval_funcs)
    S.Swarm.controller.rez_params()
    S.model = var
    c = S.extract_genotype()

    init_func = lambda c, sigma, size: np.random.normal(c, sigma, size)

    toolbox.register("attr_float", init_func, c, sigma, len(var))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_funcs)

    halloffame = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: np.array([ind.fitness.values]))
    stats.register("average", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbooks = list()
    bestvalues = list()
    medianvalues = list()
    i = 0
    t=0
    while i < (NRESTARTS):
        pop = toolbox.population(n=mu)
        strategy = cma.StrategyMultiObjective(centroid=c, sigma=sigma, lambda_=lambda_,population=pop)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        logbooks.append(tools.Logbook())
        logbooks[-1].header = "gen", "evals", "restart", "regime", "std", "min", "avg", "max"
        conditions = {"MaxIter": False, "TolHistFun": False, "EqualFunVals": False,
                      "TolX": False, "TolUpSigma": False, "Stagnation": False,
                      "ConditionCov": False, "NoEffectAxis": False, "NoEffectCoor": False}
        while not any(conditions.values()):
            if(t%5 == 0):
                S.Swarm.controller.visibility = True
            else:
                S.Swarm.controller.visibility = False
            t = t + 1
            # Generate a new population
            population = pop + toolbox.generate()
            # Evaluate the individuals
            fitnesses = toolbox.map(toolbox.evaluate, population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit
            halloffame.update(population)
            record = stats.compile(population)
            logbooks[-1].record(gen=t, restart = i, **record)
            if verbose:
                print(logbooks[-1].stream)
            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            # Log the best and median value of this population
            bestvalues.append(population[-1].fitness.values)
            medianvalues.append(population[int(round(len(population)/2.))].fitness.values)
            if t >= MAXITER:
                # The maximum number of iteration per CMA-ES ran
                conditions["MaxIter"] = True
            if len(bestvalues) > STAGNATION_ITER and len(medianvalues) > STAGNATION_ITER and \
                    np.median(bestvalues[-20:]) >= np.median(
                bestvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]) and \
                    np.median(medianvalues[-20:]) >= np.median(
                medianvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]):
                # Stagnation occured
                conditions["Stagnation"] = True
            pop = [p.fitness.values for p in population[-1:-mu]]
        stop_causes = [k for k, v in conditions.items() if v]
        print( "Stopped because of condition%s %s" % ((":" if len(stop_causes) == 1 else "s:"), ",".join(stop_causes)))
        i += 1




if(__name__=="__main__"):
    w = S.extract_genotype()
    #S.Swarm.controller.rez_params()

    #funcs_l = [lambda x:2*x ,lambda x:5*x+3]
    #eval_funcs = lambda x: tuple([f(x) for f in funcs_l])
    #print(eval_funcs(3))
    #cmaES([fitnessTuringSpot, fitnessRectanglitude] , (+2,+1) , 5 , 10,('A_VAL', 'B_VAL', 'C_VAL', 'D_u', 'D_v'), 0.1,10)
    #CMAES_MO(('A_VAL', 'B_VAL', 'C_VAL', 'D_u', 'D_v'), (+1,-1), [fitnessRectanglitude,fitnessShapeIndex], 0.01, verbose = True, MAXITER = 10, STAGNATION_ITER =10)
    NSGA([fitnessTuringSpot,fitnessRectanglitude, fitnessShapeIndex],(+5,+1,-10),('D_u', 'D_v'),sigma=2)
    S.Swarm.controller.withVisiblite(True)
    S.Swarm.controller.withTime(-1)
    S.Swarm.controller.withNombre(200)
    S.Swarm.controller.withTopology("pile")

    S.Swarm.controller.run()
