import cma
import cma.purecma as purecma


##############


def launch_cmaes_full(center, sigma, ma_func , nbeval=10000, display=True):
    res = cma.CMAEvolutionStrategy(center, sigma).optimize(ma_func,maxfun=nbeval).result
    print("res = ",res)
    return res[1]

def launch_cmaes_full_genotype(center, sigma, ma_func , nbeval=10000, display=True):
    res = cma.CMAEvolutionStrategy(center, sigma).optimize(ma_func,maxfun=nbeval).result
    print("res = ",res)
    return res[0]


# do not use restart
def launch_cmaes_pure(center, sigma, ma_func, nbeval=10000, display=True, ):
    res = purecma.fmin(ma_func,center,sigma,maxfevals=nbeval)
    return res[1].result[1]

