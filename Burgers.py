
from mpi4py.futures import MPIPoolExecutor
import pickle
from configs import Config
from systems import FHN_PDE, DiffReact, ODE, Burgers, BurgersFast
from solver import SolverRK, SolverScipy, SolverAbstr
from parareal import PararealLight
import numpy as np
import os
import sys

from pararealml.initial_condition import DiscreteInitialCondition
from pararealml.initial_value_problem import InitialValueProblem
from pararealml.operator import Operator
from pararealml import ShallowWaterEquation, Mesh, NeumannBoundaryCondition, ConstrainedProblem, GaussianInitialCondition, vectorize_bc_function
from pararealml.operators.fdm import FDMOperator, ThreePointCentralDifferenceMethod, RK4, ForwardEulerMethod


import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# print(sys.argv, sys.argv[-3:])
N, mdl, dx = sys.argv[-3:] 
N = int(N)
dx = int(dx)


if dx == 128:
    Ng = 4
    Nf = 4*10000
    thresh = Nf/200
    assert N == 128
    use_jax = True
    ode = Burgers(d_x=dx, normalization='-11', use_jax=use_jax)
elif dx == 1128:
    Ng = 293
    Nf = 4*10000*15
    thresh = 1e7
    assert N == 128
    use_jax = False
    ode = BurgersFast(d_x=dx, normalization='-11', use_jax=use_jax)
else:
    raise Exception('Invalid dx val')


_f0 = ode.get_vector_field()
def f(t, u):
    return _f0(t, u)


if __name__ == '__main__':
    
    avail_work = int(os.getenv('SLURM_NTASKS'))
    workers = avail_work - 1
    print('Total workes', workers)
    pool = MPIPoolExecutor(workers)

    
    print(N, mdl, dx, ode.name)
    
    
    ########## CHANGE THIS #########
    dir_name = 'BurgersScal'
    ################################
    
    name = f'{dir_name}_{dx}_{N}_{mdl}'
    assert workers >= N
    
    # generate folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    
    ########## CHANGE THIS #########
        
    T = 5.9
    # thresh = 4*10000/100
    s = SolverRK(f, G='RK1', Ng=Ng, F='RK8', Nf=Nf, thresh=thresh, use_jax=use_jax)
    p = PararealLight(ode, s, (0, T), N=N, epsilon=5e-7)


    
    #####################################
    
    # run the code, storing intermediates in custom folder
    if mdl == 'para':
        res = p.run(pool=pool, parall='mpi', light=True)
    elif mdl == 'elm':
        res = p.run(model='elm', degree=1, m=3, pool=pool, parall='mpi', light=True)
    elif mdl == 'nngp':
        res = p.run(model='nngp', pool=pool, parall='mpi', light=True, 
                    nn=18)
    elif mdl == 'gp':
        res = p.run(model='gpjax', pool=pool, parall='mpi', light=True)
    else:
        raise Exception('Unknown model type', mdl)
        
    res['timings'].pop('by_iter')
    print(res['timings'])
        
    # dump the final result
    p.store(name=name, path=dir_name)