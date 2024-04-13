
from mpi4py.futures import MPIPoolExecutor
import pickle
from configs import Config
from systems import FHN_PDE, DiffReact
from solver import SolverRK, SolverScipy
from parareal import PararealLight
import numpy as np
import os
import sys

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# print(sys.argv, sys.argv[-3:])
N, mdl, dx = sys.argv[-3:] 
N = int(N)
dx = int(dx)
d_y = dx

T = 20

if dx == 19:
    mul = 1
    G = 'RK1'
    F = 'RK4'
    mulF = 1800e2
    assert N == 64
elif dx == 28:
    mul = 1
    G = 'RK1'
    F = 'RK4'
    mulF = 1200e2
    assert N == 128
elif dx == 41:
    mul = 1
    G = 'RK1'
    F = 'RK4'
    mulF = 800e2
    assert N == 256
elif dx == 77:
    mul = 1
    G = 'RK4'
    F = 'RK8'
    mulF = 300e2
    assert N == 512
elif dx == 113:
    mul = 2
    G = 'RK4'
    F = 'RK8'
    mulF = 400e2
    assert N == 512
elif dx == 164:
    mul = 4
    G = 'RK4'
    F = 'RK8'
    mulF = 500e2
    assert N == 512
elif dx == 235:
    mul = 8
    G = 'RK4'
    F = 'RK8'
    mulF = 600e2
    assert N == 512
else:
    raise Exception('Invalid dx val')
                          

ode = DiffReact(d_x=dx, use_jax=False, normalization='-11')

_f = ode.get_vector_field()

def f(t, u):
    return _f(t, u)


if __name__ == '__main__':
    
    avail_work = int(os.getenv('SLURM_NTASKS'))
    workers = avail_work - 1
    print('Total workes', workers)
    pool = MPIPoolExecutor(workers)
    
    print(N, mdl, dx, ode.name)
    
    
    ########## CHANGE THIS #########
    dir_name = 'DiffReactScal'
    ################################
    
    name = f'{dir_name}_{dx}_{N}_{mdl}'
    assert workers >= N
    
    # generate folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    
    ########## CHANGE THIS #########
        
    s = SolverScipy(f, mul, mulF, G, F, verbose=False, use_jax=False)
    p = PararealLight(ode, s, tspan=(0, T), N=N)
    
    #####################################
    
    # run the code, storing intermediates in custom folder
    if mdl == 'para':
        res = p.run(pool=pool, parall='mpi', light=True)
    elif mdl == 'elm':
        res = p.run(model='elm', degree=1, m=3, pool=pool, parall='mpi', light=True)
    elif mdl == 'nngp':
        res = p.run(model='nngp', pool=pool, parall='mpi', light=True, 
                    nn=20)
    elif mdl == 'gp':
        res = p.run(model='gpjax', pool=pool, parall='mpi', light=True)
    else:
        raise Exception('Unknown model type', mdl)
        
    res['timings'].pop('by_iter')
    print(res['timings'])
        
    # dump the final result
    p.store(name=name, path=dir_name)