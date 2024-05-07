import pickle
from configs import Config
from systems import Burgers
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
from mpi4py.futures import MPIPoolExecutor
from itertools import product

dx = 128
ode = Burgers(d_x=dx, normalization='-11')

_f = ode.get_vector_field()

def f(t, u):
    return _f(t, u)

N=128
T = 5.9
Ng = 4
s = SolverRK(f, G='RK1', Ng=Ng, F='RK8', Nf=200)
p = PararealLight(ode, s, (0, T), N=N, epsilon=5e-7, verbose='')

def do(itrs):
    seed_offset, res_size, m = itrs
    seed = 8795345 + seed_offset
    try:
        out = p.run(model='elm', degree=1, m=m, res_size=res_size, seed=seed, light=True)
        res = [seed_offset, res_size, m, out['k'], out['timings']['runtime']]
    except Exception as e:
        res = [seed_offset, res_size, m, 150]
        print(e)
    return res

if __name__ == '__main__':
    
    avail_work = int(os.getenv('SLURM_NTASKS'))
    workers = avail_work - 1
    print('Total workes', workers)

    itrs = list(product(range(100), range(20, 510, 10), range(2, 21, 1)))

    pool = MPIPoolExecutor(max_workers=workers)
    out = list(pool.map(do, itrs))
    # print(out)
    # str_out = '\n'.join([','.join(l) for l in out])

    print('done')

    with open('Burgers_perf_across_m.txt', 'w') as w:
        w.write(str(out))
    pickle.dump(out, open('Burgers_perf_across_m.pkl', 'wb'))