from configs import Config
from systems import Burgers
from solver import SolverRK
from parareal import PararealLight

import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


dx = 128 # space discretization points
Ng = 4 # number of time steps for the coarse solver, per interval
Nf = 400 # number of time steps for the fine solver, per interval
N = 128 # number of intervals
G = 'RK1' # coarse solver
F = 'RK8' # fine solver

ode = Burgers(d_x=dx, normalization='-11')
f = ode.get_vector_field()

T = 5.9
s = SolverRK(f, G=G, Ng=Ng, F=F, Nf=Nf)
p = PararealLight(ode, s, (0, T), N=N, epsilon=5e-7)

res_para = p.run(light=True)
res_elm = p.run(model='elm', degree=1, m=3, light=True)
res_nngp = p.run(model='nngp', light=True, nn=18)
p.plot(skip=[0,1,4])