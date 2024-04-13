from mpi4py.futures import MPIPoolExecutor
import pickle
from configs import Config
from systems import FHN_PDE, DiffReact, ODE
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
dx = float(dx)

tspan = (0.0, 20.0)

if dx == 0.1:
    mul = 7
    mulF = 100
    assert N == 235
elif dx == 0.0701:
    mul = 8
    mulF = 200
    assert N == 235
elif dx == 0.05:
    mul = 14
    mulF = 400
    assert N == 235
elif dx == 0.038:
    mul = 24
    mulF = 500
    assert N == 235
else:
    raise Exception('Invalid dx val')



class PararealMLSolver(SolverAbstr):
    def __init__(self, ivp: InitialValueProblem, f: Operator, g: Operator):
        self.ivp = ivp
        self.f = f
        self.g = g
        self.dg = g.d_t
        self.df = f.d_t
        self.cp = ivp.constrained_problem
        self.vertex_oriented = f.vertex_oriented

        # required for pickling
        # Note: as long as the boundary conditions are static, it doesn't seem to matter
        self.cp._boundary_conditions = None

    def _check_time_step_size(self, t0, t1, dt):
        delta_t = (t1 - t0) 
        if not np.isclose(delta_t, dt * round(delta_t / dt)):
            raise ValueError(
                f"fine operator time step size ({dt}) must be a "
                f"divisor of sub-IVP time slice length ({delta_t})"
            )

    def run_F_full(self, t0, t1, u0):
        self._check_time_step_size(t0, t1, self.df)
        sub_ivp = InitialValueProblem(self.cp, (t0,t1),
                    DiscreteInitialCondition(self.cp, u0, self.vertex_oriented))
        
        sub_y_fine = self.f.solve(sub_ivp, False).discrete_y(self.vertex_oriented)
        return sub_y_fine

    def run_G_full(self, t0, t1, u0):
        self._check_time_step_size(t0, t1, self.dg)
        sub_ivp = InitialValueProblem(self.cp, (t0,t1),
                    DiscreteInitialCondition(self.cp, u0, self.vertex_oriented))
        
        sub_y_fine = self.g.solve(sub_ivp, False).discrete_y(self.vertex_oriented)
        return sub_y_fine
        

    def run_F(self, t0, t1, u0):
        self._check_time_step_size(t0, t1, self.df)
        sub_ivp = InitialValueProblem(self.cp, (t0,t1),
                    DiscreteInitialCondition(self.cp, u0, self.vertex_oriented))
        
        sub_y_fine = self.f.solve_fast(sub_ivp, False)
        return sub_y_fine
    
    def run_G(self, t0, t1, u0):
        self._check_time_step_size(t0, t1, self.dg)
        sub_ivp = InitialValueProblem(self.cp, (t0,t1),
                    DiscreteInitialCondition(self.cp, u0, self.vertex_oriented))
        
        sub_y_fine = self.g.solve_fast(sub_ivp, False)
        return sub_y_fine
    
class PararealMLODE(ODE):
    def __init__(self, ivp:InitialValueProblem, f):
        self.ivp = ivp
        self.f = f
        self.vertex_oriented = f.vertex_oriented
        self.name = f'SWE_{np.prod(ivp.constrained_problem.y_vertices_shape)}'

    def get_dim(self):
        return self.ivp.constrained_problem.y_shape(self.vertex_oriented)

    def get_init_cond(self):
        return self.ivp.initial_condition.discrete_y_0(self.vertex_oriented)
    

                          

def get_ivp(d):

    diff_eq = ShallowWaterEquation(0.5)
    mesh = Mesh([(-5.0, 5.0), (0.0, 5.0)], [d, d])
    bcs = [
        (
            NeumannBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, None, None)),
                is_static=True,
            ),
            NeumannBoundaryCondition(
                vectorize_bc_function(lambda x, t: (0.0, None, None)),
                is_static=True,
            ),
        )
    ] * 2
    cp = ConstrainedProblem(diff_eq, mesh, bcs)
    ic = GaussianInitialCondition(
        cp,
        [(np.array([2.5, 1.25]), np.array([[0.25, 0.0], [0.0, 0.25]]))] * 3,
        [1.0, 0.0, 0.0],
    )
    ivp = InitialValueProblem(cp, tspan, ic)
    return ivp



if __name__ == '__main__':

    # mulF=1
    
    avail_work = int(os.getenv('SLURM_NTASKS'))
    workers = avail_work - 1
    print('Total workes', workers)
    pool = MPIPoolExecutor(workers)

    ivp = get_ivp(dx)

    f = FDMOperator(RK4(), ThreePointCentralDifferenceMethod(), (20/235)/(1000*mulF), tqdm=False)
    g = FDMOperator(ForwardEulerMethod(), ThreePointCentralDifferenceMethod(), (20/235)/mul, tqdm=False)
    N = 235
    ode = PararealMLODE(ivp, f)
    
    print(N, mdl, dx, ode.name)
    
    
    ########## CHANGE THIS #########
    dir_name = 'SWEScal'
    ################################
    
    name = f'{dir_name}_{dx}_{N}_{mdl}'
    assert workers >= N
    
    # generate folder
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    
    ########## CHANGE THIS #########
        
    s = PararealMLSolver(ivp, f, g)
    p = PararealLight(ode, s, tspan=tspan, N=N, epsilon=5e-7)


    
    #####################################
    
    # run the code, storing intermediates in custom folder
    if mdl == 'para':
        res = p.run(pool=pool, parall='mpi', light=True)
    elif mdl == 'elm':
        res = p.run(model='elm', degree=1, m=4, pool=pool, parall='mpi', light=True)
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