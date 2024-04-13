#%%
from parareal import Parareal, PararealLight
from systems import FHN_ODE, Lorenz, Rossler, Brusselator, Hopf, DblPend, ThomasLabyrinth
from configs import Config
from solver import SolverRK


import warnings
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def test(s1, s2, kpara, knngp):
    r1 = s1.run()
    r2 = s2.run()
    assert r1['k'] == r2['k']
    assert r1['k'] == kpara

    r1 = s1.run(model='nngp', pool=5)
    r2 = s2.run(model='nngp', pool=5)
    assert r1['k'] == r2['k']
    assert r1['k'] == knngp

res = []
res_size = 20
# FHN
ode = FHN_ODE(normalization='-11')
config = Config(ode).get()
s = SolverRK(ode.get_vector_field(), **config)
s1 = Parareal(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r = s1.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
# res.append(r['k'])
s2 = PararealLight(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r2 = s2.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
assert r['k'] == r2['k']
assert r['k'] == 7
test(s1, s2, 11, 5)

# lorenz
ode = Lorenz(normalization='-11')
config = Config(ode).get()
s = SolverRK(ode.get_vector_field(), **config)
s1 = Parareal(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r = s1.run(model='elm', nn=4, res_size=res_size, M=1, R=1, alpha=0, degree=3)
# res.append(r['k'])
s2 = PararealLight(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r2 = s2.run(model='elm', nn=4, res_size=res_size, M=1, R=1, alpha=0, degree=3)
assert r['k'] == r2['k']
assert r['k'] == 9
test(s1, s2, 15, 9)

# Rossler
ode = Rossler(normalization='-11')
config = Config(ode).get()
s = SolverRK(ode.get_vector_field(), **config)
s1 = Parareal(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r = s1.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
# res.append(r['k'])
s2 = PararealLight(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r2 = s2.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
assert r['k'] == r2['k']
assert r['k'] == 12
test(s1, s2, 18, 12)

# NonAut
ode = Hopf(normalization='-11')
config = Config(ode, N=32).get()
s = SolverRK(ode.get_vector_field(), **config)
s1 = Parareal(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r = s1.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
# res.append(r['k'])
s2 = PararealLight(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r2 = s2.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
assert r['k'] == r2['k']
assert r['k'] == 9
test(s1, s2, 19, 10)

# brus - not working
# s1 = Parareal(ode_name='brus_2d_n', normalization='-11', epsilon=5e-7)
# r = s1.run(model='elm', nn=16, res_size=10, M=1, R=1, alpha=1, degree=2)
# res.append(r['k'])

# Double pend
ode = DblPend(normalization='-11')
config = Config(ode).get()
s = SolverRK(ode.get_vector_field(), **config)
s1 = Parareal(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r = s1.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
# res.append(r['k'])
s2 = PararealLight(ode, s, config['tspan'], config['N'], epsilon=5e-7)
r2 = s2.run(model='elm', nn=5, res_size=res_size, M=1, R=1, alpha=0, degree=2)
assert r['k'] == r2['k']
assert r['k'] == 10
test(s1, s2, 15, 10)
