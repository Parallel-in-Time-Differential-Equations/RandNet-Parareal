import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
from speedup import calc_speedup

def read_pickle(name):
    with open(os.path.join(name), 'rb') as f:
        data = pickle.load(f)
    return data

def store_pickle(obj, name):
    with open(os.path.join(name), 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


from parareal import Parareal

def store_fig(fig, name):
    fig.savefig(os.path.join('img', name+'.pdf'), bbox_inches='tight')


def get(path):
    try:
        s = read_pickle(path)
        return s.runs[list(s.runs.keys())[0]]
    except Exception  as e:
        # print(e)
        return None
 
for dx in [128, 1128]:
    N = 128
        
    solver = read_pickle(os.path.join('BurgersScal', f'{"BurgersScal"}_{dx}_{N}_{"elm"}'))
    solver.runs = {}
    
    solver.runs['Parareal'] = get(os.path.join('BurgersScal', f'{"BurgersScal"}_{dx}_{N}_{"para"}'))
    solver.runs['NN-GParareal'] = get(os.path.join('BurgersScal', f'{"BurgersScal"}_{dx}_{N}_{"nngp"}'))
    solver.runs['ELM'] = get(os.path.join('BurgersScal', f'{"BurgersScal"}_{dx}_{N}_{"elm"}'))

    for k in list(solver.runs.keys()):
        if solver.runs[k] is None:
            solver.runs.pop(k)
        
    exp_serial_c = solver.N * np.array([v['timings']['F_time']/v['k'] for k,v in solver.runs.items()]).mean()

    print()
    print()
    Parareal.print_speedup(solver, md=False, fine_t = exp_serial_c, mdl_title=f'Viscous Burgers\' $d={dx}$', readable=True)

## Plotting system evolution
elm = get(os.path.join('BurgersScal', f'{"BurgersScal"}_{1128}_{128}_{"elm"}'))


fig, ax = plt.subplots()
cax = ax.imshow(elm['u'].T, aspect=2.5, extent=(0,5.9,-1,1))
cbar = fig.colorbar(cax, shrink=0.8)
ax.set_xlabel('Time')
ax.set_ylabel('x')
store_fig(fig, 'burgers_system_evolution')

