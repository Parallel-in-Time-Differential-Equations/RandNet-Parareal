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



def make_lightweight(path):
    try:
        s = read_pickle(path)
        run =  s.runs[list(s.runs.keys())[0]]
        keys = ['u', 'x', 'D']
        for k in keys:
            if k in run.keys():
                run.pop(k)
        store_pickle(s, path)
    except Exception  as e:
        print(e)

for dx in [0.1, 0.07, 0.05, 0.038]:
    N = 235
        
    make_lightweight(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"para"}'))
    make_lightweight(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"nngp"}'))
    make_lightweight(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"elm"}'))




def get(path):
    try:
        s = read_pickle(path)
        return s.runs[list(s.runs.keys())[0]]
    except Exception  as e:
        # print(e)
        return None
 
ds = []
for dx in [0.1, 0.07, 0.05, 0.038]:
    N = 235
        
    solver = read_pickle(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"elm"}'))
    solver.runs = {}
    ds.append(np.prod(solver.n))
    
    solver.runs['Parareal'] = get(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"para"}'))
    solver.runs['NN-GParareal'] = get(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"nngp"}'))
    solver.runs['ELM'] = get(os.path.join('SWEScal', f'{"SWEScal"}_{dx}_{N}_{"elm"}'))

    for k in list(solver.runs.keys()):
        if solver.runs[k] is None:
            solver.runs.pop(k)
        
    exp_serial_c = solver.N * np.array([v['timings']['F_time']/v['k'] for k,v in solver.runs.items()]).mean()

    print()
    print()
    Parareal.print_speedup(solver, md=False, fine_t = exp_serial_c, mdl_title=f'Shallow Water Equation $d={dx}$', readable=True)

[3*d**2 for d in [0.1, 0.07, 0.05, 0.038]]


## Plotting system evolution
# The final solution of shape (Time, dx, dy, dims)=(236, 264, 133, 3) is too heavy to store fully. We store only a few times. The full vector is available, just ask the lead author.

elm = get(os.path.join('SWEScal', f'{"SWEScal"}_{0.038}_{235}_{"elm"}'))


t = elm['t']
idxs = [0, 5, 10, 15, 20, 30, 50,100,150, 235]

# u= elm['u']
# np.savetxt(os.path.join('SWEScal','SWE_states'),elm['u'][idxs,...,0].reshape(-1,order='F'))

u = np.loadtxt(os.path.join('SWEScal','SWE_states')).reshape(10, 264, 133, order='F')


fig, axs = plt.subplots(2,5, figsize=(10,4))
for i, idx in enumerate(idxs[:5]):
    ax = axs.flatten()[i]
    # cax = ax.imshow((u[idx,...,0]), aspect=0.5, extent=(-5,5,0,5), vmin=0, vmax=0.6)
    cax = ax.imshow((u[i,...]), aspect=2, extent=(-5,5,0,5), vmin=0, vmax=0.6, cmap='pink')
    ax.set_title(f'Time {t[idx]:0.2f}')
    if i%5 > 0:
        ax.get_yaxis().set_visible(False)
    if i < 5:
        ax.get_xaxis().set_visible(False)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.58, 0.025, 0.31])
fig.colorbar(cax, cax=cbar_ax)

for i, idx in enumerate(idxs[5:]):
    i = i+5
    ax = axs.flatten()[i]
    # cax = ax.imshow((u[idx,...,0]), aspect=0.5, extent=(-5,5,0,5), vmin=0, vmax=0.1, cmap='copper')
    cax = ax.imshow((u[i,...]), aspect=2, extent=(-5,5,0,5), vmin=0, vmax=0.1, cmap='copper')
    ax.set_title(f'Time {t[idx]:0.2f}')
    if i%5 > 0:
        ax.get_yaxis().set_visible(False)
    if i < 5:
        ax.get_xaxis().set_visible(False)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.15, 0.025, 0.31])
fig.colorbar(cax, cax=cbar_ax)
store_fig(fig, 'swe_system_evolution')