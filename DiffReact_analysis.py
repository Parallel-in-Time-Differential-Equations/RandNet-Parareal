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

for dx in [19, 28, 41, 77, 113, 164, 235]:
    if dx == 19:
        N = 64
    elif dx == 28:
        N = 128
    elif dx == 41:
        N = 256
    else:
        N = 512
        
    make_lightweight(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"para"}'))
    make_lightweight(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"nngp"}'))
    make_lightweight(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"elm"}'))


def get(path):
    try:
        s = read_pickle(path)
        return s.runs[list(s.runs.keys())[0]]
    except Exception  as e:
        # print(e)
        return None
fine_time = {}
for dx in [19, 28, 41, 77, 113, 164, 235]:
    if dx == 19:
        N = 64
    elif dx == 28:
        N = 128
    elif dx == 41:
        N = 256
    else:
        N = 512
        
    solver = read_pickle(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"elm"}'))
    solver.runs = {}
    
    solver.runs['Parareal'] = get(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"para"}'))
    solver.runs['NN-GParareal'] = get(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"nngp"}'))
    solver.runs['ELM'] = get(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{"elm"}'))

    for k in list(solver.runs.keys()):
        if solver.runs[k] is None:
            solver.runs.pop(k)
        
    exp_serial_c = solver.N * np.array([v['timings']['F_time']/v['k'] for k,v in solver.runs.items()]).mean()
    fine_time[dx] = exp_serial_c
    print()
    print()
    Parareal.print_speedup(solver, md=False, fine_t = exp_serial_c, mdl_title=f'DiffReact $d={dx**2*2}$', readable=True)



### Plots Speedup 


def tool_append(d, k, val):
    l = d.get(k, list())
    l.append(val)
    d[k] = l
    
def tr_key(key):
    mdl, typ = key.split('_')
    mdl_d = {'gp':'GPara','para':'Para','nngp':'NN-GPara', 'elm':'ELMPara'}
    typ_d = {'exp':'Theoretical', 'act':'Actual','exprough':'Theoretical approx', 'ub': 'Upper bound'}
    return f'{mdl_d[mdl]} {typ_d[typ]}'
    
n_restarts = 1
n_cores_d = {32:47, 64:47*2, 128:47*3, 256:47*6, 512:47*11}
store = {}
store_time = {}
dxs = [19, 28, 41, 77, 113, 164, 235]
Ns = []
for dx in dxs:
    if dx == 19:
        N = 64
    elif dx == 28:
        N = 128
    elif dx == 41:
        N = 256
    else:
        N = 512
    Ns.append((int(N//47)+1)*47)
    for mdl in ['para', 'elm', 'nngp']:
        try:
            solver = read_pickle(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{dx}_{N}_{mdl}'))

            run = solver.runs[list(solver.runs.keys())[0]]
            act = calc_speedup(run, N=N)
            tool_append(store, mdl+'_act', act)
            tool_append(store_time, mdl+'_act', run['timings']['runtime'])
        except Exception as e:
            tool_append(store, mdl+'_act', np.nan)
            tool_append(store_time, mdl+'_act', np.nan)
    tool_append(store_time, 'fine'+'_act', fine_time[dx])    



### Scatter plot
fontsize = 15
fs_ticks = 13
ds = np.array([2*d**2 for d in dxs])
c = {'elm':'red','para':'gray','nngp':'blue', 'fine':'black'}    
legend_tags = {'elm':'ELM', 'para':'Parareal', 'nngp':'NN-GParareal', 'fine':'Fine solver'}
markers = {'elm':'2', 'para':'x', 'nngp':'+', 'fine':'_'}
fig, axs = plt.subplots(1,2,figsize = [6.4*2, 4.8])
ax=axs[0]
# fig, ax = plt.subplots()
for key, val in store.items():
    mdl, _ = key.split('_')
    mask = np.arange(7)
    x = np.log10(ds[mask])
    y = np.array(val)[mask]  
    if mdl == 'nngp':
        ax.scatter(x, y, label=legend_tags[mdl], marker=markers[mdl], facecolors='blue', edgecolor='blue', s=300)
        # facecolor='none'
    else:
        ax.scatter(x, y, label=legend_tags[mdl], c=c[mdl], marker=markers[mdl], s=300)

    ax.set_xticks(x)
ax.scatter(x, np.array([1]*len(x)),  c='black',marker='_', label='Fine solver',s=300)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ax.get_xticks())
ax2.set_xticklabels(np.array(Ns)[mask])
ax.tick_params(axis='both', labelsize=fs_ticks)
ax2.tick_params(axis='both', labelsize=fs_ticks)
ax2.set_xlabel(r"Number of processors $N$", fontsize=fontsize)
ax.legend(prop={'size':fontsize}, loc='upper left', frameon=False, fontsize=fontsize)
ax.set_xlabel(r'$\log_{10}(d)$',fontsize=fontsize)
ax.set_ylabel('Speed-up',fontsize=fontsize)
fig.tight_layout()

ax = axs[1]
for key, val in store_time.items():
    mdl, _ = key.split('_')
    mask = np.arange(7)
    x = np.log10(ds[mask])
    y = np.array(val)[mask]  
    y = np.log10(y/3600)
    if mdl == 'nngp':
        ax.scatter(x, y, label=legend_tags[mdl], marker=markers[mdl], facecolors='blue', edgecolor='blue', s=300)
        # facecolor='none'
    else:
        ax.scatter(x, y, label=legend_tags[mdl], c=c[mdl], marker=markers[mdl], s=300)

    ax.set_xticks(x)
# ax.scatter(x, np.array([1]*len(x)),  c='black',marker='_', label='Fine solver',s=300)
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks(ax.get_xticks())
ax2.set_xticklabels(np.array(Ns)[mask])
ax.tick_params(axis='both', labelsize=fs_ticks)
ax2.tick_params(axis='both', labelsize=fs_ticks)
ax2.set_xlabel(r"Number of processors $N$", fontsize=fontsize)
ax.legend(prop={'size':fontsize}, loc='upper left', frameon=False, fontsize=fontsize)
ax.set_xlabel(r'$\log_{10}(d)$',fontsize=fontsize)
ax.set_ylabel('Runtime (hours, $\log$)',fontsize=fontsize)
fig.tight_layout()
store_fig(fig, 'diffreact_speedup_w_time')


## Plotting system evolution
# The final solution of shape (Time, dx, dy, dims)=(236, 264, 133, 3) is too heavy to store fully. We store only a few times. The full vector is available, just ask the lead author.

    
elm = get(os.path.join('DiffReactScal', f'{"DiffReactScal"}_{235}_{512}_{"elm"}'))

t = elm['t']
idxs = [0, 20, 40, 80, 150, 225, 300,375,450, 512]

# u= elm['u']

# u_bis = np.zeros((513, 235, 235, 2))
# for i in range(513):
#     u_bis[i,...,0] = u[i, :235**2].reshape(235, 235)
#     u_bis[i,...,1] = u[i, 235**2:].reshape(235, 235)


# u = u_bis[idxs,...,0]

# np.savetxt(os.path.join('DiffReactScal','DiffReact_states'),u.reshape(-1,order='F'))

u = np.loadtxt(os.path.join('DiffReactScal','DiffReact_states')).reshape(10, 235, 235, order='F')


fig, axs = plt.subplots(2,5, figsize=(10,4))
for i, idx in enumerate(idxs):
    ax = axs.flatten()[i]
    # cax = ax.imshow((u[idx,...,0]), aspect=0.5, extent=(-5,5,0,5), vmin=0, vmax=0.6)
    cax = ax.imshow((u[i,...]), extent=(-1,1,-1,1), vmin=-0.15, vmax=0.25, cmap='copper')
    ax.set_title(f'Time {t[idx]:0.2f}')
    if i%5 > 0:
        ax.get_yaxis().set_visible(False)
    if i < 5:
        ax.get_xaxis().set_visible(False)
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.25, 0.025, 0.5])
fig.colorbar(cax, cax=cbar_ax)
store_fig(fig, 'diffreact_system_evolution')