import numpy as np

def calc_exp_gp_cost(run_obj, n_cores, n_jitter=9, *args, **kwargs):
    Tm = run_obj['timings']['avg_serial_train_time']
    d = run_obj['u'].shape[1]
    exp_train_time = np.sum(Tm * max(n_jitter * d / n_cores, 1))
    return run_obj['timings']['mdl_pred_t'] + exp_train_time

def get_act_mdl_cost(run_obj):
    return run_obj['timings']['mdl_tot_t']

def get_act_cost(run_obj):
    return run_obj['timings']['runtime']


def calc_exp_nngp_cost_rough(run_obj, n_cores, N, n_jitter=9, n_restarts=None, *args, **kwargs):
    k = run_obj['k']
    Tm = run_obj['timings']['avg_serial_train_time']
    if n_restarts is None:
        n_restarts = run_obj['mdl'].n_restarts
    d = run_obj['u'].shape[1]
    
    exp_c_rough = k * (Tm*max(((n_jitter*n_restarts*d)/n_cores),1)) * (N-(k+1)/2)
    return exp_c_rough
    
# This uses the average serial training time
def calc_exp_nngp_cost_precise_v1(run_obj, n_cores, N, n_jitter=9, n_restarts=None, *args, **kwargs):
    k = run_obj['k']
    Tm = run_obj['timings']['avg_serial_train_time']
    if n_restarts is None:
        n_restarts = run_obj['mdl'].n_restarts
    d = run_obj['u'].shape[1]
    
    conv_int = np.array([0]+run_obj['conv_int'][:-1])
    exp_cost_precise = ((N-conv_int)*(Tm*max(((n_jitter*n_restarts*d)/n_cores),1))).sum()
    return exp_cost_precise
    
    
def calc_exp_para_mdl_cost(run_obj, *args, **kwargs):
    return 0
    
    
def est_serial(run_obj, N):
    return run_obj['timings']['F_time_serial_avg']*N
    
def calc_speedup(run_obj, serial=None, N=None):
    if N is None:
        raise Exception('Cannot compute speedup without either N or serial.')
    else:
        serial = est_serial(run_obj, N)
    return serial/get_act_cost(run_obj)

def calc_exp_speedup(run_obj, mdl_cost_fn, *args, **kwargs):
    if 'N' not in kwargs:
        raise Exception('Cannot compute speedup without either N or serial.')
    else:
        serial = est_serial(run_obj, kwargs['N'])
    Tf = run_obj['timings']['F_time_serial_avg']*run_obj['k']
    Tg = run_obj['timings']['G_time']
    return serial/(Tf + Tg + mdl_cost_fn(run_obj, *args, **kwargs))