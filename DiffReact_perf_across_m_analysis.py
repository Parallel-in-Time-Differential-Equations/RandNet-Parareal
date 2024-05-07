import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def store_fig(fig, name):
    fig.savefig(os.path.join('img', name), bbox_inches='tight')
    fig.savefig(os.path.join('img', name+'.pdf'), bbox_inches='tight')

l = pickle.load(open('DiffReact_perf_across_m.pkl', 'rb'))
records = np.array([i for i in l if len(i) == 5])

df = pd.DataFrame.from_records(l)
df.columns = ['seed', 'res_size', 'm', 'k', 't']

def get_bin_lab(bins, gp):
    out = [r'$K_{RW} '+fr' < {bins[0]}$']
    for i in range(1, len(bins)):
        # out.append(rf'${bins[i-1]} \leq' + ' K_{NN} ' + rf'< {bins[i]}$')
        out.append(r'$K_{RW} '+rf'={bins[i-1]}$' )
    out.append(r'$K_{RW} \geq K_{NN\text{-}GPara}' + rf' = {bins[-1]}$')
    return out

def build_bins(df, bins, bins_lab):
    bins.insert(0, df.k.min())
    bins.append(df.k.max()+1)
    
    out_df = []
    for idx, gr in df.groupby('m'):
        for i in range(len(bins)-1):
            mask = (gr.k>=bins[i]) & (gr.k < bins[i+1])
            bin_val = gr.loc[mask, 'count'].sum()/gr['count'].sum()
            out_df.append([idx, bins_lab[i], bin_val])
    out_df = pd.DataFrame.from_records(out_df)
    out_df.columns = ['m','bin','count']
    return out_df.set_index('m')

def do(df, ax, bins_k, show_legend=True, is_res_size = False):
    df['m'] = df['m'].astype(int)
    out = df.groupby(['m', 'k']).agg(count=('seed','count')).reset_index(level=1)
    out.pivot_table(values='count', index=out.index, columns='k', fill_value=0)
    # bins = bins_k
    bins_lab = get_bin_lab(bins_k, 0)
    out = build_bins(out.reset_index(), bins_k, bins_lab)
    out_pv = out.pivot_table(values='count', index=out.index, columns='bin', fill_value=0)
    out_pv.reindex(sorted(out_pv.columns, key=lambda x: bins_lab.index(x)), axis=1).plot.bar(stacked=True, ax=ax, label=2)
    # sum_filt = lambda k: k[k<= kgp].count()/k.count()
    # temp = gr.groupby(['nn']).agg(count_filt=('k',sum_filt)).reset_index()
    # ax.plot(temp.count_filt, label='$K_{GPara}='+f'{kgp}$', c='white')
    # ax.fill_between(np.arange(len(temp.count_filt)), [0]*len(temp.count_filt), temp.count_filt, hatch='/', alpha=0.0)
    ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    ax.get_legend().set_visible(show_legend)
    ax.set_xlabel('m')
    ax.set_ylabel('Proportion')
    # ax.set_title(f'$t_N={5.9}$')
    if is_res_size:
        ax.set_xticks(np.arange(0,49,3),(np.arange(0,49,3)*10+20).astype(int))
        ax.set_xlabel('M')

fig, axs = plt.subplots(1,2,figsize=(10,4))
df.columns = ['seed', 'res_size', 'm', 'k', 't']
do(df, axs[0], [9,10,11,12, 13, 14], show_legend=False)
df.columns = ['seed', 'm', 'mr', 'k', 't']
do(df, axs[1], [9,10,11,12, 13, 14], is_res_size=True)


store_fig(fig, f'DiffReact_perf_across_m_M')