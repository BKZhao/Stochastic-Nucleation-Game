"""Descriptive background and risk panels from checksummed follow-up diagnostics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_diagnostics(report: dict, output: Path) -> None:
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for group,color in zip(report['background'],['#0072B2','#E69F00','#009E73']):
        rows=[row for row in group['landmarks'] if row['cooperation'] is not None]
        time=np.array([row['landmark'] for row in rows])
        rho=np.array([row['cooperation']['median'] for row in rows])
        low=np.array([row['cooperation']['q25'] for row in rows])
        high=np.array([row['cooperation']['q75'] for row in rows])
        label=f"b={group['cell']['parameters']['temptation']}"
        axes[0,0].plot(time,rho,'o-',color=color,label=label)
        axes[0,0].fill_between(time,low,high,color=color,alpha=.12)
        for t,value,row in zip(time,rho,rows):
            axes[0,0].annotate(str(row['at_risk']),(t,value),xytext=(0,5),textcoords='offset points',fontsize=7,color=color)
        km=group['cell']['km']
        axes[0,1].step([row['time'] for row in km],[row['survival'] for row in km],where='post',color=color,label=label)
    axes[0,0].set(xlabel='Sweep',ylabel='Cooperation among survivors',ylim=(0,1.05),
                  title='N01: median / IQR; labels = at-risk n')
    axes[0,1].set(xlabel='Confirmation time (sweeps)',ylabel='Survival probability',ylim=(0,1.05),
                  title='N01: unconditional Kaplan–Meier')
    for axis in axes[0]:
        axis.legend(fontsize=8)
    for axis,diffusion in zip(axes[1],[.35,.40]):
        for shape,color,marker,offset in zip(['compact','stripe','four_patches','random'],
                                            ['#0072B2','#E69F00','#009E73','#CC79A7'],['o','s','^','D'],[-.45,-.15,.15,.45]):
            cells=sorted([cell for cell in report['seeds']['cells'] if cell['parameters']['diffusion']==diffusion and cell['damage']['shape']==shape],key=lambda cell:cell['damage']['count'])
            x=np.array([cell['damage']['count'] for cell in cells])
            y=np.array([cell['collapse_risk'] for cell in cells])
            low=np.array([cell['wilson95'][0] for cell in cells])
            high=np.array([cell['wilson95'][1] for cell in cells])
            axis.errorbar(x+offset,y,yerr=np.maximum(0,[y-low,high-y]),fmt=marker+'-',
                          color=color,label=shape.replace('_',' '),capsize=2,ms=4)
        axis.set(xlabel='Damaged nodes m (small display offsets)',ylabel='Confirmed depletion fraction',
                 ylim=(-.04,1.04),title=f'N02: b=1.16, diffusion={diffusion}; Wilson 95%')
        axis.legend(fontsize=8,loc='best')
    fig.suptitle('Exploratory follow-up: survivor composition can change with time',fontsize=11)
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:
        fig.savefig(output/f'followup.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics',type=Path)
    parser.add_argument('--output',type=Path,default=Path('figures/nucleation/followup'))
    args=parser.parse_args()
    plot_diagnostics(json.loads(args.diagnostics.read_text()),args.output)
