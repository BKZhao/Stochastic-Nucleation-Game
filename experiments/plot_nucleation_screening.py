"""Plot completed campaign summaries; all intervals describe screening samples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_screening(pilot: dict, seeds: dict, output: Path) -> None:
    fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
    for axis,mutation in zip(axes[:2],[.0001,.001]):
        cells=[cell for cell in pilot['cells'] if cell['parameters']['mutation']==mutation]
        temptations=sorted({cell['parameters']['temptation'] for cell in cells})
        diffusions=sorted({cell['parameters']['diffusion'] for cell in cells})
        matrix=np.full((len(diffusions),len(temptations)),np.nan)
        labels={}
        for cell in cells:
            row=diffusions.index(cell['parameters']['diffusion'])
            col=temptations.index(cell['parameters']['temptation'])
            matrix[row,col]=cell['collapse_risk']
            labels[row,col]=f"{cell['events']}/{cell['n']}"
        artist=axis.imshow(matrix,vmin=0,vmax=1,cmap='cividis',aspect='auto')
        for (row,col),label in labels.items():
            axis.text(col,row,label,ha='center',va='center',color='white' if matrix[row,col]<.5 else 'black')
        axis.set(xticks=range(len(temptations)),xticklabels=temptations,
                 yticks=range(len(diffusions)),yticklabels=diffusions,
                 xlabel='Temptation b',ylabel='Resource diffusion',title=f'N01: mutation = {mutation:g}')
    fig.colorbar(artist,ax=axes[:2],label='Confirmed depletion fraction',shrink=.8)
    for shape,marker,color in zip(['compact','stripe','four_patches','random'],['o','s','^','D'],
                                  ['#0072B2','#E69F00','#009E73','#CC79A7']):
        cells=sorted([cell for cell in seeds['cells'] if cell['damage']['shape']==shape],key=lambda cell:cell['damage']['count'])
        x=np.array([cell['damage']['count'] for cell in cells])
        y=np.array([cell['collapse_risk'] for cell in cells])
        low=np.array([cell['wilson95'][0] for cell in cells])
        high=np.array([cell['wilson95'][1] for cell in cells])
        offset={'compact':-2.4,'stripe':-.8,'four_patches':.8,'random':2.4}[shape]
        axes[2].errorbar(x+offset,y,yerr=np.maximum(0,np.array([y-low,high-y])),
                         fmt=marker+'-',color=color,label=shape.replace('_',' '),capsize=2,ms=4,lw=1)
    axes[2].set(xlabel='Damaged nodes m (small display offsets)',ylabel='Confirmed depletion fraction',
                title='N02: Wilson 95% intervals',ylim=(-.03,1.03))
    axes[2].legend(fontsize=8,loc='lower right')
    fig.suptitle('Exploratory screening; no metastability or critical-nucleus claim',fontsize=11)
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:
        fig.savefig(output/f'screening.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot',type=Path,required=True)
    parser.add_argument('--seeds',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('figures/nucleation'))
    args=parser.parse_args()
    plot_screening(json.loads(args.pilot.read_text()),json.loads(args.seeds.read_text()),args.output)
