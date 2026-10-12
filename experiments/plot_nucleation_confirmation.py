"""Display the fixed N03 risk comparison and secondary confirmation survival."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_confirmation(report: dict, output: Path) -> None:
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.9),layout='constrained')
    for index,(shape,color) in enumerate([('compact','#0072B2'),('random','#E69F00')]):
        group=report['groups'][shape]
        risk=group['collapse_risk']
        low,high=group['wilson95']
        axes[0].errorbar(index,risk*100,yerr=np.array([[risk-low],[high-risk]])*100,
                         fmt='o',color=color,ms=8,capsize=6,lw=2)
        axes[0].text(index,high*100+3,f"{group['events']}/{group['n']}",ha='center',fontsize=11)
        km=group['km']
        axes[1].step([row['time'] for row in km],[row['survival'] for row in km],
                     where='post',color=color,label=shape)
    axes[0].set(xlim=(-.5,1.5),ylim=(0,100),xticks=[0,1],xticklabels=['Compact','Uniform random'],
                ylabel='Confirmed depletion by T (%)',title='Independent risk; Wilson 95% intervals')
    axes[1].set(xlabel='Confirmation time (sweeps)',ylabel='Survival probability',ylim=(0,1.03),
                title='Secondary: Kaplan–Meier confirmation survival')
    axes[1].legend(loc='lower left')
    group=report['groups']['compact']; model=group['parameters']
    delta=report['risk_difference']*100; low,high=np.array(report['newcombe95'])*100
    fig.suptitle(f"N03: L={group['L']}, b={model['temptation']}, diffusion={model['diffusion']}, mutation={model['mutation']:g}, m={group['damage']['count']}\n"
                 f"Random − compact = {delta:.1f} percentage points (95% CI {low:.1f} to {high:.1f}); Fisher p={report['fisher_two_sided_p']:.2e}",fontsize=10)
    output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:
        fig.savefig(output/f'confirmation.{suffix}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('confirmation',type=Path)
    parser.add_argument('--output',type=Path,default=Path('figures/nucleation/confirmation'))
    args=parser.parse_args()
    plot_confirmation(json.loads(args.confirmation.read_text()),args.output)
