"""Separate geometries in long-time outcomes and unsmoothed interface histories."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, record_figure

STYLES=[('compact_1',r'local ($m=1$)'),('compact_16',r'compact ($m=16$)'),
        ('random_16',r'dispersed ($m=16$)')]



def growth_kinetics(report: dict, analysis: Path) -> None:
    """Resolve long-time fates and early interfacial motion without merging their clocks."""
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 5.05), constrained_layout=True)
    geometries = [('compact_16', 'compact', ps.BLUE, '-'),
                  ('random_16', 'dispersed', ps.RED, '--')]
    for ax, (key, label, _, _) in zip(axes[0], geometries):
        row = report['groups'][key]['kinetics']
        time = np.asarray(row['time'])/1000
        for outcome, color, style, title in [('A', ps.BLUE, '-', 'recovery'),
                    ('B', ps.RED, '--', 'collapse'), ('U', ps.AMBER, ':', 'unresolved')]:
            ax.step(time, row['probability'][outcome], where='post', color=color,
                    ls=style, lw=1.5, label=title)
        ax.set(xlabel=r'$t$ ($10^3$ MCS)', ylabel=r'$P_\alpha(t)$',
               xlim=(0, 40), ylim=(-.04, 1.04), xticks=[0, 20, 40], title=label)
    axes[0, 0].legend(loc='center right', fontsize=9)
    for key, label, color, style in geometries:
        row = report['groups'][key]['early']
        time, values = np.asarray(row['time']), np.asarray(row['mean'])
        axes[1, 0].plot(time, values[:, 1]/4, color=color, ls=style, lw=1.1, label=label)
        bias = np.divide(4*values[:, 2], values[:, 1], out=np.zeros(len(time)), where=values[:, 1] > 0)
        if np.any(np.abs(bias) > 1+1e-12):
            raise ValueError('normalized interface bias outside [-1,1]')
        axes[1, 1].plot(time, bias, color=color, ls=style, lw=1.1, label=label)
    for ax, ylabel in zip(axes[1], [r'$\langle P\rangle/4$', r'$4\langle J_{\rm pre}\rangle/\langle P\rangle$']):
        ax.set(xlabel=r'$t$ (MCS)', ylabel=ylabel, xlim=(0, 500))
        ax.set_xscale('symlog', linthresh=1)
        ax.set_xticks([0, 1, 10, 100, 500], ['0', '1', '10', '100', '500'])
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].axhline(0, color=ps.MUTED, lw=.7, ls=':')
    axes[1, 0].legend(loc='upper left', fontsize=9)
    for ax, letter in zip(axes.flat, 'abcd'):
        ax.set_box_aspect(.86)
        ps.clean_axes(ax)
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'fig11_growth_kinetics', OUTPUT)
    record_figure('fig11_growth_kinetics', __file__, [analysis],
                  'Top: full-ensemble fate probabilities through 40000 MCS, event-time steps. '
                  'Bottom: all recorded times through 500 MCS, all 64 preparations per geometry, '
                  'mean interface activity P/4 and normalized bias 4*mean(Jpre)/mean(P). The latter is a ratio of ensemble means, '
                  'not the mean of individual ratios; zero by convention only when every interface is empty. Distinct archived '
                  'ensembles and time windows; no joined trajectories, interval artists or smoothing.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis',type=Path,required=True)
    args=parser.parse_args();report=json.loads(args.analysis.read_text())
    ps.apply_pre()
    growth_kinetics(report, args.analysis.resolve())
    fig,axes=plt.subplots(1,3,figsize=(6.2,2.4),sharey=True,constrained_layout=True)
    for ax,(key,title),letter in zip(axes,STYLES,'abc'):
        row=report['groups'][key]['kinetics'];t=np.array(row['time'])/1000
        for outcome,color,style,label in [('A',ps.BLUE,'-','recovery'),('B',ps.RED,'--','collapse'),
                                           ('U',ps.AMBER,':','unresolved')]:
            ax.step(t,row['probability'][outcome],where='post',color=color,ls=style,label=label,lw=1.4)
        ax.set(xlabel=r'$t$ ($10^3$ MCS)',title=title,xlim=(0,40),ylim=(-.04,1.04),xticks=[0,20,40])
        ps.clean_axes(ax);ps.panel_label(ax,letter,x=0,y=1.04)
    axes[0].set_ylabel(r'$P_\alpha(t)$')
    axes[0].legend(fontsize=9,loc='center right')
    ps.save(fig,'fig03_fate_kinetics',OUTPUT)
    record_figure('fig03_fate_kinetics',__file__,[args.analysis.resolve()],
                  'One initial geometry per panel, all 64 independent paths. A/B/U empirical curves '
                  'without interval shading; no smoothing and no survivor conditioning.')

    fig,axes=plt.subplots(3,3,figsize=(6.3,5.95),sharex=True,sharey='row',constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.065, w_pad=.045, hspace=.10, wspace=.025)
    for col,(key,title) in enumerate(STYLES):
        row=report['groups'][key]['early'];t=np.array(row['time'])
        mean=np.array(row['mean']);sem=np.array(row['sem'])
        for index,(column,scale,label) in enumerate([(0,1,r'$\langle n_D\rangle$'),
                                                     (2,1,r'$\langle J_{\rm pre}\rangle$'),
                                                     (1,.25,r'$\langle P\rangle/4$')]):
            ax=axes[index,col]
            ax.plot(t,scale*mean[:,column],color=ps.BLUE,lw=.85)
            ax.set(xlim=(0,500));ax.set_xscale('symlog',linthresh=1)
            ax.set_xticks([0,1,10,100,500],['0','1','10','100','500'])
            if col==0:ax.set_ylabel(label)
            if index==0:ax.set_title(title)
            if index==1:ax.axhline(0,color=ps.MUTED,lw=.6)
            if index==2:ax.set_xlabel(r'$t$ (MCS)')
            ps.clean_axes(ax);ps.panel_label(ax,'abcdefghi'[3*index+col],x=0,y=1.04)
    ps.save(fig,'fig04_interface_dynamics',OUTPUT)
    record_figure('fig04_interface_dynamics',__file__,[args.analysis.resolve()],
                  'All recorded MCS 0..500, one geometry and one observable per panel; 64 paths, '
                  'including all-C absorptions. Ensemble means, no smoothing or interval shading. Symlog time linear '
                  '0..1 then logarithmic. P/4 is frozen-visit activity, not a full-MCS variance.')


if __name__=='__main__':
    main()
