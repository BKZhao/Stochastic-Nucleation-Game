"""Angular response and the spatial construction of interface allocations."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import pubstyle as ps
from nucleation_interface import resource_substep
from nucleation_rearrange import effective_weights, rearranged_resource
from nucleation_paper_inputs import OUTPUT, REPORTS, load_report, record_figure
from research_common import PROJECT_ROOT, file_hash


def main() -> None:
    ps.apply_pre()
    direction = load_report('direction')['direction']
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE, 2.6), constrained_layout=True)
    rotations = [[(2, 1), (-1, 2), (-2, -1), (1, -2)],
                 [(5, -17), (17, 5), (-5, 17), (-17, -5)]]
    for ax, vectors, title, color, letter in zip(
            axes, rotations, [r'$|\mathbf{d}|=\sqrt{5}$', r'$|\mathbf{d}|=\sqrt{314}$'],
            [ps.AMBER, ps.BLUE], 'ab'):
        rows = [direction['arms'][f'shifted{vector}'] for vector in vectors]
        angle = np.arange(4)*np.pi/2
        mean = np.array([row['mean_delta'] for row in rows])
        ax.plot(angle, mean, color=color, marker='o', ms=4, lw=1.1)
        ax.axhline(0, color=ps.MUTED, ls=':', lw=.8)
        ax.set(xticks=angle, xticklabels=['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$'],
               xlabel=r'rotation $\theta$', ylabel=r'$\Delta q_B$', title=title,
               xlim=(-.2, 3*np.pi/2+.2), ylim=(-.18, .18))
        ps.clean_axes(ax)
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'figS05_translation_directions', OUTPUT)
    record_figure('figS05_translation_directions', __file__, [REPORTS['direction']],
                  'Four sampled rotations of each reference vector, 48 configurations, '
                  '16 continuations per treatment; point estimates. Connecting segments '
                  'are guides between measurements, not interpolation evidence or proof of isotropy.')

    mechanism = load_report('mechanism')
    model = mechanism['identity']['model']
    path = PROJECT_ROOT/sorted(direction['mothers'], key=lambda row: row['mother_path'])[0]['mother_path']
    with np.load(path, allow_pickle=False) as state:
        strategy = state['strategy']
        resource = state['resource']
    size = model['size']
    post = resource_substep(strategy, resource, size, model['growth'],
                            model['depletion'], model['diffusion'])
    weights = effective_weights(strategy, post, model)
    changes = [rearranged_resource(strategy, resource, model, arm, file_hash(path)).reshape(size, size)
               - resource.reshape(size, size) for arm in ['interface_protect', 'interface_harm']]
    if any(abs(change.sum()) > 1e-12 for change in changes):
        raise ValueError('ordered allocation must preserve the total resource')
    center = np.unravel_index(np.argmax(weights), weights.shape)
    rows = (np.arange(24)+center[0]-12) % size
    columns = (np.arange(24)+center[1]-12) % size
    window = np.ix_(rows, columns)
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_DOUBLE, 2.8), constrained_layout=True)
    im = axes[0].imshow(weights[window], origin='lower', cmap=ps.SEQ,
                        vmin=0, vmax=weights.max(), interpolation='nearest', rasterized=True)
    bar = fig.colorbar(im, ax=axes[0], fraction=.046, pad=.02)
    bar.set_label(r'$W_i$')
    axes[0].set_title('effective sensitivity')
    limit = max(np.abs(change).max() for change in changes)
    for ax, change, title in zip(axes[1:], changes, ['low-bias allocation', 'high-bias allocation']):
        im = ax.imshow(change[window], origin='lower', cmap=ps.DIV_A, vmin=-limit, vmax=limit,
                       interpolation='nearest', rasterized=True)
        ax.set_title(title)
    bar = fig.colorbar(im, ax=axes[1:], fraction=.022, pad=.02)
    bar.set_label(r'$\Delta r_i$')
    for ax, letter in zip(axes, 'abc'):
        contour = ax.contour(strategy.reshape(size, size)[window], levels=[.5],
                             colors=[ps.INK], linewidths=.5)
        contour.set_path_effects([pe.Stroke(linewidth=1.0, foreground='white'), pe.Normal()])
        ax.set(xticks=[], yticks=[])
        ax.plot([2, 8], [2, 2], color=ps.INK, lw=1.4)
        ps.heatmap_frame(ax)
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'fig04_mechanism', OUTPUT)
    record_figure('fig04_mechanism', __file__,
                  [REPORTS['mechanism'], REPORTS['direction'], path,
                   PROJECT_ROOT/'code/nucleation_rearrange.py',
                   PROJECT_ROOT/'code/nucleation_interface.py'],
                  'Outcome-blind spatial example, same configuration as the alignment figure. '
                  'Effective diffusion-mixed sensitivity and resource increments from the two exact '
                  'existing allocation rules; no new stochastic simulation. Strategies and boundary '
                  'resource multiset remain fixed. Shared signed scale for increments; 24-by-24 periodic window at maximum sensitivity, 6-site bars.')


if __name__ == '__main__':
    main()
