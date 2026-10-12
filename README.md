# Stochastic Nucleation Game

Code, frozen analyses, and manuscript sources for the paper
**"Stochastic collapse of spatial cooperation: Perturbation geometry and resource alignment"**
(Physical Review E submission).

The model is a stochastic spatial evolutionary game on a periodic square lattice,
coupled to a renewable, diffusing resource. Cooperators renew the resource they
harvest; defectors deplete it. Strategies spread by imitation (Fermi rule) and the
resource field evolves through local renewal/depletion plus nearest-neighbor
diffusion. The paper studies **finite-amplitude defector perturbations**:

1. **Perturbation geometry** — compact vs. dispersed perturbations with identical
   defector counts and resource deficits have different finite-time collapse
   probabilities, linked by exact first-visit identities to interfacial activity
   and signed update bias.
2. **Relative strategy–resource alignment** — translating the whole resource field
   on the periodic lattice preserves the strategy configuration and every
   single-field resource statistic, yet changes the probability of subsequent
   collapse. Exact resource-balance and sensitivity relations connect the effect
   to strategy–resource covariance and local payoff differences.

## Repository layout

| Path | Contents |
|---|---|
| `code/` | Model kernel and observables. `b_eco_unified.py` and `lattice.py` own the physics; `research_*.py` and `nucleation_*.py` implement events, state, seeding, interfaces, and statistics. |
| `experiments/` | Campaign runner (`nucleation_campaign.py`), per-question analyses (`analyze_*`), figure scripts (`plot_*`), and the manuscript pipeline (`build_nucleation_manuscript.py`, `refresh_nucleation_paper.py`). |
| `configs/nucleation/` | JSON campaign configurations used in the paper. |
| `data/nucleation/analysis/` | Frozen analysis outputs; every figure manifest records their SHA-256 hashes. |
| `data/nucleation/N01…N06/campaigns/` | Campaign summaries of the recorded runs. |
| `data/nucleation/archives/` | Checksum manifests of the archived raw trajectories (tarballs are distributed with Git LFS; see `docs/nucleation/trajectory_archive.md`). |
| `figures/nucleation/paper/` | Final figure assets with per-figure manifests (inputs, plotting sources, hashes). |
| `paper/` | REVTeX 4-2 sources of the article and the Supplemental Material, generated from `paper/drafts/` by `experiments/build_nucleation_manuscript.py`. |
| `tests/` | Unit tests for the model kernel, runners, and analyses. |
| `docs/` | Research design notes, data layout, and archiving protocol. |

## Requirements

Python ≥ 3.10 with `numpy`, `numba`, and `matplotlib` (see `requirements.txt`).
Building the PDFs additionally requires
[Tectonic](https://tectonic-typesetting.github.io) at `.tools/bin/tectonic`.

## Quickstart

```bash
python -m pip install -r requirements.txt

# run the test suite (model kernel, runners, analyses)
PYTHONPATH=code:experiments python -m unittest discover -s tests -v
```

Run one simulation campaign (compact 16-site perturbations at the reference
parameters; `--dry-run` prints the job grid without simulating):

```bash
PYTHONPATH=code:experiments python experiments/nucleation_campaign.py \
    --config configs/nucleation/n01.json --dry-run
```

Each campaign writes provenance, per-replica trajectories, and a summary under
`data/nucleation/<experiment-id>/campaigns/<id>/`. One job runs one continuously
seeded stream; replicas are independent processes.

## Reproducing the paper

All figures and both PDFs can be rebuilt from the frozen analyses and figure
assets included here, without the raw trajectories:

```bash
PYTHONPATH=code:experiments OPENBLAS_NUM_THREADS=1 \
    python experiments/refresh_nucleation_paper.py --reuse-figures
```

The mode hash-checks every figure asset, plotting source, and still-available
input, records missing raw inputs instead of revalidating them, recompiles
`paper/sections/` from `paper/drafts/`, and compiles `paper/main.pdf` and
`paper/supplement.pdf` (two alternating passes resolve cross-document
references). Regenerating figures from raw trajectories additionally requires
restoring the archives described in `docs/nucleation/trajectory_archive.md`.

## Data policy

- Frozen analysis JSONs and campaign summaries are versioned in full, so every
  number and figure in the paper is traceable to a hash-recorded input.
- Raw trajectory archives are not checked in; `data/nucleation/archives/` carries
  their SHA-256 manifests, and the tarballs are available via the companion
  research repository (Git LFS).

## Citation

If you use this code, please cite the paper and this repository:

> B. Zhao and X. Li, *Stochastic collapse of spatial cooperation: Perturbation
> geometry and resource alignment* (2026). Code: https://github.com/BKZhao/Stochastic-Nucleation-Game

## License

MIT — see [LICENSE](LICENSE).
