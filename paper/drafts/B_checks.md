<!-- Problem Alignment Notes: P1–P3. Definitions, sampling, controls, and reproducibility support the physical narrative. -->
# Numerical methods and supplementary controls
\label{sec:checks}

## Event clocks and continuation probabilities

Let $I_t=\mathbf 1\{\rho(t)\le0.2,\ \bar r(t)\le0.2\}$ and let $c_t$ count consecutive MCS inside this joint threshold:
$$
\label{eq:clock}
c_t=
\begin{cases}
c_{t-1}+1,& I_t=1,\\
0,& I_t=0.
\end{cases}
$$
For a newly perturbed trajectory $c_0=0$. Define $\tau_B=\inf\{t:c_t\ge250\}$ and $\tau_A=\inf\{t:\rho(t)=1,\ \bar r(t)\ge0.6\}$. The first event label is retained. Saved states include both fields, the counter, and any recorded label; the selected configurations have zero counters and no events.

The conditional fate probabilities are defined in main-text Eq.~\eqref{main-eq:conditional_fate}. Times are measured from the continuation origin, unlike $P_\alpha(T)$ and the reach-conditioned factor in Eq.~\eqref{main-eq:reach_decomposition}, which use a common absolute horizon.

For $M$ continuations, the estimates are $\widehat q_\alpha=M_\alpha/M$. If the remaining unresolved paths are eventually assigned either to depletion or to no depletion, their eventual event fraction lies in
$$
\label{eq:unresolved_bound}
\frac{M_B}{M}\le \widehat q_{B,\mathrm{eventual}}
\le\frac{M_B+M_U}{M}.
$$
This interval is a deterministic bound obtained by assigning every unresolved trajectory to one or the other eventual outcome. It is not a confidence interval, and it does not assume that the threshold-defined depletion event is absorbing. Subtracting opposite endpoints for two samples gives $[31/64,41/64]$ for dispersed minus compact in Fig.~\ref{fig:kinetics}.

Figure~\ref{fig:outcomes} gives the complementary recovery and unresolved maps; every cell satisfies Eq.~\eqref{main-eq:partition}. Figure~\ref{fig:sections} resolves all three probabilities at three diffusion strengths.

## Absorption and stopping

Without mutation, copying neighbors preserves the all-cooperator state. Diffusion conserves resources, while renewal gives
$$
\label{eq:absorbing}
\bar r^+-\bar r=g(1-\bar r)\ge0.
$$
Once $\bar r\ge0.6$, recovery persists and depletion is excluded. Stopping preserves the event label. Snapshot times record actual execution; the observation horizon remains $T$.

No corresponding proof makes the threshold-defined $B$ set absorbing. A separate persistence check continues 36 confirmed states, one per treatment from each of 12 distinct configurations under original, short-shift, and long-shift protocols. Four new 3000-MCS continuations are generated per state. None of the 144 continuations exceeds cooperation 0.2, and the maximum observed value is 0.1519. These continuations are nested within states and configurations and establish persistence only over that extra window.

## Finite-window drift and fluctuations

To follow the response beyond one visit, define the conditional moments from a configuration $X$,
$$
\label{eq:finite_moments}
\begin{aligned}
a_\Delta(X)&=\frac{\mathbb E[n_D(\Delta)-n_D(0)\mid X]}{\Delta},\\
b_\Delta(X)&=\frac{\operatorname{Var}[n_D(\Delta)-n_D(0)\mid X]}{\Delta}.
\end{aligned}
$$
These moments include all updates over $\Delta$ MCS, unlike Eq.~\eqref{main-eq:moments}. Their units are defectors per MCS and defectors squared per MCS. For $x=n_D/N$, they become $a_\Delta/N$ and $b_\Delta/N^2$. The variance rate $b_\Delta$ differs from the payoff parameter $b$.

Figure~\ref{fig:windows} compares original, fixed-strategy-adapted, and translated $(17,12)$ resource fields on the same strategy configurations. Adaptation generally changes the total resource, whereas translation preserves its distribution and total.

Original and adapted fields give similar finite-window moments. At $\Delta=1$, translation changes the mean drift from $-3.89$ to $-15.12$ for compact origins and from $-5.04$ to $-20.02$ for dispersed origins. This difference weakens but persists through $\Delta=20$. The conditional variance rate also decreases with the observation window, with weaker separation between treatments. These scale-dependent moments do not support a constant-coefficient drift--diffusion closure.

## Spatial selection and conditional-moment estimates

Figure~\ref{main-fig:fields} uses the smallest seed among confirmed-$B$ trajectories per ensemble, translated once using the initial perturbation center. These are successful examples; threshold statistics use every reaching trajectory.

For the finite-window response in Fig.~\ref{fig:windows}, each configuration contributes $M=64$ independent fresh continuations for each resource treatment. If $d_{\ell,\Delta}=n_D^{(\ell)}(\Delta)-n_D(0)$, the estimated moments are
$$
\label{eq:moment_estimators}
\begin{aligned}
\widehat a_\Delta&=\frac{1}{M\Delta}\sum_{\ell=1}^M d_{\ell,\Delta},\\
\widehat b_\Delta&=\frac{1}{(M-1)\Delta}
\sum_{\ell=1}^M(d_{\ell,\Delta}-\bar d_\Delta)^2.
\end{aligned}
$$
Variances are computed within configuration and treatment, then averaged; between-configuration variation is excluded.

The 30 first-5%-crossing configurations are all 15 available compact-origin states and the first 15 dispersed-origin seeds of an independent ensemble. Each contributes 64 fresh continuations per treatment at $\Delta=1,2,5,10,20$, with strategies initially fixed across treatments.

Pointwise uncertainty uses 10000 bootstrap resamples \cite{efron1979bootstrap} of whole configurations within each preparation, retaining treatments and windows together. No continuous-time or scaling limit is assumed.

## Longer-time resource controls

The sensitivity-allocation comparison uses a separate set of 48 compact-origin configurations at their first 5% crossing, selected by random-number-seed order, with 16 continuations per treatment to $H=40000$ MCS from the configuration. Boundary values are paired with effective sensitivity weights in same order, reverse order, or random order. An additional treatment permutes resource values among defectors. Interventions preserve the existing resource multiset on the chosen set. Complete configurations are resampled for paired 95% bootstrap intervals of $q_B(40000\mid X)$ and the reported correlations. For the one-MCS test, $a_1$ uses $n_D(1)-n_D(0)$, and the reported correlation uses intervention-minus-original changes in both $J_{\rm pre}$ and $a_1$.

With $\lambda=0$, resources leave the strategy payoff. Across all 48 configurations, matched random streams give identical strategy trajectories under the five resource treatments, providing a numerical decoupling check.

Translations are applied once to 48 grown configurations and followed to $H=40000$ MCS. Four directions are averaged within each magnitude and configuration. Table~\ref{main-tab:translation_risk} reports paired contrasts with 95% whole-configuration bootstrap intervals.

A separate fixed-direction ensemble retains a negative long-minus-short contrast at 20000, 40000, and 80000 MCS. Selecting 24 configurations at a fixed time gives $-0.1146$ $[-0.1719,-0.0547]$. With a fixed initial count at $L=96$, the interval approaches or includes zero; both intervals at the looser depletion threshold 0.25 include zero. At $L=96$ with fixed 5% fraction, or at $b=1.18$ or $D=0.6$, every continuation collapses. At $b=1.14$ or $D=0.2$, no eligible initial configurations are available. These controls delimit the regime in which the alignment contrast can be estimated. In particular, changing system size, the initial disturbance fraction, or the depletion threshold can weaken the measured difference or saturate the event probabilities. Saturated outcomes and missing eligible snapshots do not establish that the underlying intervention has zero effect; nor do the current experiments establish a universal system-size or threshold dependence.

## Parameter sampling and geometry controls

The parameter plane contains $\PhaseBPoints$ values of $b$ from 1.12 to 1.20 and $\PhaseDPoints$ values of $D$ from 0.20 to 0.60. Each point uses 64 independent realizations per geometry to $T=10000$ MCS. At the reference point, the dispersed-minus-compact collapse contrast is $0.609$ with 95% interval $[0.474,0.719]$. Independent confirmation with 256 realizations per geometry gives 149 dispersed and 3 compact events, a difference of $0.570$ $[0.505,0.630]$.

Separate ensembles of 64 realizations per preparation determine fate to $T=40000$ and early interface observables at every MCS through $t=500$. The long-time compact and dispersed ensembles contain 9 and 47 collapse events, and 7 and 3 unresolved paths, respectively. At the first 5% crossing, the dispersed largest-cluster share has interquartile range $[0.328,0.622]$. The 20% threshold is reached by 47 dispersed and 9 compact trajectories. Conditional on reaching both 5% and 20%, the median passage times between them are 3252 and 5358 MCS. Each threshold statistic uses all trajectories reaching that threshold; its denominator can therefore differ from those of later crossings.

A control that retains unit resource at the initially converted sites gives 32 dispersed and 1 compact collapse event, compared with 34 and 0 after the usual clearing. Each group contains 64 realizations and is observed to $T=10000$ MCS. Subsequent resource feedback is identical in the two preparations.

## Interface motifs and resource profiles

Figure~\ref{fig:initialinterfaces} uses the archived initial states of the early-interface ensembles. The example for each geometry is selected by lexical path order, independently of its outcome. Every undirected cooperator--defector edge contributes once to $H_{ac}$, and the displayed frequencies are $\sum H_{ac}/\sum P$ over the 64 states. The curves evaluate Eq.~\eqref{eq:motifs} at fixed preparations as $b$ or $D$ varies; they are analytic initial-state diagnostics. The motif expression is checked against direct edge summation at three parameter pairs spanning the displayed domain.

Figure~\ref{main-fig:resourcelayers} uses all 48 saved configurations of the four-orientation translation comparison, but displays only the vectors $(2,1)$ and $(5,-17)$. Coordinates follow the array convention $(d_y,d_x)$. For each site, let $d$ be the shortest periodic nearest-neighbor distance to a site of the opposite strategy. The profile coordinate is $\ell=d-1/2$ for cooperators and $\ell=-(d-1/2)$ for defectors. Profiles pool resource values over all sites in each shell; the payoff-gap cumulative distributions pool all undirected interfacial edges after the resource update. Their weights are therefore site and edge counts, respectively, rather than equal weights for each configuration. Full shell counts are retained in the analysis. The displayed distance range is $-4\le\ell\le12$; the spatial map uses the first path in lexical order without conditioning on fate. These profiles are deterministic evaluations of saved fields and do not add stochastic continuations.

The separate displacement-length series averages two orthogonal directions and uses its own original-field ensemble. Shorter lengths give positive responses; length 14 gives a negative one. The interval at 11 includes zero, whereas the paired 8-to-11 and 11-to-14 differences exclude zero. The four-rotation ensemble resolves no directional ordering within a magnitude.

## Numerical ensembles and uncertainty

The parameter grid contains \PhaseRuns{} realizations, each long-time and early-interface ensemble 192, and the finite-window comparison 5760 continuations. Each trajectory uses one continuous random-number stream; continuations start with fresh seeds.

Binomial probabilities are estimated from the full ensemble, retaining unresolved realizations in the denominator. In Fig.~\ref{main-fig:regimes}, an outcome is assigned a resolved majority only when its pointwise Wilson 95% lower bound \cite{wilson1927inference} exceeds $1/2$; otherwise the cell is left unclassified. Differences between independent probabilities use Newcombe intervals \cite{newcombe1998difference}, whereas resource-intervention contrasts retain pairing within the initial configurations. Numerical intervals are reported in the text and tables. The curves show measured probabilities, empirical distributions, or ensemble means without interval shading.

The archived growth trajectories include file checksums. Full public reproducibility will additionally require an immutable release of the newer parameter-grid and continuation data, together with the associated plotting and analysis sources.

## Time-resolved resource covariance

For the two short-time continuation ensembles, every MCS from $t=0$ to 20 is recorded. The resource balance determines the covariance at the start of each of the first 20 MCS:
$$
\label{eq:reconstruct_covariance}
C_{sr}(t)=
\frac{\bar r(t+1)-\bar r(t)-g\rho(t)+\delta\bar r(t)}
{\delta-g}
-\rho(t)\bar r(t).
$$
Here $\delta-g=0.4\ne0$. Reconstruction uses the pre-strategy resource update and retains each trajectory's product $\rho(t)\bar r(t)$ before averaging. Saved initial fields provide an independent check.

The translation and local-allocation ensembles contain 7680 and 3840 trajectories, recorded every MCS through $t=20$. Means first combine 16 continuations per configuration and treatment, then directions within each translation magnitude. Pointwise 95% intervals use 10000 bootstrap resamples of complete configurations, keeping all times and treatments together. Figures show means; probability contrasts appear in the text and intervals in the analysis.

For all 48 configurations, direct array evaluation verifies preservation of the translated resource mean and autocorrelation, as well as the balance relation in Eq.~\eqref{main-eq:translation_balance}.

Table~\ref{tab:local_risk} gives local-permutation outcomes from the long-time continuations; short-time growth is not used as a collapse-probability estimator.

\begin{table}[t]
\caption{Change in conditional collapse probability relative to the original field for local resource permutations at $H=40000$. The 95% intervals preserve pairing within the 48 initial configurations.}
\label{tab:local_risk}
\centering
\begin{tabular}{lrr}
\toprule
Allocation & $\Delta q_B$ & 95% interval \\
\midrule
Low bias & $0.0221$ & $[-0.0221,0.0651]$ \\
High bias & $0.0117$ & $[-0.0352,0.0560]$ \\
Boundary permutation & $0.0195$ & $[-0.0273,0.0664]$ \\
Defector permutation & $0.0065$ & $[-0.0391,0.0495]$ \\
\bottomrule
\end{tabular}
\end{table}

## Effect sizes accompanying the spatial comparisons

At the reference parameters, the original long-time preparations give collapse fractions $47/64$ for dispersed and $9/64$ for compact disturbances. Their Wilson intervals are $[0.615,0.827]$ and $[0.076,0.246]$. The initial interface ensembles have mean lengths $63.625$ and $20$ and mean biases $14.9744$ and $-2.7989$, respectively. The first complete-MCS population changes are $+6.67$ and $-3.27$.

At the first 5% crossing, dispersed and compact reaching ensembles have median largest-cluster shares $0.433$ and $0.979$, with median cluster counts 13 and 2. At 20%, the shares are $0.978$ and $0.998$. The observed establishment factors are $63/64$ and $22/64$, and the corresponding collapse-given-establishment fractions are $47/63$ and $9/22$. Conditional median passage times from 20% to 50% are 2518 and 2421 MCS.

The original and displayed long-shift boundary-cooperator shells in Fig.~\ref{main-fig:resourcelayers} have mean pre-update resources 0.572 and 0.977. For the local resource permutations, changes in the initial bias correlate with first-MCS increments with rank correlation 0.782 and paired interval $[0.724,0.835]$. Low-bias and high-bias allocations change the first increment by $-1.98$ and $+2.83$ defectors, and the accumulated response at 20 MCS by $+2.33$ and $+3.09$. The corresponding short- and long-translation accumulated responses are $+8.26$ and $-22.65$. The high-minus-low local-allocation collapse contrast is $-0.0104$ with interval $[-0.0521,0.0313]$; neither ordering nor equivalence is established.
