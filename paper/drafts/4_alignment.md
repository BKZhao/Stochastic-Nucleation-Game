# Resource layers and relative field alignment
\label{sec:alignment}

As a disturbance expands, defectors deplete resources in its interior while cooperators maintain higher resource levels in the surrounding region. Their competition therefore generates a resource gradient across the strategy boundary. To isolate the role of this relative arrangement, we hold the strategy configuration fixed and translate the entire resource field on the periodic lattice,
$$
\label{eq:translation}
(\mathcal T_{\mathbf d}r)_i=r_{i-\mathbf d}.
$$
This operation preserves the total resource, its empirical distribution, and its autocorrelation at every displacement,
$$
\label{eq:autocorrelation}
\Gamma_r(\mathbf a)=\frac1N\sum_i(r_i-\bar r)(r_{i+\mathbf a}-\bar r),
$$
but generally changes the cross-covariance between the strategy and resource fields,
$$
\label{eq:crosscovariance}
C_{sr}(\mathbf d)=\frac1N\sum_i(s_i-\rho)(r_{i-\mathbf d}-\bar r).
$$
Figure~\ref{fig:alignment} illustrates this controlled comparison. The translated field has exactly the same internal resource structure as the original, but its depleted and resource-rich regions are differently positioned relative to the unchanged strategy interface.

[FIGURE:alignment]

For a grown configuration $X$ in which neither event has occurred, let $\tau_A$ and $\tau_B$ be the first recovery and collapse confirmation times measured from the continuation origin. The conditional fate probabilities at horizon $H$ are
$$
\label{eq:conditional_fate}
\begin{aligned}
q_B(H\mid X)&=\Pr(\tau_B\le H,\ \tau_B<\tau_A\mid X),\\
q_A(H\mid X)&=\Pr(\tau_A\le H,\ \tau_A<\tau_B\mid X),\\
q_U(H\mid X)&=1-q_A(H\mid X)-q_B(H\mid X).
\end{aligned}
$$
We compare these probabilities after a single resource translation of each grown configuration. Both fields then evolve under the original dynamics, retaining the same strategy state at the start of each continuation.

\begin{table}[t]
\caption{Translation-induced changes in conditional collapse probability at $H=40000$ MCS. The comparison uses 48 configurations at their first 5% defector crossing, with 16 fresh continuations per field and direction. Four directions are averaged within each magnitude; 95% bootstrap intervals retain pairing within configurations.}
\label{tab:translation_risk}
\centering
\begin{tabular}{lrr}
\toprule
Comparison & $\Delta q_B$ & 95% interval \\
\midrule
Short shift--original & $0.0498$ & $[0.0052,0.0944]$ \\
Long shift--original & $-0.0628$ & $[-0.0983,-0.0273]$ \\
Long--short shift & $-0.1126$ & $[-0.1370,-0.0882]$ \\
\bottomrule
\end{tabular}
\end{table}

Despite these invariants, the translated configurations have different finite-time collapse probabilities. A short translation increases the conditional probability of collapse relative to the original field, whereas a longer translation decreases it. In the paired continuation experiment at $H=40\,000$ MCS, the changes are $+0.0498$ and $-0.0628$, respectively (Table~\ref{tab:translation_risk}). An independent displacement-length series also shows a turnover rather than a monotonic dependence on translation magnitude. Thus the relative arrangement of the two fields affects the outcome even when the strategy pattern and the single-field resource statistics are held fixed. Displacement vectors, sampling details, and uncertainty estimates are given in Sec.~\ref{sm-sec:checks} of the Supplemental Material.

The interfacial resource profiles in Fig.~\ref{fig:resourcelayers} provide a local view of this effect. Distance is measured relative to the strategy boundary, with negative values inside defector regions and positive values inside cooperative regions. Before translation, the mean resource profile rises sharply across the interface. A short shift broadens this transition, whereas the displayed long shift moves resource-rich regions toward parts of the boundary and raises resource levels at neighboring cooperative sites.

[FIGURE:resourcelayers]

Renewal, depletion, and diffusion subsequently act on the translated field, but a single update does not remove the change in boundary conditions. For the displayed long shift, the interfacial payoff-gap distribution shifts toward values favoring cooperation. Relative field alignment therefore changes the immediate microscopic bias of interface motion. Its effect on collapse probability also depends on how the interface and resource layer evolve after this initial perturbation.

Alignment also affects net resource production through the strategy--resource covariance. Subtracting Eq.~\eqref{eq:balance} for original and translated fields with equal initial strategy and resource means gives
$$
\label{eq:translation_balance}
\Delta\bar r(1)=(\delta-g)\Delta C_{sr}(0).
$$
Equation~\eqref{eq:translation_balance} shows that changing relative alignment alters the next mean-resource increment even though the initial resource total is unchanged. Interfacial payoff differences and total resource production thus respond to the same spatial rearrangement through distinct local and global mechanisms. Neither response can be inferred from the initial mean resource alone.
