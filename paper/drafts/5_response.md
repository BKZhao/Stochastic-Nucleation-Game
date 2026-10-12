# Transient response and resource memory

## Adaptation to a moving strategy pattern

The resource field contains information about the recent spatial arrangement of strategies. Let $M_D$ represent nearest-neighbor diffusion and $A_s$ be diagonal with entries $a_i=1-gs_i-\delta(1-s_i)$. When the strategy configuration is held fixed, resource dynamics take the affine form
$$
\label{eq:affine_map}
T_s(r)=M_D(A_s r+g s).
$$
For $0<g,\delta\le1$, this map is a maximum-norm contraction under a fixed strategy pattern, with $q=\max\{1-g,1-\delta\}<1$:
$$
\label{eq:contraction}
\|T_s^t(r)-T_s^t(\widetilde r)\|_\infty
\le q^t\|r-\widetilde r\|_\infty.
$$
It therefore has a unique equilibrium resource field $r_{\rm eq}(s)$. The deviation from that equilibrium can be bounded by the one-step residual,
$$
\label{eq:equilibrium_error}
\|r-r_{\rm eq}\|_\infty
\le\frac{\|r-T_s(r)\|_\infty}{1-q}.
$$
as demonstrated numerically in Fig.~\ref{sm-fig:resourcecontrols}. This result concerns two resource fields evolving under the same fixed strategy pattern.

In the coupled game, however, strategy updates continually change the equilibrium resource field itself. To describe this mismatch, we measure the mean distance to the instantaneous fixed-strategy equilibrium and compare the actual interfacial bias with its adapted-field counterpart:
$$
\label{eq:adaptation}
\eta(s,r)=\frac1N\sum_i|r_i-r_{{\rm eq},i}(s)|,
\qquad J_{\rm eq}(s)=J(s,r_{\rm eq}(s)).
$$
The mismatch $\eta$ measures incomplete resource adaptation, while $J-J_{\rm eq}$ compares the current interfacial bias with that obtained from a resource field equilibrated to the same strategy configuration. These diagnostics distinguish relaxation toward a fixed target from evolution in which the target moves. Consequently, the contraction bound for fixed strategies does not define a relaxation time for the fully coupled dynamics.

## Spatial sensitivity and the loss of initial ordering

Different parts of the interface have different sensitivities to resource changes. On the positive reward branch, differentiating the signed bias with respect to the resource level of a boundary cooperator yields
$$
\label{eq:sensitivity}
\begin{aligned}
w_u=-\frac{\partial J}{\partial r_u}
&=\frac{\lambda\sigma k_C(u)}{4K}
 \sum_{v\in\mathcal N_D(u)}[1-\tanh^2 z_{uv}],\\
z_{uv}&=\frac{\Pi_D(v)-\Pi_C(u)}{2K}.
\end{aligned}
$$
Only boundary cooperators contribute directly to this derivative. Resource perturbations introduced before the resource substep are additionally redistributed by local reaction and diffusion, giving the effective first-order response
$$
\label{eq:effective_weight}
\begin{aligned}
\delta J_{\rm pre}&=-\sum_u a_u W_u h_u+O(\|h\|^2),\\
W_u&=(1-D)w_u+\frac D4\sum_{v\sim u}w_v,
\end{aligned}
$$
The sensitivity weights are evaluated at the unperturbed post-update field. The remainder is quadratic for a fixed lattice and parameter set, provided the perturbation remains on the positive reward branch. For boundary cooperators, the local reaction factor is $a_u=1-g$.

Figure~\ref{fig:calibration} visualizes these spatially heterogeneous sensitivities. Assigning the largest existing boundary resources to the largest effective weights minimizes the linearized signed growth bias; reversing the assignment maximizes it. Both rearrangements preserve the multiset of boundary resource values, total resource, and initial strategy--resource covariance. Because the rearrangement argument concerns a linearized objective, the finite changes in $J_{\rm pre}$ are evaluated using the full nonlinear expression rather than assumed to attain the same extrema.

[FIGURE:calibration]

The two arrangements produce contrasting initial responses, but this ordering does not determine later population trajectories. To separate immediate drift from accumulated changes, define $a_a(t)=\mathbb E_a[n_D(t+1)-n_D(t)]$, with $a=0$ denoting the original resource field. Because all treatments begin with the same strategy configuration, the cumulative difference satisfies
$$
\label{eq:cumulative_response}
\mathcal I_a(t)=
\mathbb E_a[n_D(t)]-\mathbb E_0[n_D(t)]
=\sum_{u=0}^{t-1}\bigl[a_a(u)-a_0(u)\bigr].
$$
Figure~\ref{sm-fig:response} shows that a resource assignment that initially suppresses defector growth can produce a positive accumulated population difference at later measured times. Thus a first-step response does not establish a persistent ordering of population changes. The longer-time continuation experiment likewise does not statistically resolve an ordering of collapse probabilities between the two local assignments; this should not be interpreted as evidence that their outcomes are identical.

Resource translations and boundary permutations probe different features of spatial organization. Because their initial signed biases are not matched, they do not constitute equal-strength interventions. The comparisons establish differences under the interventions actually applied, not a universal ranking of their persistence. Numerical results and uncertainty estimates appear in Sec.~\ref{sm-sec:checks} of the Supplemental Material.

## Coupled evolution of the two fields

The exact resource balance helps distinguish the two interventions. Translating the resource field immediately changes the strategy--resource covariance, which can alter the next mean-resource increment through Eq.~\eqref{eq:translation_balance}. In contrast, permuting resource values among boundary cooperators preserves the strategy--resource covariance, mean resource, and mean strategy. It therefore leaves the next total-resource increment unchanged while potentially redistributing payoff bias along the interface. Subsequent strategy changes create new covariance differences and corresponding changes in resource production.

Figure~\ref{sm-fig:feedback} displays these distinct transient responses through mean population--resource paths and reconstructed covariance changes. Translation changes how the resource landscape is positioned relative to the strategies. Boundary permutation instead changes where resources act along an unchanged interface while preserving the global initial covariance. Their subsequent trajectories reflect the coupled evolution of strategies and resources. The plotted ensemble means are descriptive and do not define a closed deterministic flow in $(\rho,\bar r)$.

The covariance term in the exact balance applies separately to each trajectory before ensemble averaging. Together with the spatially varying interface sensitivity, it explains why mean population and resource alone omit information relevant to the disturbance response. Sec.~\ref{sm-sec:proofs} of the Supplemental Material gives the derivations.
