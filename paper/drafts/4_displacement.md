# Interfacial bias and spatial growth

To connect initial geometry to collapse risk, we examine the motion and organization of cooperator--defector interfaces. Figure~\ref{fig:fields} compares illustrative trajectories at matched defector fractions, rather than at matched times. Compact seeds initially expand from a single patch, whereas dispersed seeds establish growth through several separated neighborhoods. Both can eventually develop a connected defector-rich region with a depleted interior and a surrounding resource layer. The distinct early morphologies motivate a quantitative description of interfacial activity and directional bias.

[FIGURE:fields]

## Geometry and the direction of local motion

Interface geometry determines both the number of possible strategy-changing contacts and their local payoff differences. An isolated defector is surrounded by cooperators and can receive payoff contributions from all four neighboring sites. In a compact defector cluster, interior defectors have fewer cooperative neighbors, while the shorter boundary offers fewer opportunities for imitation across strategies. The microscopic organization of an interface is known to matter for spatial game dynamics \cite{roca2009imitation}; here its contribution can be expressed directly. All quantities are evaluated on the state after the resource update and before the first strategy visit. For an undirected interfacial edge $e=(u=C,v=D)$, let $x_e=[\Pi_D(v)-\Pi_C(u)]/K$. We define the signed interfacial bias as
$$
\label{eq:J}
J=\frac14\sum_{e\in\mathcal I}\tanh(x_e/2).
$$
Under the smooth imitation rule, uniform sampling of a site and one neighbor during the first strategy visit gives
$$
\label{eq:moments}
\begin{aligned}
\mathbb E[Z\mid s,r]&=J/N,\\
\mathbb E[Z^2\mid s,r]&=P/(4N),\\
\operatorname{Var}(Z\mid s,r)&=P/(4N)-(J/N)^2,
\end{aligned}
$$
Here $P=|\mathcal I|$ is the number of undirected interfacial edges and $Z$ is the change in defector count during the first strategy visit, conditional on the fixed pre-visit state. Equation~\eqref{eq:moments} separates activity from direction: $P/(4N)$ is the probability of a population-changing event at that visit, whereas $J/N$ is its signed mean. A long interface can therefore remain active even when opposing update biases nearly cancel.

For a state sampled before its resource update, $J_{\rm pre}(s,r)=J(s,T_s(r))$ includes that update once. An isolated defector has four cooperative neighbors, each with three cooperative neighbors of its own. On the positive reward branch its initial bias is
$$
\label{eq:isolated}
J_{\rm iso}=
\tanh\!\left[
\frac{4b-3\{1+\lambda\sigma(1-D/2)\}}{2K}
\right]
$$
The isolated-defector expression illustrates how payoff advantage and diffusion act together at the initial boundary. At the reference parameters, the ensemble-averaged initial bias is positive for dispersed perturbations and negative for compact ones (Sec.~\ref{sm-sec:additional} of the Supplemental Material, Fig.~\ref{sm-fig:initialinterfaces}). The dispersed interface is also substantially longer, producing more opportunities for strategy changes.

Figure~\ref{fig:growthkinetics} follows these effects beyond the first visit. Early in the evolution, the dispersed interface becomes increasingly active even as the normalized signed bias approaches zero. The compact interface remains less active. The upper panels show how these different early responses are reflected in finite-time recovery, collapse, and unresolved probabilities. The normalized quantity $4\langle J_{\rm pre}\rangle/\langle P\rangle$ is a ratio of ensemble means, not an average of edgewise biases. Equation~\eqref{eq:moments} applies exactly to the first visit conditioned on a fixed interface; it is not an exact formula for the net population change over a complete MCS, during which the strategy configuration evolves. The clipped implementation and its corresponding identities are treated in Sec.~\ref{sm-sec:proofs} of the Supplemental Material.

[FIGURE:growthkinetics]

## Establishment and spatial consolidation

To distinguish establishment from subsequent expansion, let $\mathcal C_\alpha$ be the periodic nearest-neighbor components of the defector set. The largest-cluster share is
$$
\label{eq:clusters}
f_{\max}(t)=\frac{\max_\alpha|\mathcal C_\alpha(t)|}{n_D(t)},
\qquad n_D=N(1-\rho),
$$
with $f_{\max}=0$ for an empty defector set. Configurations are compared at first crossings,
$$
\label{eq:crossing}
\tau_f=\inf\{t:n_D(t)/N\ge f\}.
$$
The threshold-crossing ensembles support the spatial interpretation in Fig.~\ref{fig:fields}. Near the first 5% crossing, compact-origin disturbances are typically concentrated in one dominant cluster, whereas dispersed-origin disturbances retain several separated clusters. By the first 20% crossing, both preparations are usually dominated by a connected component; the complete cluster-count and arrival-time distributions are shown in Fig.~\ref{sm-fig:growthdistributions}.

The geometry contrast is especially pronounced in the probability of reaching the first growth threshold. Because every trajectory that satisfies the collapse criterion must previously have crossed 5% defectors,
$$
\label{eq:reach_decomposition}
P_B(T)=\Pr(\tau_{0.05}\le T)\,
\Pr(\tau_B\le T\mid\tau_{0.05}\le T).
$$
This decomposition separates an establishment factor from a subsequent conditional factor. Dispersed seeds have a much larger establishment probability, whereas the conditional travel times between later population thresholds are more similar. These observations locate an important part of the geometry effect in the early establishment stage, without requiring a constant propagation speed or a universal coarsening law.

The ordering is retained when the initially converted sites are not depleted of resources. Hence the initial resource loss can affect the quantitative outcome but is not required for the observed geometry contrast. What matters is the spatial organization of interacting strategies and the resource field that develops as their dynamics unfold.
