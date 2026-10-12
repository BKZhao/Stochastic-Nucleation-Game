<!-- Problem Alignment Notes: P1–P3. Physical kernel remains code/b_eco_unified.py; exact resource balance is verified separately. -->
# Model and observables

We consider a periodic square lattice containing $N=L^2$ sites, each with four nearest neighbors. Each site $i$ carries a binary strategy $s_i\in\{0,1\}$, with $s_i=1$ representing cooperation, and a resource level $r_i\in[0,1]$. One Monte Carlo step (MCS) consists of a synchronous resource update followed by one strategy-update visit to each site in random order. Resource dynamics combine local renewal or depletion with nearest-neighbor diffusion:
$$
\label{eq:resource}
\begin{aligned}
u_i&=r_i+g s_i(1-r_i)-\delta(1-s_i)r_i,\\
r_i^{+}&=(1-D)u_i+\frac D4\sum_{j\sim i}u_j .
\end{aligned}
$$
For $0\le g,\delta,D\le1$, the update preserves $r_i\in[0,1]$ in exact arithmetic; numerical clipping to this interval is therefore a safeguard rather than an additional model rule.

During the strategy-update stage, the resource field remains fixed. Sites are visited in a uniformly random order, and a visited site $i$ samples a nearest neighbor $j$. The strategy of $j$ is copied according to the Fermi rule
$$
\label{eq:fermi}
p_{i\leftarrow j}=\frac{1}{1+\exp[(\Pi_i-\Pi_j)/K]} .
$$
where the payoffs are evaluated using the strategy configuration at the time of the visit. Writing $k_C(i)=\sum_{j\sim i}s_j$ for the number of cooperative neighbors, the payoffs are
$$
\label{eq:payoffs}
\begin{aligned}
\Pi_C(i)&=R(r_i^+)k_C(i),\qquad \Pi_D(i)=b\,k_C(i),\\
R(r)&=\max\{0,1+\lambda\sigma(2r-1)\}.
\end{aligned}
$$
The Fermi rule is standard in stochastic evolutionary games \cite{szabo1998lattice,traulsen2006stochastic}. We use $L=48$, $b=1.16$, $D=0.4$, $g=0.1$, $\delta=0.5$, $\lambda=0.5$, $\sigma=1$, and $K=0.1$ as reference parameters. At this coupling, $R(r)=0.5+r$ remains on the positive branch. All parameters are dimensionless, and time is measured in MCS. In the absence of mutation, the two homogeneous strategy configurations are absorbing under imitation. The implementation clips the Fermi exponent to $[-30,30]$; the Supplemental Material \cite{supplement}, Sec.~\ref{sm-sec:proofs}, distinguishes this numerical convention from the smooth diagnostic used in the analysis.

## Resource balance and spatial correlation

Let $\rho=N^{-1}\sum_i s_i$, $\bar r=N^{-1}\sum_i r_i$, and $\langle sr\rangle=N^{-1}\sum_i s_i r_i$. Diffusion conserves the sum of its input field on the periodic lattice. The change of the mean resource during a complete resource substep is therefore
$$
\label{eq:balance}
\begin{aligned}
\bar r^+-\bar r
={}&g\rho-\delta\bar r+(\delta-g)\langle sr\rangle\\
={}&g\rho(1-\bar r)-\delta(1-\rho)\bar r\\
&+(\delta-g)C_{sr},
\end{aligned}
$$
where $C_{sr}=\langle sr\rangle-\rho\bar r$ is the spatial strategy--resource covariance. The first two terms in the covariance form describe renewal and depletion evaluated at the mean strategy and resource levels. The covariance term measures their spatial association and cannot generally be reconstructed from $\rho$ and $\bar r$ alone. Thus the pair $(\rho,\bar r)$ does not form a closed description of the resource dynamics. Diffusion cancels from the instantaneous total-resource balance on the periodic lattice, but it reshapes the resource field and thereby influences later resource and strategy updates.

## Equal-budget perturbations and fate probabilities

Starting from an all-cooperator configuration with unit resources, we convert exactly $m=16$ sites to defectors and set the resources at those sites to zero. For compact perturbations, the selected sites are the 16 closest to a common center in squared periodic Euclidean distance, with ties resolved by a fixed coordinate ordering. For dispersed perturbations, 16 sites are sampled uniformly without replacement. The two protocols therefore have identical initial defector counts and resource deficits, differing only in the spatial placement of the perturbation. We also consider a single-defector seed as a size control.

We label persistent depletion as event $B$ when both $\rho\le0.2$ and $\bar r\le0.2$ hold for 250 consecutive MCS. Recovery (event $A$) is recorded when all sites cooperate and $\bar r\ge0.6$, provided event $B$ has not already occurred. Each trajectory retains its first recorded event. Trajectories satisfying neither criterion by time $t$ belong to the unresolved class $U$, giving
$$
\label{eq:partition}
P_A(t)+P_B(t)+P_U(t)=1.
$$
The persistence condition defines an operational first-passage event; it does not make the depleted threshold region mathematically absorbing. Unresolved trajectories remain in the denominators of all numerical event probabilities. Observation windows, simulation counts, and uncertainty estimates are given in Sec.~\ref{sm-sec:checks} of the Supplemental Material. Conditional fate probabilities for continuations from grown configurations are defined in Eq.~\eqref{eq:conditional_fate}.
