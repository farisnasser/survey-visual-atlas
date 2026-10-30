## copositive | Copositivity (Sec. 3, corollary)
A symmetric $Q$ is **copositive** if $x^\top Qx\ge0$ for all $x\ge0$; PSD asks the same for all $x$.
$$Q=\begin{pmatrix}1&q\\q&1\end{pmatrix}:\qquad \text{PSD}\iff|q|\le1,\qquad \text{copositive}\iff q\ge-1$$
On the simplex $x_1+x_2=1$, $x\ge0$: $\min x^\top Qx=\min\{1,\tfrac{1+q}2\}$.
- The origin is a local minimum of $x^\top Qx$ over $x\ge0$ iff $Q$ is copositive.
- Murty–Kabadi: checking that an integer matrix is **not** copositive is NP-complete.
> Check script: copositivity and PSD thresholds confirmed on 20 000 simplex samples for seven values of $q$.

## quartic | A quartic with a zero Hessian
$$p(u)=\sum_{ij}Q_{ij}u_i^2u_j^2=u_1^4+2q\,u_1^2u_2^2+u_2^4,\qquad \nabla p(0)=0,\ \nabla^2p(0)=0$$
The origin is a local minimum of $p$ iff $Q$ is copositive, so deciding local minimality of a degree-4 polynomial is NP-hard (restated in Ahmadi & Zhang 2020).
- $q>-1$: strict local (and global) minimum at 0, invisible to first- and second-order tests
- $q=-1$: $p=(u_1^2-u_2^2)^2$, a valley of non-strict minima
- $q<-1$: $p(t,t)=2(1+q)t^4<0$
> Check script: GD from $(0.6,0.62)$, $\eta=0.05$: creeps to 0 for $q\in\{-0.5,0.7\}$, stops on $u_1=u_2$ for $q=-1$, runs to the box edge for $q=-1.4$.

## subset | Subset sum as a QP (Example 2)
Given integers $d_1,\dots,d_n,d_0$:
$$F(y)=\Big(\sum_j d_jy_j-d_0\Big)^2+\sum_j y_j(1-y_j),\qquad 0\le y\le1$$
Both terms are $\ge0$; $F=0$ forces $y\in\{0,1\}^n$ with $\sum_j d_jy_j=d_0$. So $\min F=0$ iff the subset-sum instance is solvable, and computing a global minimum of a non-convex QP is NP-hard.
> Check script ($d=(3,5)$): $\min F=0$ for $d_0\in\{3,5,8\}$; for $d_0=4$, $\min F=0.1562$ at $(1,0.1875)$. Projected GD also stops at the non-global value $0.24$ for $d_0\in\{3,5\}$.

### Paper's main statement
For smooth non-convex NLP, checking (a) that a feasible point is **not** a local minimum or (b) that the objective is **not** bounded below is NP-complete, already for indefinite QPs with simple constraints and integer data.
