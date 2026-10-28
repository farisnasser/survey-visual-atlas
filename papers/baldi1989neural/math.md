## pca | Unique minimum, every other critical point a saddle
### The model
A linear network with one hidden layer of $p$ units: $\hat y = ABx$, $B\in\mathbb R^{p\times n}$, $A\in\mathbb R^{n\times p}$, squared error
$$E(A,B)=\sum_t \lVert y_t - ABx_t\rVert^2$$
### Main result (Sec. 2)
Up to equivalence, $E$ has a unique local and global minimum: the orthogonal projection onto the top-$p$ eigenvectors of $\Sigma=\Sigma_{YX}\Sigma_{XX}^{-1}\Sigma_{XY}$ (autoencoder: $\Sigma=\Sigma_{XX}$). All other critical points are saddles.
### Toy model on screen
$\Sigma=\mathrm{diag}(9,1)$, $p=1$, tied weights $B=A^\top=w^\top$:
$$E(w)=\operatorname{tr}\Sigma-2\,w^\top\Sigma w+\lVert w\rVert^2\,w^\top\Sigma w$$
- $w=\pm u_1$: $E=1$, Hessian eigenvalues $16, 72$ (minimum)
- $w=\pm u_2$: $E=9$, eigenvalues $-16, 8$ (saddle)
- $w=0$: $E=10$, eigenvalues $-36,-4$ on this slice
> Assumes $\Sigma_{XX}$ invertible and distinct eigenvalues. Checked in scripts/check-math/baldi1989neural.py.

## angle | PCA from the loss
For a unit direction $u$, keep $z=u^\top x$ and rebuild $\hat x = u z = uu^\top x$:
$$\tfrac1N\sum_i\lVert x_i-uu^\top x_i\rVert^2=\operatorname{tr}\Sigma-u^\top\Sigma u$$
With $u(\theta)=(\cos\theta,\sin\theta)$ and $\Sigma=\mathrm{diag}(9,1)$:
$$E(\theta)=1+8\sin^2\theta$$
Minimum at $\theta=0$ (PC1). $\theta=90°$ is stationary but a maximum along the circle; in the full weight space it is a saddle.

## equivalent | Weights are never unique
For any invertible $C$: $AB=(AC)(C^{-1}B)$. Uniqueness holds only for the map $W=AB$.
### Slice on screen
$A=(a,0)^\top$, $B=(b,0)$:
$$E(a,b)=9\,(1-ab)^2+1$$
- every point with $ab=1$ is a global minimum with the same $W$
- floor $E=1=\lambda_2$: the variance a one-unit code cannot keep
- origin: saddle with $\lambda=\pm18$
This symmetry is why neural-network Hessians have zero eigenvalues along valleys.
