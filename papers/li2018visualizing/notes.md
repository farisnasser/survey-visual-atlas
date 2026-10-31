# Visualizing Loss Landscapes

:::minute The paper in one minute
**Question:** Why are some deep networks trainable while others have difficult loss landscapes, and can loss "sharpness" plots be trusted?

**Model/Method:** Visualize 1D/2D slices of neural-network loss after *filter-wise normalizing* the random plot directions, then compare networks, batch sizes, and Hessian eigenvalues.

**Finding:** Unnormalized comparisons can reverse a sharp/flat conclusion merely through weight scaling. In the tested models, skips and greater width yield smoother-looking landscapes and lower test error; normalized sharpness correlates with test error.

**Why we care:** A careful visualization method links network architecture to optimization geometry without claiming the entire high-dimensional loss is convex.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the four ideas the paper stands on, built from scratch.
### 1. Random directions in high dimension are almost orthogonal

The plots use two random Gaussian directions $d$ and $e$. In $D$ dimensions, the cosine of the angle between two independent Gaussian vectors is about $\pm 1/\sqrt{D}$: about $0.01$ for $D = 10^4$, and about $0.001$ for a network with $10^6$ parameters. So the two directions span a genuine plane, and each is nearly orthogonal to any particular direction of interest, such as the Hessian's top eigenvector. A random slice is a *typical* slice, not a special one.

### 2. A slice sees a weighted average of the Hessian's eigenvalues

Along a unit direction $u$, the curvature of the slice at $\theta^*$ is $u^\top H u = \sum_i \lambda_i\,(u^\top q_i)^2$, where $q_i$ are the eigenvectors of $H$ and the weights $(u^\top q_i)^2$ add up to one. So it always lies between $\lambda_{\min}$ and $\lambda_{\max}$. Example: for $H = \mathrm{diag}(1, 1, -2)$ (the notes' §4 saddle) and $u = (1, 0, 1)/\sqrt2$, the curvature is $(1 - 2)/2 = -0.5$; for $u = (1, 1, 0)/\sqrt2$ it is $+1$. A negative value proves that $\lambda_{\min} < 0$; a positive value proves nothing about the other directions.

### 3. Random slices see the average curvature, so they hide rare negative directions

For a random unit direction the weights are spread almost evenly, about $1/D$ each, so the slice curvature is close to $\mathrm{tr}(H)/D$, the mean eigenvalue. Example: $D = 1000$ with one eigenvalue $-2$ and $999$ eigenvalues $+1$ gives a mean of $(999 - 2)/1000 = 0.997$; almost every random slice looks like a bowl even though the point is a saddle. This is why the paper checks its pictures against the extreme eigenvalues $\lambda_{\min}$ and $\lambda_{\max}$ (Fig. 7) instead of trusting them.

### 4. Extreme eigenvalues can be found without ever forming the Hessian

A network with $10^6$ parameters has a Hessian with $10^{12}$ entries, about $4$ terabytes in 32-bit floats. But the product $Hv$ with any vector costs about as much as two gradient evaluations, since $Hv \approx \big(\nabla L(\theta + \varepsilon v) - \nabla L(\theta - \varepsilon v)\big)/(2\varepsilon)$, and autodiff computes it exactly. Iterative methods such as power iteration and **Lanczos** only need these products to find $\lambda_{\max}$ and $\lambda_{\min}$: that is how the paper measures $|\lambda_{\min}/\lambda_{\max}|$.

:::key Putting the four ideas together
Random plot directions give typical, nearly orthogonal slices (idea 1), and the curvature a slice shows is a weighted average of the Hessian's eigenvalues (idea 2). Averages hide rare negative directions (idea 3), so the paper backs its pictures with extreme eigenvalues computed from Hessian–vector products (idea 4). The notes below add the scale problem that filter normalization fixes and the paper's findings.
:::
:::

:::divider Study notes
:::

## 1. Start with the high-dimensional loss, not the pictures

A trained network has millions of parameters $\theta$, so its loss $\mathcal{L}(\theta)$ cannot be displayed directly. Pick a trained parameter vector $\theta^*$ and two directions $d, e$; then draw only the *slice*

$$\boxed{F(\alpha, \beta) = \mathcal{L}(\theta^* + \alpha d + \beta e)}\,.$$

Moving right changes the weights along $d$; moving up changes them along $e$. A contour shows combinations of $(\alpha, \beta)$ giving the same loss. Close, nearly elliptical contours suggest a locally bowl-like slice; irregular contours reveal non-convex behavior *in that slice*. Paper Secs. 3–4.

## 2. Worked example: the same function can look sharper or flatter

A ReLU network can represent exactly the same input–output function using very different weight scales. For $x > 0$ and $c > 0$,

$$\hat{y} = w_2\,\mathrm{ReLU}(w_1x) = \underbrace{\frac{w_2}{c}}_{w_2'}\,\mathrm{ReLU}(\underbrace{cw_1}_{w_1'}x).$$

**Step 1 — Two equivalent networks.** Let $x = 1$ and compare $(w_1, w_2) = (2, 0.5)$ against $(w_1', w_2') = (20, 0.05)$ with $c = 10$. Both predict $\hat{y} = 1$.

**Step 2 — Apply the same absolute perturbation.** Increase *only* the first weight by 0.1. The original now predicts $0.5 \times 2.1 = 1.05$; the rescaled model predicts $0.05 \times 20.1 = 1.005$.

**Step 3 — Interpret.** Identical networks appear to have *different sensitivity* merely because 0.1 is 5% of one first-layer weight and 0.5% of the other. Thus naive sharpness comparisons can reflect parameter scale, not a meaningful difference in learned function.

:::key The key warning
Li et al. show an analogous failure empirically: in the same small-versus-large SGD batch comparison, enabling weight decay can *reverse* which solution looks sharper under naive 1D interpolation (paper Fig. 2), even though small-batch training still gives lower test error.
:::

## 3. What filter-wise normalization actually does

First sample a random direction tensor $d$ matching the network parameters. For every filter $j$ in layer $i$, rescale its direction to the *norm of that filter's trained weights*:

$$\boxed{\widetilde{d}_{i,j} = \frac{d_{i,j}}{\|d_{i,j}\|_F}\,\|\theta_{i,j}\|_F} \qquad (d_{i,j} \neq 0).$$

Use separately normalized directions $\widetilde{d}, \widetilde{e}$ in $F(\alpha, \beta)$. This changes *the measurement directions for the plots*, not how the network was trained. For our scalar example, a 5% *relative* first-weight change gives $2 \to 2.1$ and $20 \to 21$; both models now output 1.05. This is the intuition behind a scale-aware visualization, not a proof of complete parameterization invariance.

## 4. A second mathematical trap: a nice 2D bowl may hide a saddle

A picture cannot show all curvature directions. Consider

$$\mathcal{L}(x, y, z) = \tfrac{1}{2}(x^2 + y^2 - 2z^2), \qquad H = \nabla^2\mathcal{L} = \mathrm{diag}(1, 1, -2).$$

**Step 1 — Slice in the $xy$ plane:** setting $z = 0$ gives $\tfrac{1}{2}(x^2 + y^2)$, a perfectly convex-looking bowl.

**Step 2 — Check the full Hessian:** its eigenvalues are $(1, 1, -2)$, so $z$ is a hidden negative-curvature direction and the origin is a saddle.

**Step 3 — Draw the correct conclusion:** an observed *negative* curvature in a slice guarantees a negative curvature direction in the full space; a convex-looking slice *does not* guarantee a globally convex loss. This is exactly the caution the authors give in Sec. 6.

## 5. Three experimental findings worth remembering

| Question | What the paper actually observed |
|---|---|
| **Does batch size affect sharpness?** | On VGG-9 / CIFAR-10, naive 1D sharpness comparisons flipped under changes in weight decay. With filter-normalized 1D/2D plots, the smaller batch (128) had wider minima and lower test error than the large batch (8192) in the studied settings (Figs. 2–3). |
| **Does depth change geometry?** | Without skip connections, ResNet-like nets at 56/110 layers displayed visually irregular, high-curvature landscapes. Skip-connected ResNets retained large, relatively smooth-looking basins as depth increased (Fig. 5). |
| **Does width help?** | Increasing ResNet-56 width from multiplier $k = 1$ to $k = 8$ yielded larger, smoother-looking low-loss regions and improved test error in the plotted models (Fig. 6). |

## 6. How strongly are the observations supported?

The authors also compute minimum and maximum Hessian eigenvalues via Hessian–vector products and Lanczos iterations, plotting the relative negative curvature $\lvert\lambda_{\min}/\lambda_{\max}\rvert$ (paper Fig. 7). Convex-looking slices generally coincide with regions where the most negative curvature is small *relative to* the positive curvature, while irregular regions show stronger negative eigenvalues.

:::warn Read the result precisely
**Test error is not training loss.** For example, paper Fig. 5 reports **5.79%** test error for ResNet-110 *with* skips versus **16.44%** without; this is an experimental comparison, not a theorem that skips always improve performance. Likewise, "chaotic" is the authors' visual description of irregular non-convex slices, *not* a formal proof of dynamical chaos.
:::

## 7. Why is this relevant to gradient-based training?

A loss can have severe non-convex regions but training can still succeed when initialization falls inside a broad, relatively well-behaved basin. The authors *hypothesize* that skip connections and width enlarge such useful regions. Their visualizations help explain how architecture changes the terrain encountered by gradient-based optimizers; they do *not* establish a universal convergence guarantee for SGD.

## 8. Contributions, assumptions, and limitations

**Main contributions.**

- **Measurement method:** propose filter-wise normalized plot directions so weight-scale differences distort sharpness comparisons less (Secs. 4–5).
- **Architecture evidence:** show how depth, residual connections, and width correlate with the appearance of loss surfaces and trainability (Sec. 6).
- **Curvature check:** supplement 2D pictures with minimum Hessian eigenvalues to detect hidden negative curvature (Fig. 7).
- **Training connection:** analyze SGD batch-size comparisons and motivate how benign initial regions may aid optimization.

**Important assumptions / methodological choices.** This is primarily empirical: networks trained on CIFAR-10, largely SGD with momentum, specified initialization and regularization; geometries observed near selected trained models. The 2D directions are random Gaussian then filter normalized. Batch-normalization parameters are held fixed in the main surface plots; the scale-invariance intuition is specific to architectures/transformations where that rescaling preserves predictions.

**Limitations.** A 2D cross-section may miss directions in a high-dimensional loss. Filter normalization mitigates a scale artifact but is *not a universal invariant definition* of sharpness or proof that flatness causes good generalization. Model, optimization, and hyperparameter choices can confound comparisons; the results are associations for these architectures, not universal laws. The paper does *not* invent a new optimizer or prove general GD/SGD escape rates.

:::key Connections to papers we have already studied
**Dauphin (2014)** uses Hessian eigenvalues to diagnose saddles and motivates a curvature-based update; **Li (2018)** uses eigenvalues to test whether its *pictures* conceal negative curvature. **Lee (2016)** proves almost-sure avoidance of strict-saddle convergence by GD under assumptions; Li supplies empirical, architecture-dependent geometry but no analogous theorem. **Saxe (2014)** explains why initialization and multiplicative network structure affect learning dynamics; Li observes differences in landscape geometry induced by architectural choices.
:::

## 9. Quick reference

| Concept | Meaning / formula |
|---|---|
| **2D loss slice** | $F(\alpha, \beta) = \mathcal{L}(\theta^* + \alpha d + \beta e)$ |
| **Filter normalization** | $\widetilde{d}_{i,j} = d_{i,j}\,\lVert\theta_{i,j}\rVert_F/\lVert d_{i,j}\rVert_F$ |
| **Scale invariance** | $w_2\,\mathrm{ReLU}(w_1x) = (w_2/c)\,\mathrm{ReLU}(cw_1x)$ for $c > 0$ |
| **Sharp / flat** | Sensitivity of local *plotted* loss to weight perturbations; depends on how steps are scaled |
| **Negative curvature** | $\lambda_{\min}(H) < 0$; can exist outside an apparently bowl-shaped 2D slice |
| **Architecture finding** | Deeper no-skip nets looked more irregular; skips and greater width looked more benign in tested models |

## 10. What to read, and what to preserve

**Targeted skim:** paper pp. 3–4 for the definition of filter normalization; pp. 5–6 for the sharpness reversal and normalized plots; pp. 7–9 for Figures 5–7, architecture results, and caveats. Revisit the SGD trajectory appendix only if Section IV requires it. Do not memorize the plotting implementation.

:::plain One–three lines for Papers Master
Li et al. (2018) propose filter-wise normalized loss-landscape slices, showing that naive sharpness comparisons can be reversed by weight scaling. On CIFAR-10 networks, residual connections and width correlate with smoother-looking, easier-to-train landscapes; minimum Hessian eigenvalues corroborate some plotted curvature. Results are empirical and 2D visualizations cannot certify full-space convexity or causality.
:::

> **Source:** Li et al. (2018), NeurIPS, Sections 3–7, Figures 2–7. [Original PDF](https://papers.nips.cc/paper_files/paper/2018/file/a41b3bb3e6b050b6c9067c67f663b915-Paper.pdf). Scalar ReLU rescaling example and the 3D hidden-curvature example are *pedagogical adaptations*, not the paper's experiments.
