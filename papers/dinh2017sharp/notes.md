# Sharp Minima Can Generalize for Deep Nets

:::minute The paper in one minute
**Question.** *Is ordinary parameter-space sharpness a property of a trained function, or merely of how that function is parameterized?*

**Method.** Exploit the exact symmetry of ReLU networks: the key identity of rectified linear units is *positive homogeneity*.

**Main result.** Reparameterization/symmetry can make functionally equivalent deep-network solutions arbitrarily sharper by common parameter-space measures, while predictions and generalization remain unchanged. Therefore naive sharpness alone cannot be a universal explanation for generalization.

**Limits.** It does *not* prove that all flatness measures are useless, that sharpness never correlates with generalization empirically, or that large-batch generalization gaps are imaginary.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. Many weight vectors compute the same function

A network is a map from weights to a function. That map is **many-to-one**. Take the one-unit model $f(x) = w_2\,\sigma(w_1 x)$ with $\sigma(z) = \max\{0, z\}$ and $w_1 > 0$. Then $f(x) = w_1 w_2\,\sigma(x)$: only the product $w_1 w_2$ matters. The weights $(1, 1)$, $(2, 0.5)$ and $(4, 0.25)$ give the same predictions on every input, so they have the same training loss and the same test loss.

### 2. Curvature numbers depend on the coordinates

A Hessian eigenvalue is measured in units of loss per (weight unit)$^2$. Rename the weight and the number changes. In one dimension, $L(\theta) = \theta^2$ has curvature $2$. Write $\theta = \phi/10$: the same loss is $L = \phi^2/100$, with curvature $0.02$ in $\phi$. Nothing about the function changed; only the ruler did. In general a linear change of variables $\theta = D\phi$ turns the Hessian $H$ into $D^\top H D$, which can have very different eigenvalues.

### 3. Neighbourhood measures inherit the same problem

Sharpness can also be measured without a Hessian: take the largest loss increase over a ball of radius $\epsilon$ around the minimum. For $L(a, b) = (ab - 1)^2$ and $\epsilon = 0.1$, this increase is about $0.021$ at the minimum $(1, 1)$ and about $0.161$ at the minimum $(4, 0.25)$. Both points compute the same product $ab = 1$. A fixed-radius ball covers a different set of functions depending on where it is placed, so the measure changes along with the coordinates.

:::key Putting the three ideas together
Equal functions have equal generalization (idea 1). Any sharpness measure built from raw weight coordinates, whether a Hessian eigenvalue (idea 2) or a worst case over a ball (idea 3), can still give them different values. For ReLU networks the rescaling symmetry is exact, so such a measure can be pushed as high as we like without changing predictions. The notes below derive this for a two-layer model and state what it does and does not rule out.
:::
:::

:::divider Study notes
:::

## 1. The research question: does flatness explain generalization?

Keskar et al. connected large-batch training with sharp minima and a test-performance gap. Dinh et al. ask a more fundamental question: *is ordinary parameter-space sharpness a property of a trained function, or merely of how that function is parameterized?* If two parameter vectors give identical predictions for every input, they have identical training and test performance; a measure intended to explain generalization should ideally not give conflicting conclusions for them.

## 2. Method: exploit the exact symmetry of ReLU networks

The key identity of rectified linear units is *positive homogeneity*:

$$\sigma(cx) = c\sigma(x) \qquad (c > 0), \qquad \sigma(z) = \max\{0, z\}.$$

For a simple two-layer model, $f(x; W_1, W_2) = W_2\sigma(W_1 x)$, choose any $c > 0$ and change weights to

$$W_1' = cW_1, \qquad W_2' = c^{-1}W_2.$$

Then, for every $x$,

$$f(x; W_1', W_2') = c^{-1}W_2\sigma(cW_1 x) = W_2\sigma(W_1 x) = f(x; W_1, W_2).$$

Hence a loss that depends only on predictions obeys $L(W_1', W_2') = L(W_1, W_2)$, even though the coordinates in weight space can change by an arbitrarily large scale factor. The same symmetry extends to appropriate units and layers of rectifier networks.

## 3. Why this changes sharpness: first-principles calculation

A one-dimensional *teaching model*, not a theorem quoted from the paper, makes the coordinate issue transparent. Let $L(a, b) = (ab - 1)^2$ and take the global minimum $(a, b) = (c, c^{-1})$. Its Hessian is

$$\nabla^2 L(c, c^{-1}) = 2\begin{pmatrix} c^{-2} & 1 \\ 1 & c^2 \end{pmatrix}.$$

The eigenvalues are $0$ and $2(c^2 + c^{-2})$. Thus, at equally good global minima representing the *same* product $ab = 1$, the largest Hessian eigenvalue grows without bound as $c \to \infty$. Curvature is not invariant under this symmetry. Dinh et al. establish analogous problems for definitions of flatness and sharpness in actual rectified networks.

:::theorem Main result
Reparameterization/symmetry can make functionally equivalent deep-network solutions arbitrarily sharper by common parameter-space measures, while predictions and generalization remain unchanged. Therefore naive sharpness alone cannot be a universal explanation for generalization.
:::

## 4. Assumptions and limits

The central construction relies on parameter symmetries, particularly positive homogeneity of rectifier architectures and appropriate losses. It does *not* prove that all flatness measures are useless, that sharpness never correlates with generalization empirically, or that large-batch generalization gaps are imaginary. Geometry-aware or reparameterization-invariant notions may still be informative.

:::plain Survey relevance (RQ1, RQ4)
This is a crucial qualification to "flat minima generalize": compare geometric measurements across models only after stating the parameterization and normalization. The conclusion is a counterexample to universal interpretation, not a claim that SGD does not prefer particular basins.
:::

## 5. Our notes / comparison with Ref. [26]

Keskar et al. report an empirical association between batch size, sharpness and generalization. Dinh et al. show why the plain weight-space sharpness in that explanation can change without changing the model's function. The two papers are best read together: empirical correlation is not a coordinate-invariant causal explanation.

> **Original source:** [https://proceedings.mlr.press/v70/dinh17b.html](https://proceedings.mlr.press/v70/dinh17b.html) (main paper and supplement).
