# Deep Residual Learning for Image Recognition

:::minute The paper in one minute
**Question:** Why can adding layers make a neural network harder to train, even when the deeper model can represent the shallower one?

**Model/Method:** Replace direct learning of a mapping $H(x)$ by learning its residual $F(x) = H(x) - x$; construct identity-shortcut residual blocks and test very deep CNNs.

**Finding:** Residual networks optimize much more successfully than comparable plain networks as depth increases; the authors train models up to 152 layers on ImageNet and explore 100/1000-layer CIFAR models.

**Why we care:** Architecture can alter the difficulty of optimization without changing the intended prediction task. Residual connections are an empirical design breakthrough, not a universal global-convergence theorem.
:::

:::fold First principles: what you need before the notes | Written for the Atlas, not part of the original notes: the three ideas the paper stands on, built from scratch.
### 1. A solution can exist and still not be found

Take a trained 20-layer network and stack 36 more layers on top. If those extra layers computed the identity, the 56-layer network would compute exactly the same function, with exactly the same training error. So the deeper network's *best possible* training error is no worse than the shallow one's. If, after training, its training error is *higher*, the optimizer failed to find a solution that provably exists. That is an **optimization** failure, not **overfitting**: overfitting means low training error with high test error. Telling the two apart requires looking at training error, which is what the paper does.

### 2. Gradients through depth are products of many factors

By the chain rule, the gradient that reaches an early layer is a product of one factor per later layer. Products of many numbers drift exponentially: $0.9^{50} \approx 0.005$ and $1.1^{50} \approx 117$. A scalar toy shows why identity shortcuts matter at initialization. Let each of $20$ layers multiply its input by a weight near $0$, say $0.1$. In a plain chain the signal reaching the first layer is scaled by $0.1^{19} = 10^{-19}$. In a residual chain each layer computes $x + a\,x$, and with the same small weight $a = 0.1$ the factor is $1.1^{19} \approx 6.1$; with $a = 0$ it is exactly $1$. A residual block initialized near zero behaves like the identity and passes the signal through. This is the textbook mechanism; the paper itself argues that its *plain* networks, trained with batch normalization, do not suffer vanishing signals (idea 3).

### 3. Batch normalization keeps each layer's signal in range

**Batch normalization (BN)** rescales every channel, over the examples in a minibatch, to mean $0$ and variance $1$ before a learned scale and shift. The three activations $(2, 4, 6)$ become about $(-1.22, 0, 1.22)$, whatever their original size. With BN after every convolution, forward activations cannot shrink to zero or blow up layer by layer, and the authors check that backward gradients also keep healthy norms. That is why they conclude the degradation of deep plain networks is *not* caused by vanishing gradients and conjecture slow convergence instead.

:::key Putting the three ideas together
A deeper network can always copy a shallower one, so higher training error with depth is an optimization failure (idea 1). Identity shortcuts make each block start near the identity, so depth no longer multiplies the signal by many small factors (idea 2); and because the networks already use batch normalization (idea 3), the paper's evidence is about trainability beyond vanishing gradients. Residual learning removes the degradation in practice, as made precise in the notes below.
:::
:::

:::divider Study notes
:::

## 1. What is the degradation problem?

If a shallow network fits the training data, a deeper network of sufficient capacity could simply copy the shallow mapping and make added layers act as identities. Yet the authors observe that plain networks become harder to optimize: a 56-layer plain model has higher training error than a 20-layer counterpart (Figure 1). This is not merely overfitting, because the training error itself is worse.

## 2. The residual reformulation

Instead of forcing stacked layers to directly learn $H(x)$, let them learn

$$F(x; W) = H(x) - x, \qquad y = F(x; W) + x.$$

If the desirable mapping is already close to the identity, the residual function can be close to zero. A shortcut copies $x$ directly to the output and does not itself introduce trainable parameters in the identity case.

### Worked scalar example

Imagine the desired transformation is $H(x) = 1.05x$. A plain layer would learn multiplication by 1.05; the residual branch only needs to learn

$$F(x) = 0.05x, \qquad y = x + 0.05x = 1.05x.$$

If $F(x) = 0$, the block immediately implements the identity. This toy example illustrates the representational convenience; it is not a general proof that optimization must be fast.

## 3. Why shortcuts can improve backpropagation

For a residual block, its input-output Jacobian is

$$\frac{\partial y}{\partial x} = I + J_F(x).$$

For a scalar residual branch $F(x) = 0.05x$, the Jacobian is 1.05 rather than 0.05. Gradients have a direct identity contribution through each shortcut, even when the learned branch is small. This helps explain why residual pathways can provide more direct signal propagation; it does not imply that vanishing or exploding gradients become impossible in arbitrary deep ResNets.

:::key A subtle but crucial distinction
The authors' central observation is about optimization difficulty, as shown by training error, not merely test accuracy or theoretical expressivity.
:::

## 4. What exactly is implemented?

The basic residual unit applies a stack of convolutions/normalizations and adds the shortcut: $y = F(x; W) + x$. If feature dimensions differ, the paper discusses zero padding or a learned projection shortcut, usually written $W_s x$; shortcut and residual outputs must have the same shape.

### Plain versus residual comparison

Experiments control architecture families so that a plain network and a corresponding residual network have similar layered components, but the residual one adds shortcuts. Training curves show that very deep plain networks can have worse training error, whereas residual counterparts benefit substantially from depth.

### Bottleneck blocks for very deep nets

The ImageNet 50/101/152-layer designs use a bottleneck pattern with $1 \times 1$, $3 \times 3$, and $1 \times 1$ convolutions, reducing and then restoring channel width. This lowers computational cost relative to stacking wide $3 \times 3$ convolutions at every stage.

## 5. How to interpret the experiments

**CIFAR-10.** The paper shows successful optimization of residual networks with 100 and even 1000 layers, illustrating that very deep architectures are trainable in this design. **ImageNet.** A 152-layer residual network achieves a 4.49% single-model top-5 *validation* error, while an ensemble attains 3.57% top-5 *test* error in ILSVRC 2015. *Do not attribute the ensemble score to a single model.* **Transfer.** Strong image-recognition representations also improve COCO object detection, supporting the practical importance of depth but not isolating every causal mechanism of residual optimization.

:::plain Connection to the survey
This is a foundational *empirical architecture paper*. Later results on deep ResNet optimization and depth-dependent conditioning analyze more restrictive mathematical models; those later theorems should not be attributed retroactively to He et al.
:::

## 6. Contributions, assumptions, limitations

**Contributions.** (i) Residual learning through identity shortcuts; (ii) a controlled demonstration of depth-related training-error degradation in plain nets; (iii) practical very-deep ResNet designs; (iv) substantial ImageNet and transfer results. **Assumptions/context.** Residual additions require aligned tensor dimensions; projected shortcuts may be used when shape changes. Empirical comparisons use particular initialization, batch normalization, SGD schedules, datasets, and CNN block designs. **Limitations.** The work does not prove global optimality, absence of spurious minima, or saddle escape. The Jacobian intuition does not constitute an exact trajectory theorem for 152-layer SGD. Residual and plain architectures also can differ in optimization statistics and parameterization.

## 7. Cross-paper connections

**Du et al. (2019):** provides convergence theory for specified overparameterized ResNet settings. **Nguyen (2021):** studies convergence with favorable feature conditioning. **Saxe (2014):** explains nonlinear optimization dynamics in deep linear nets. Here, the architectural shortcut acts as an intervention on trainability, not an added saddle-escape perturbation.

## 8. Quick reference

| | |
|---|---|
| **Residual block** | $y = x + F(x; W)$ (identity shortcut). |
| **Target reformulation** | $F(x) = H(x) - x$. |
| **Gradient route** | $\partial y / \partial x = I + J_F$. |
| **Projection shortcut** | $y = F(x) + W_s x$ when shapes require alignment. |
| **Observed obstruction** | Plain-network training error can worsen with depth. |
| **ImageNet outcome** | 152-layer single: 4.49% validation top-5; ensemble: 3.57% test top-5. |
| **Theorem status** | Strong empirical evidence; no universal global-GD theorem. |

### Papers Master note

He et al. show that learning residual functions using identity shortcuts greatly improves the optimization of very deep CNNs, overcoming training-error degradation observed in plain networks. Important architectural foundation for later ResNet convergence theory, not a saddle-escape or global-optimality proof.

> **Original:** [https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf).
