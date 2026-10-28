## scale | Filter-wise normalization (Sec. 4)
### Random-direction slice
$$f(\alpha,\beta)=L(\theta^*+\alpha\delta+\beta\eta),\qquad \delta,\eta\sim\mathcal N(0,I)$$
### Scale invariance
For ReLU, $w_2\,\mathrm{ReLU}(w_1x)=\tfrac{w_2}{c}\,\mathrm{ReLU}(c\,w_1x)$, $c>0$: same function, different weight scale. Raw directions see the scaled copy as sharper or flatter.
### Fix
Normalize each filter $j$ of layer $i$ of the direction to the norm of the matching filter of $\theta$:
$$\delta_{i,j}\leftarrow\frac{\delta_{i,j}}{\lVert\delta_{i,j}\rVert}\,\lVert\theta_{i,j}\rVert$$
Then $\theta'+\alpha\delta'$ computes exactly the same function as $\theta+\alpha\delta$, so the two slices coincide.
> Offline check: normalized slices identical (max diff 0); raw slices differ by up to 425.
### Network on screen
2 → 16 → 1 ReLU, squared loss, 80 points (two moons), trained to 100% accuracy (loss 0.0095).

## hidden | What a slice cannot show (Sec. 6)
$$L(x,y,z)=\tfrac12\big(x^2+y^2-2z^2\big),\qquad \nabla^2L=\mathrm{diag}(1,1,-2)$$
Slice spanned by $e_x$ and $\cos\varphi\,e_y+\sin\varphi\,e_z$:
$$F(a,b)=\tfrac12\big(a^2+(\cos^2\varphi-2\sin^2\varphi)\,b^2\big)$$
Negative curvature appears once $\tan^2\varphi>\tfrac12$ ($\varphi>35.3°$). The paper checks plots with $|\lambda_{\min}/\lambda_{\max}|$ computed by Lanczos (Fig. 7).

## evidence | Architecture and trainability (Figs. 5–7)
- without skip connections, deeper nets show chaotic, high-curvature landscapes and higher test error
- skip connections and width keep large, smooth-looking basins
- normalized sharpness correlates with test error; raw sharpness does not (Fig. 2, App. Fig. 11–12)
These are empirical associations on CIFAR-10, not theorems.
