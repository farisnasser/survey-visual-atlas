## steps | Constrained and regularized steepest descent (Props. 3.1–3.2)
With momentum $m_t=\beta m_{t-1}+(1-\beta)g_t$ and $\mathrm{LMO}(v)=\arg\min_{\lVert u\rVert\le1}\langle v,u\rangle$:
$$\text{CSD: }w_{t+1}=\arg\min_{\lVert w-w_t\rVert\le\eta}\langle m_t,w-w_t\rangle=w_t+\eta\,\mathrm{LMO}(m_t)$$
$$\text{RSD: }w_{t+1}=\arg\min_w\ \langle m_t,w-w_t\rangle+\tfrac1{2\eta}\lVert w-w_t\rVert^2=w_t+\eta\,\lVert m_t\rVert_*\,\mathrm{LMO}(m_t)$$
For the spectral norm on a matrix, $\mathrm{LMO}(M)=-\mathrm{polar}(M)=-UV^\top$ and $\lVert M\rVert_*=\lVert M\rVert_{\text{nuc}}$. Muon is CSD with the spectral norm.
### On screen
$W=\mathrm{diag}(a,b)$: spectral norm $=\max(|a|,|b|)$, $\mathrm{polar}(\mathrm{diag}(g))=\mathrm{diag}(\mathrm{sign}\,g)$, nuclear norm $=|g_1|+|g_2|$.
> Check script (400 steps): GD $10^{-30}$; Muon-style CSD ends in a 2-cycle on the grid of step length 0.05 with loss of order $10^{-3}$ (which cycle is rounding-sensitive: $1.5/3.9\times10^{-3}$ in Python, $0.4/5.0\times10^{-3}$ in the browser); spectral RSD $5.7\times10^{-10}$ and still falling.

## lmo | Product norms, MuonMax and Momo
- **Lemma 3.3:** for a product norm $\lVert(w_1,\dots,w_n)\rVert=f(\lVert w_1\rVert_{(1)},\dots,\lVert w_n\rVert_{(n)})$ the LMO and dual norm follow layer by layer. **Prop. 3.6:** MuonAdam (spectral norm on hidden matrices, an adaptive norm elsewhere, max-combined) is exactly CSD in such a norm; Props. 3.4–3.5: Adam itself is CSD and RSD in suitable norms.
- **MuonMax:** RSD in a hybrid product norm; each matrix moves by $-\eta\big(\sum_j\lVert M^j_t\rVert_{\text{nuc}}\big)\,\mathrm{polar}(M^i_t)$ (schematic), using stale dual norms from the previous step to save a pass (about 5% overhead).
- **Momo (Props. 4.1–4.3):** truncate the linear model at a loss lower bound $F_*$; the step becomes $\min\{\eta,(\tilde F_t-F_*)/\lVert m_t\rVert_*\}\,\mathrm{LMO}(m_t)$ (constrained version). Reported: Muon is sensitive to the learning rate; MuonMax and especially MuonMax-Momo are far more robust in GPT-2-scale experiments.
### Notes' example
$G=\mathrm{diag}(3,1)$: Frobenius unit step $-G/\sqrt{10}$ gives $\langle G,D\rangle=-\sqrt{10}$; spectral step $-I$ gives $-4=-\lVert G\rVert_{\text{nuc}}$.
> Check script: $-\mathrm{polar}(M)$ beats every one of 40 000 random spectral-ball candidates; $\langle M,\mathrm{polar}(M)\rangle$ equals the sum of singular values.
