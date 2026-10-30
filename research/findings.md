# Verified findings per paper (source of truth for visuals)

Each entry: what the paper actually proves (checked against the PDF text), the toy models used on its page, and what the picture simplifies. Page/equation numbers refer to the official PDF.

## baldi1989neural — Baldi & Hornik, Neural Networks 2(1) 1989
- Main result (abstract, Sec. 2): for a linear net with one hidden layer of p units and squared error E(A,B)=Σ‖y−ABx‖², E has, up to equivalence, a unique local and global minimum = orthogonal projection onto the span of the top-p eigenvectors of Σ (Σ = Σ_YX Σ_XX⁻¹ Σ_XY; auto-associative: Σ = Σ_XX). All other critical points are saddles (other p-index sets of eigenvectors, and rank-deficient points).
- Facts 1–2: E is convex in B for fixed A (and in A for fixed B); strictly convex when full rank.
- Non-uniqueness: AB = (AC)(C⁻¹B) for any invertible C — uniqueness only for the map W = AB.
- Assumptions: Σ_XX invertible, Σ full rank with distinct eigenvalues, no bias, single hidden layer (claims to hold for several hidden layers, conjectured).
- Notes example: Σ = diag(9,1), p=1, E(θ)=1+8 sin²θ.
- Visual toy models:
  - Tied-weight autoencoder slice B = Aᵀ = wᵀ, w∈R²: E(w)=tr Σ − 2wᵀΣw + ‖w‖² wᵀΣw. Critical points: 0 (max on slice; saddle in full space), ±u₁ (global min, E=λ₂=1), ±u₂ (saddle, E=λ₁=9).
  - Untied slice A=(a,0)ᵀ, B=(b,0): E = 9(1−ab)² + 1 → hyperbola ab=1 of equivalent global minima, floor = discarded variance 1.
  - 2D: projection angle θ, E(θ)=1+8 sin²θ.

## kawaguchi2016deep — Kawaguchi, NeurIPS 2016
- Theorem 2.3: assume XXᵀ, XYᵀ full rank, d_y ≤ d_x, Σ has d_y distinct eigenvalues. For any depth H≥1 and any widths: (i) non-convex & non-concave, (ii) every local min is global, (iii) every other critical point is a saddle, (iv) if rank(W_H⋯W_2)=p then the Hessian at any saddle has a strictly negative eigenvalue.
- Corollary 2.4: H=1 → every saddle has a negative Hessian eigenvalue; H≥2 → there exist saddles with no negative eigenvalue ("bad saddles").
- Corollary 3.2: same 4 statements for nonlinear nets under the independence assumption (unrealistic, acknowledged).
- Visual = reference/saddles.html: strict L=(1−w₁w₂)², H(0)=[[0,−2],[−2,0]]; bad L=(1−w₁w₂²)² (slice w₃=w₂ of a 2-hidden-layer scalar chain), H(0)=0.

## saxe2014exact — Saxe, McClelland, Ganguli, ICLR 2014
- Eq. (8)–(9): each decoupled mode follows gradient flow on E(a,b)=½(s−ab)²; τ da/dt = b(s−ab), τ db/dt = a(s−ab).
- Conserved quantity a²−b² (scaling symmetry, Noether) → trajectories follow hyperbolas a²−b²=const until the fixed-point manifold ab=s. Origin is an unstable fixed point.
- Eq. (10)–(12): with a=b, u=ab: τ du/dt = 2u(s−u); u(t) = s e^{2st/τ} / (e^{2st/τ} − 1 + s/u₀). Learning time t ≈ (τ/s) ln(s/ε): stronger modes learned first; sigmoidal plateau → rapid transition.
- Eq. (14)–(15) deep: E=½(s−∏aᵢ)², τ du/dt = (N_l−1) u^{2−2/(N_l−1)} (s−u).
- Fig. 6 / Sec. 3: products of scaled Gaussian matrices (std 1/√N) preserve norm on average but singular values become highly kurtotic with depth (most tiny, few huge); products of orthogonal matrices keep all singular values = 1 → depth-independent learning times (empirical on MNIST, theoretical in the linear model).
- Visual toy models: (1) surface ½(s−ab)², s=1, GD from small random init, conserved a²−b² curves; (2) analytic sigmoid curves from Eq. (12) for modes s=3,2,1,0.5 + simulated discrete GD; (3) live random-matrix experiment: singular values of product of D random N×N Gaussian (std 1/√N) vs orthogonal matrices.

## laurent2018deep — Laurent & von Brecht, ICML 2018
- Theorem 1: deep linear net L(W₁..W_L)=Σℓ(W_L⋯W₁x,y) with (i) ℓ convex and differentiable, (ii) thinnest layer = input or output layer (min dₖ ≥ min(d₀,d_L)) → all local minima are global.
- Theorem 3 (the bridge): if f is any differentiable function of A and the width condition holds, then at any local minimizer Ŵ of F(W)=f(W_L⋯W₁), ∇f(Â)=0. Convexity of f then gives global optimality.
- Theorem 2 / Eq. (22)–(23): differentiability is necessary. f(x,y)=|x|+(1−y)₊−1 (convex, Lipschitz, not differentiable), W₁∈R, W₂∈R². Ŵ₁=0, Ŵ₂=(1,0)ᵀ is a sub-optimal local minimizer (f=0 vs f_opt=−1).
- Remark: critical points ≠ local minima: for L≥3 the all-zero point is always critical.
- Visual toy models: (1) scalar chain F(w₁,w₂)=ℓ(w₂w₁−2) with ℓ ∈ {squared, log-cosh} — both convex+differentiable → only global valley + saddle at 0; (2) 2D slice of the paper's counterexample: W₁=α, W₂=(1−β, β): F(α,β)=|α(1−β)|+(1−αβ)₊−1; bad local min along α=0 near β=0 (F=0), global min ray β=1, α≥1 (F=−1). Subgradient descent.

## ghadimi2013stochastic — Ghadimi & Lan, SIAM J. Optim. 2013 (arXiv 1309.5549)
- RSG method: x_{k+1}=x_k−γ_k G(x_k,ξ_k); output x_R with R random, P(R=k) ∝ 2γ_k−Lγ_k² (Eq. 2.3).
- Assumption A1: f L-smooth, bounded below; E[G]=∇f, E‖G−∇f‖² ≤ σ².
- Theorem 2.1(a): (1/L)E‖∇f(x_R)‖² ≤ [D_f² + σ² Σγ_k²] / Σ(2γ_k − Lγ_k²), D_f = [2(f(x₁)−f*)/L]^{1/2}.
- Corollary 2.2: γ_k = min{1/L, D̃/(σ√N)} ⇒ (1/L)E‖∇f(x_R)‖² ≤ B_N = L D_f²/N + (D̃ + D_f²/D̃) σ/√N. → O(1/√N) with noise.
- Sec. 3 RSGF (zeroth-order): Gaussian smoothing f_μ(x)=E_u f(x+μu); estimator G_μ = [F(x+μu,ξ)−F(x,ξ)]/μ · u, u~N(0,I) (Eq. 3.12), unbiased for ∇f_μ.
- Only first-order stationarity: small gradient can be a saddle or max.
- Visual toy models: smooth bounded-below 2D landscape (quadratic + Gaussian wells/bump: globally L-smooth); SGD with Gaussian gradient noise (σ² = 2s²); chart of empirical mean ‖∇f(x_R)‖² over runs vs N against L·B_N; ‖∇f‖ heat map showing the small-gradient set contains saddle and max; RSGF ball.

## lee2016gradient — Lee, Simchowitz, Jordan, Recht, COLT 2016
- Def. 2 strict saddle: λ_min(∇²f(x*))<0 (zero eigenvalues allowed; includes maxima).
- Theorem 4: f∈C², ∇f L-Lipschitz, x* strict saddle, 0<α<1/L, init absolutely continuous ⇒ Pr(lim x_k = x*) = 0.
- Proposition 8: g(x)=x−α∇f(x) is a diffeomorphism for α<1/L. Corollary 9: countably many saddles, all strict ⇒ Pr(converge to a saddle)=0. Theorem 11: + lim x_k exists ⇒ limit is a local minimizer a.s.
- Section 3 example (Nesterov 2004 §1.2.3): f(x,y)=½x²+¼y⁴−½y²; critical points (0,0) saddle, (0,±1) minima; stable set of the saddle = x-axis; ∇²f=diag(1, 3y²−1).
- No finite-time escape bound (asymptotic only).
- Visual toy models: the paper's function; random starts; one start on the x-axis; escape-time chart (iterations ≈ ln(c/|y₀|)/ln(1+α), unbounded as y₀→0); GD-map grid warp with det Dg=(1−α)((1+α)−3αy²) to show invertibility for small α and folding for large α.

## dauphin2014identifying — Dauphin et al., NeurIPS 2014
- Sec. 2: Bray & Dean (2007) random Gaussian fields: critical points concentrate on a monotonically increasing curve in the (index α, error ε) plane; Hessian spectrum at a critical point = Wigner semicircle shifted by an amount set by ε; for large N most critical points are saddles. Fig. 1: same qualitative trend measured in small MLPs on downsampled MNIST/CIFAR-10.
- Sec. 4 / Fig. 2: GD step along eigvec i is −λᵢΔvᵢ (slow when |λᵢ| small); Newton step −Δvᵢ moves toward the saddle along negative-curvature directions (saddle is an attractor for Newton). Fig. 2 uses (a) 5x²−y² and (b) monkey saddle x³−3xy².
- Eq. (4)–(5): generalized trust region |sᵀHs| ≤ sᵀ|H|s ⇒ Δθ = −∇f |H|⁻¹ (saddle-free Newton). |H| = Q|Λ|Qᵀ. Algorithm 1: Krylov subspace (Lanczos) + damping.
- Empirical only (no finite-time theorem).
- Visual toy models: paper's 5x²−y² and x³−3xy² with GD / damped Newton / damped SFN (|λ|→max(|λ|,δ)); GOE spectrum shifted by m(ε) (illustration, linear shift is our simplification); Monte Carlo P(all eigenvalues of an N×N GOE matrix > 0) vs N.

## jin2017escape — Jin, Ge, Netrapalli, Kakade, Jordan, ICML 2017
- Def. 6: ε-second-order stationary point: ‖∇f‖ ≤ ε and λ_min(∇²f) ≥ −√(ρε). A1: ℓ-smooth gradient, ρ-Lipschitz Hessian.
- Algorithm 2 (PGD): GD with η=c/ℓ; if ‖∇f(x_t)‖ ≤ g_thres and t−t_noise > t_thres, add ξ uniform in ball B₀(r); if after t_thres steps f decreased by less than f_thres, return the pre-perturbation point.
- Theorem 3: with prob ≥ 1−δ, PGD outputs an ε-SOSP in O(ℓ(f(x₀)−f*)/ε² · log⁴(dℓΔ_f/(ε²δ))) iterations — same as GD's first-order rate up to polylog(d).
- Sec. 5 / Figs. 1–2: perturbations that fail form a thin "pancake" (3D) / "narrow band" (2D) around the stable direction; two points separated along e₁ by enough cannot both be stuck (Lemma 11) ⇒ small volume.
- Corollary 4 (strict-saddle A2) ⇒ close to local minima; Sec. 4 matrix factorization ½‖UUᵀ−M‖²_F.
- Visual toy models: notes' f=½x²+cos y−1 (ℓ=ρ=1, bounded below, minima at y=±π); GD stuck on the stable set vs PGD kick; stuck region computed live in 2D/3D (with a bent variant); escape iterations vs dimension d.

## li2018visualizing — Li, Xu, Taylor, Studer, Goldstein, NeurIPS 2018
- Method: 2D slice f(α,β)=L(θ*+αδ+βη) with random Gaussian directions, filter-normalized: δ_{i,j} ← δ_{i,j}/‖δ_{i,j}‖·‖θ_{i,j}‖ (per filter / per neuron). 1D interpolation f(α)=L((1−α)θ+αθ′) is shown to be misleading.
- Scale invariance (ReLU / BN): rescaling a layer by c and the next by 1/c leaves the function unchanged but changes naive sharpness; filter normalization removes this artifact. Fig. 2: weight decay can flip which of small-batch/large-batch looks sharper under naive 1D plots.
- Fig. 5 test errors (CIFAR-10): ResNet-20/56/110 = 7.37/5.89/5.79 %; no-shortcut = 8.18/13.31/16.44 %. Fig. 6 Wide-ResNet-56 k=1,2,4,8: with shortcuts 5.89/5.07/4.34/3.93 %, without 13.31/10.26/9.69/8.70 %.
- Fig. 7: |λ_min/λ_max| maps; convex-looking regions have small relative negative curvature. Caveat (Sec. 6): a convex-looking 2D slice does not imply convexity.
- Empirical only.
- Visual toy models: a real 2-16-1 ReLU network trained (offline, Python) on a small 2D dataset; live 2D slices with and without filter normalization under the rescaling (c·W₁, W₂/c) (exactly invariant only with normalization); notes' hidden saddle ½(x²+y²−2z²) with a rotatable slice; bar chart of reported test errors.

## du2019gradient — Du, Lee, Li, Wang, Zhai, ICML 2019
- Setting: n samples, ‖xᵢ‖=1, |yᵢ|=O(1), squared loss, smooth Lipschitz analytic non-polynomial activation (softplus is the guiding example), Gaussian init, full-batch GD.
- Theorem 5.1 (deep FC): width m = 2^{O(H)} · poly(n, 1/λ_min(K^{(H)}), log(Hn/δ)) and step η = O(λ_min(K^{(H)}) / (n² 2^{O(H)})) ⇒ w.p. ≥ 1−δ, L(θ(k)) ≤ (1 − ηλ_min(K^{(H)})/2)^k L(θ(0)).
- Theorem 6.1 (ResNet, branch scaling c_res/H): width polynomial in n, H (m ≳ n⁴H⁶/λ_min⁴ terms) and η = O(λ_min H²/n²) give the same geometric rate. Theorem 7.1: conv ResNet, poly(n, p, H).
- Mechanism: Gram matrix G(k) = J Jᵀ of prediction gradients stays close to its initial value because weights move little when m is large; residuals u−y contract like (I − ηG).
- Footnote 7: exponential dependence for FC nets is from their analysis, not necessarily real.
- Visual toy models: one-hidden-layer softplus net, inputs x=(cos t, sin t, 1)/√2 (‖x‖=1; constant coordinate acts as a bias — 6 points on a plain circle gave λ₀≈0), NTK scaling, full-batch GD, η=1/λ_max(G₀). Checked: wide nets stay under (1−ηλ₀/2)^k L(0), λ_min(G_k) stays within 0.1% of λ₀ at m=1024, weights move 0.5% at m=8192 vs 28% at m=8; n=2 residual path matches the linear prediction (I−ηG₀)^k r₀ to 0.03% at m=4096 vs 14% at m=4 (the path is NOT straight — G is anisotropic). Dropped idea: a 2^H vs H⁶ width-bound chart (misleading with constants dropped).

## allenzhu2019convergence — Allen-Zhu, Li, Song, ICML 2019
- Setting: L-layer fully connected ReLU net, fixed random input map A and output map B, trained hidden matrices W₁..W_L, Gaussian init, inputs normalized with pairwise separation ‖xᵢ−xⱼ‖ ≥ δ, squared loss.
- Theorem 1 (GD): m ≥ poly(n, L, δ⁻¹)·d, η = Θ(dδ/(poly(n,L) m)) ⇒ F(W(T)) ≤ ε for T = poly(n,L)/δ² · log(1/ε). Theorem 2: same for minibatch SGD (log(1/ε) dependence kept).
- Theorem 3 (no critical point): w.h.p., for every W within distance ω ≤ 1/poly(n,L,δ⁻¹) of init, ‖∇F(W)‖²_F ≥ Ω(δ m/(d n²)) · F(W) (gradient lower bound, PL-like, local). Theorem 4: semi-smoothness (descent lemma with an extra first-order error term from ReLU sign changes).
- Fig. 1: 2D CIFAR-10 loss slices along the negative gradient and a negative-curvature direction at points on the SGD path: negative curvature exists, but the gradient direction gives large decrease.
- Theorems 5–7: other losses, CNN, ResNet.
- Visual toy models: deep ReLU nets (fixed A, B; trained W's) trained offline in Python: loss curves vs width, ratio ‖∇F‖²/F along the path, separation δ experiment (two inputs with opposite labels at distance δ), and a real 2D slice along (−∇F, v_min) at a point on the path (Fig. 1 analogue).

## safran2018spurious — Safran & Shamir, ICML 2018
- Notation (paper): teacher width k with orthonormal v₁..v_k ∈ R^k, student width n ≥ k, x ~ N(0,I). Eq. (2): F(w) = ½ E_x[(Σᵢⁿ[wᵢᵀx]₊ − Σⱼᵏ[vⱼᵀx]₊)²]. Global min 0 at wᵢ=vᵢ (rest 0).
- Closed form (Sec. 4.2, arc-cosine kernel): E[[wᵀx]₊[vᵀx]₊] = ‖w‖‖v‖/(2π)·(sin θ + (π−θ) cos θ).
- Theorem 1: spurious local minima exist for k=n ∈ {6..20} and for (k,n) = (8,9) and (k,k+1), 10 ≤ k ≤ 19. Corollary 1: extends to d > k; holds w.p. ≥ 1−exp(−Ω(d)) for Gaussian teacher vectors.
- Lemmas 1–2: computer-assisted certificate (small gradient, λ_min > 0, bounded third derivative ⇒ true local min nearby; Lipschitz bound ⇒ non-global).
- Experiments: 1000 GD runs per (k,n), step 0.1, Xavier init wᵢ ~ N(0, I/k), stop when gradient norm ≤ 1e−9, ignore F < 1e−3. Table 1 (k=n): 6: 0.3 %, 7: 5.5 %, 8: 12.6 %, 9: 21.8 %, 10: 34.6 %, 12: 58.5 %, 15: 80.3 %, 20: 94 %; average spurious F ≈ 0.02–0.03, λ_min ≈ 0.005–0.04. Mild over-parameterization (n ≥ k+2) found no spurious minima up to 20 (observed, not proved).
- Visual toy models: exact closed-form population loss; offline replication of the GD experiment (scripts/check-math/safran2018spurious.py) for several (k,n); a real 2D slice through a found spurious minimum and the nearest global minimum. Replication: k=n=6: 0/600, 8: 10.2 %, 10: 36.3 %, 12: 50.0 %; (10,11) and (10,12): 0 %; spurious F ≈ 0.02, λ_min ≈ 0.02.

## wu2025loss — Wu, Simsek, Ged, ICLR 2025
- Model (Eq. 1–2): ŷ = H σ(Wx), σ(z)=α₊z (z≥0), α₋z (z<0), α₊≠α₋ (ReLU: α₊=1, α₋=0), empirical squared loss. Non-differentiable only where wᵢ ⊥ some xₖ.
- Def. 3.5 directional stationarity (all one-sided directional derivatives ≥ 0). Def. 4.1 escape neuron (tangential direction with zero ODD but nonzero coefficient). Theorem 4.2: no escape neuron ⇒ local min; for scalar output (|J|=1) also necessary. Corollary 4.4: scalar-output non-minimum stationary point has a direction with L(P+εΔ)−L(P) ≤ Cε², C<0 (second-order decrease).
- Fact 4.5: under gradient flow ‖wᵢ‖² − h²ᵢ is conserved ⇒ with vanishing init ‖wᵢ‖ = |hᵢ|. Cor. 4.6: reachable non-minimum stationary points have a neuron with wᵢ=0, hᵢ=0; escape = growth of small living neurons (Fact 4.7). Prop. 4.8: unit replication preserves stationarity / type-1 minima only under sign conditions.
- Sec. 4.1.1 / Fig. 3 experiment: 50 hidden ReLU neurons, inputs xₖ=(xₖ,1) with xₖ ∈ {−1,−0.6,−0.1,0.3,0.7}, targets {0.28,−0.1,0.03,0.23,−0.22}, all params ~ N(0,(5·10⁻⁶)²), GD step 0.001, 500k epochs; saddle escapes near epochs 42k and 152k; final plateau = local min; dead neurons have angles in (3.93, 5.67) rad; group-1 neurons stuck at ≈2.53 rad (⊥ x₅).
- Visual toy models: one-neuron L(a,w)=½(a·ReLU(w)−1)² (dead half-plane vs escapable kink); offline replication of the Fig. 3 experiment (loss curve, neuron angles, ‖wᵢ‖ vs |hᵢ|) with a time scrubber. Replication result (our random draw of the same init law): plateaus 0.0953 → 0.0755 → 0.0697 (last = local min), escapes near 25k and ~65k epochs (paper: 42k, 152k — timing depends on the draw); group-1 neurons stuck at 2.53 rad and dead neurons exactly in (3.93, 5.67) rad as in the paper.

## kalra2025universal — Kalra, He, Barkeshli, ICLR 2025
- UV model (Sec. 2): f(x)=vᵀUx/√n_eff, n_eff=n^{1−p}, one example (x,y), MSE ½(f−y)², GD. λ := Tr H = (‖Ux‖² + ‖x‖²‖v‖²)/n_eff (also the scalar NTK); proxy for sharpness (top eigenvalue verified to follow it, App. C.5).
- Eq. (1): f̃_{t+1} = f̃_t(1 − ηλ_t + (η²‖x‖²/n_eff) f̃_t(f̃_t+y)); Eq. (2): λ_{t+1} = λ_t + (η‖x‖² f̃_t/n_eff)(ηλ_t f̃_t − 4(f̃_t+y)), f̃ = residual. (Eq. 1 re-derived by hand from the parameter updates; Eq. 2 checked numerically.)
- Fixed points: I zero-loss line (stable iff λ<2/η), II (−y,0) = origin in parameter space, a saddle; III, IV unstable, function-space only. Sharpening vs reduction regions separated by ηλf̃ = 4(f̃+y).
- Result 1: necessary condition for EoS: η > η_c = √n_eff/(‖x‖y). Corollary 5.1: if η_max < η_c, no EoS. Fig. 3: x=1,y=2,n_eff=1, η_c=0.5, period-doubling route to chaos, diverges ≈ 0.8.
- Visual toy models: n=1, d=1 UV model L(u,v)=½(uv−2)² (η_c=0.5): GD ball settles below / bounces above η_c; live (f̃,λ) map with nullclines; bifurcation diagram from Eqs. (1)–(2); full-width UV simulation for the four regimes.

## zhang2026saddle — Zhang, Saxe, Latham, ICLR 2026
- Eq. (1): f(x;θ) = Σᵢ g_out(g_in(x); uᵢ) vᵢ (additive units; covers FC, conv, attention heads).
- Theorem 1 (embedded fixed points): a fixed point of a width-(H−1) net embeds as a fixed point of width H (via zero unit, duplication with split v, or u-scaling, depending on activation properties). Corollary 2: extra units/layers.
- Theorem 3 (invariant manifolds): e.g. identical units stay identical; zero/proportional/linearly dependent units under activation conditions.
- Eq. (8)–(10) linear case f = Σ vᵢuᵢᵀz(x): small init ⇒ v̇=Σ_yz u, u̇=Σ_yzᵀv; Theorem 4: when the top-r singular subspace reaches O(1), the rest is O(σ^{1−s_{r+1}/s₁}) ⇒ data-induced timescale separation (low rank).
- Eq. (13)–(14) quadratic case f = Σ vᵢ uᵢᵀ Z(x) uᵢ (e.g. linear self-attention): Prop. 5 ⇒ initialization-induced separation between units (rich-get-richer, v̇=v² ⇒ v(t)=1/(1/v(0)−t)).
- Fig. 2: width barely changes linear-net loss curves but shortens plateaus in linear self-attention; power-law exponent θ=0 removes plateaus in linear nets but not in self-attention; larger init weakens plateaus.
- Visual toy models: exact loss on the balanced, aligned invariant manifold of a two-layer linear net, L(a₁,a₂)=½(s₁−a₁²)²+½(s₂−a₂²)² (saddles at 0 and (√s₁,0), minimum (√s₁,√s₂)); full simulations of a linear two-layer net and a quadratic-in-weights net from small init (unit norms vs time); first-layer weights of a small ReLU net over training.

## galli2026flatland — Galli, Fox, Bartolomaeus, Schmidt, Rauhut, ICML 2026 (arXiv 2606.06722)
- Local segment smoothness ℓ(w,η)=sup_{0<t≤η} ‖∇f(w)−∇f(w−t∇f(w))‖/(t‖∇f(w)‖). Def. 2.3 large steps: η_k ≥ 1/ℓ(w_k,η_k) and within P steps η > 2/ℓ.
- Nonmonotone acceptance (Eq. 2): f(w_k−η∇f) ≤ R_k − cη‖∇f‖², Zhang–Hager reference R_k = (ξQ_{k−1}R_{k−1} + f(w_k))/Q_k, Q_k = ξQ_{k−1}+1. Equality line search (Alg. 2): halve until accepted, double while accepted (from the previous step) ⇒ step near the equality boundary.
- Prop. 2.5: η_k ≥ 2(1−c)/ℓ(w_k, 2η_k). Theorem 2.9: f ∈ C¹, bounded below, compact level set ⇒ lim ‖∇f(w_k)‖ = 0 (first-order only).
- Fig. 1: η_k·λ_max ≈ 2 or above from early training (EoS). Sec. 3: MSE with K outputs and final bias ⇒ bias block of the Hessian = (2/K)I ⇒ λ_max ≥ 2/K globally; some runs hit sharpness 2/K early with near-random loss (flat, almost stationary saddle-like points with tiny hidden-layer gradients). NLS-ub caps the step below K and trains better in their experiments (5k-image subsets).
- Visual toy models: 2D curved valley ¼(x²−1)²+3(y−0.3x²)² with monotone vs nonmonotone (Zhang–Hager, ξ=0.85) equality line search, c=1e−4, β=0.5: NLS median η·λ_max = 2.2, 198 accepted loss increases, still ‖∇f‖→0. Tiny tanh MSE net (2-4-K): λ_max ≥ 2/K at random weights; the 'Flatland' point (hidden weights 0, output bias = class means) is exactly stationary with top-K eigenvalues = 2/K and negative eigenvalues (checked K=2,4,8).

# Round 2 — survey references with new notes (2026-10-30)

## murty1987some — Murty & Kabadi, Math. Programming 39 (1987) [survey 2]
- Source read: author scan public.websites.umich.edu/~murty/np.pdf (pp. 117–120 read as images) + Ahmadi & Zhang arXiv 2008.12170 restatement.
- Abstract: for smooth non-convex NLP, analyze complexity of checking (a) a given feasible solution is NOT a local minimum, (b) the objective is NOT bounded below on the feasible set. For a special class of indefinite QPs with simple constraints and integer data, (a) and (b) are NP-complete. Corollary: checking that an integer square matrix is NOT copositive is NP-complete.
- Sec. 2 Example 2 (Subset Sum): min (Σdⱼyⱼ − d₀)² + Σyⱼ(1−yⱼ) on 0 ≤ yⱼ ≤ 1 has global value 0 iff the subset-sum instance is solvable ⇒ computing a global min of a smooth non-convex QP is NP-hard.
- Via Ahmadi–Zhang: origin is a local min of xᵀQx on x ≥ 0 iff Q copositive; also deciding local minimality of a quartic polynomial is NP-hard (f(u) = Σ Qᵢⱼuᵢ²uⱼ²: origin local min iff Q copositive; Hessian at 0 is zero). So the survey's "deciding whether a point is a local minimum of a general smooth function is NP-hard" is correct.
- Visual toy models: 2×2 Q = [[1,q],[q,1]] (copositive iff q ≥ −1; PSD iff |q| ≤ 1) on the orthant; unconstrained quartic u₁⁴ + 2q u₁²u₂² + u₂⁴ (zero Hessian at 0); Example 2 with d = (3,5), d₀ ∈ {3,4,5,8} (4 unsolvable).

## blum1992training — Blum & Rivest, Neural Networks 5(1) 1992 (NIPS 1988 version) [survey 3]
- Source read: author reprint people.csail.mit.edu/rivest/pubs/BR93.pdf.
- 3-Node Network = n binary inputs, TWO hidden linear threshold nodes N1, N2, ONE threshold output node N3. Theorem 2: training (deciding whether weights exist consistent with all examples) is NP-complete. ⚠ Survey Sec. I says "three hidden nodes" — wrong; it is 2 hidden + 1 output.
- Geometric view (4.3.1): equivalent to: one plane separating ±, or two planes with one quadrant containing all and only '+' (or all and only '−'). Restricted "Quadrant of Positive Boolean Examples" (AND output) is NP-complete (Theorem 3) by reduction from Set-Splitting.
- Reduction: origin '+'; '−' at unit vectors pᵢ for each sᵢ; '+' at Σ_{j∈cᵢ} pⱼ for each subset cᵢ. Planes from a split: P1: a·x = −½ with aᵢ = −1 if sᵢ∈S1 else n; P2 likewise with S2; quadrant {a·x > −½, b·x > −½} holds all '+' and no '−'. Paper's example: S = {s1,s2,s3}, C = {{s1,s2},{s2,s3}}: '−' at 001, 010, 100; '+' at 000, 110, 011.
- Also NP-complete: equal-weight hidden nodes / ±1 weights; k hidden nodes with AND output; XOR output with ternary features. Positive result: anything the 3-node XOR-output net classifies is linearly separable using the n inputs + all pairwise products (perceptron with O(n²) inputs, trainable by LP). Single perceptron: polynomial via LP.
- Visual toy models: the paper's 6-point cube example with the two planes for any user-chosen split (valid splits work, invalid fail); 2D intuition of the hidden-unit choice; XOR lifted by the product feature x₁x₂ (the paper's representation remark).

## jin2018accelerated — Jin, Netrapalli, Jordan, COLT 2018 (arXiv 1711.10456) [survey 10]
- Algorithm 1 (AGD): y_t = x_t + (1−θ)v_t; x_{t+1} = y_t − η∇f(y_t); v_{t+1} = x_{t+1} − x_t.
- Algorithm 2 (PAGD) adds (i) perturbation ξ ~ Unif(B₀(r)) when ‖∇f(x_t)‖ ≤ ε and no kick in the last T steps, (ii) Negative Curvature Exploitation (Alg. 3) when f(x_t) ≤ f(y_t) + ⟨∇f(y_t), x_t−y_t⟩ − (γ/2)‖x_t−y_t‖² (Eq. 2): if ‖v_t‖ ≥ s keep x_t, else move ±s·v_t/‖v_t‖ to the better side; momentum reset to 0.
- Parameters (Eq. 3): η = 1/(4ℓ), θ = 1/(4√κ), γ = θ²/η, s = γ/(4ρ), κ = ℓ/√(ρε).
- Theorem 3: w.p. ≥ 1−δ an ε-SOSP within Õ(ℓ^{1/2}ρ^{1/4}(f(x₀)−f*)/ε^{7/4}) iterations (log⁶ factor), vs Õ(ε⁻²) for PGD. First Hessian-free and single-loop method faster than GD for SOSP.
- Lemma 4: Hamiltonian E_t = f(x_t) + ‖v_t‖²/(2η) decreases monotonically when (2) fails (θ ∈ [2ηγ, ½], η ≤ 1/(4ℓ)); Lemma 5: NCE also decreases it. f alone can increase. "Improve or localize" framework.
- Visual: escape from a strict saddle with weak negative curvature −λ: GD grows the unstable coordinate by (1+ηλ) per step, AGD by ≈ 1 + √(ηλ)·O(1) (momentum root) ⇒ escape time ∝ 1/(ηλ) vs 1/√(ηλ); Hamiltonian vs f along PAGD.

## fang2018spider — Fang, Li, Lin, Zhang, NeurIPS 2018 (arXiv 1807.01695) [survey 11]
- SPIDER estimator: every q steps v_k = ∇f_{S1}(x_k) (full gradient in finite sum), else v_k = ∇f_{S2}(x_k) − ∇f_{S2}(x_{k−1}) + v_{k−1} (same minibatch at both points).
- Lemma 1: if ‖x_k − x_{k−1}‖ ≤ ε₁ and E‖B_i(x)−B_i(y)‖² ≤ L²‖x−y‖², then E‖V_k − B(x_k)‖² ≤ k L² ε₁²/S₂ + E‖V₀ − B(x₀)‖².
- SPIDER-SFO: normalized step x_{k+1} = x_k − η v_k/‖v_k‖ (η = ε/(L n₀)) [Option I] or ηₖ = min(ε/(Ln₀‖v_k‖), 1/(2Ln₀)) [Option II].
- Rates: ε-FOSP in O(min(n^{1/2}ε⁻², ε⁻³)) stochastic gradients (finite-sum: O(n + n^{1/2}ε⁻²), Theorem 2; online O(ε⁻³), Theorem 1); SPIDER-SFO+ (ε, O(√ε))-SOSP in Õ(min(n^{1/2}ε⁻² + ε⁻²·⁵, ε⁻³)). Theorem 3: finite-sum lower bound Ω(L Δ n^{1/2} ε⁻²) when n ≤ O(Δ²L²ε⁻⁴) ⇒ near-optimal. Zeroth-order: O(d·min(n^{1/2}ε⁻², ε⁻³)).
- Visual: real finite sum f = (1/n)Σ fᵢ on a 2D non-convex landscape; SGD vs SPIDER estimator error vs iteration (resets at refresh); Monte Carlo of Lemma 1's linear growth; rate orders vs n.

## arjevani2020second — Arjevani, Carmon, Duchi, Foster, Sekhari, Sridharan, COLT 2020 (arXiv 2006.13476) [survey 12]
- ε-stationary points with O(ε⁻³) stochastic gradients + stochastic HVPs (single-point oracle; earlier O(ε⁻³) methods needed two-point same-seed queries); Ω(ε⁻³) lower bound, also for any stochastic p-th order method p ≥ 2 ⇒ "elbow effect" (Fig. 1): stochastic p=1: ε⁻⁴, p≥2: ε⁻³; noiseless: ε^{−(p+1)/p} (ε⁻² at p=1 → ε⁻¹).
- HVP-RVR (Alg. 1): with prob. b reset g = mean of n stochastic gradients; else g = g_prev + Σ_{k=1}^{K} ∇²F(x⁽ᵏ⁻¹⁾, z⁽ᵏ⁾)(x⁽ᵏ⁾ − x⁽ᵏ⁻¹⁾), x⁽ᵏ⁾ = (k/K)x + (1−k/K)x_prev, K ∝ ‖x − x_prev‖²/(bε²) (biased but controlled). Alg. 2: SGD with η = 1/(2L₁) and this estimator, output a uniformly random iterate.
- (ε,γ)-SOSP: upper O(ε⁻³ + ε⁻²γ⁻² + γ⁻⁵), lower Ω(ε⁻³ + γ⁻⁵). Footnote 1 cites Murty–Kabadi: NP-hard to decide whether an SOSP is a local min or a higher-order saddle.
- Visual: elbow chart (exact exponents); HVP-integral vs two-point difference estimator error vs K on a real 2D function with noisy Hessians; SGD vs SGD+HVP-RVR on a noisy landscape.

## allenzhu2018neon2 — Allen-Zhu & Li, NeurIPS 2018 (arXiv 1711.06673) [survey 33]
- Reduction: (1) turns a stationary-point finder into an approximate-local-minimum finder, (2) replaces Hessian-vector products by gradients, stochastic and deterministic, "without hurting performance"; applies to Natasha2, SGD, GD, SCSG, SVRG.
- Neon2online weak (Alg. 2): x₁ = x₀ + ξ (Gaussian, tiny norm σ); x_{t+1} = x_t − η(∇f_i(x_t) − ∇f_i(x₀)) with i uniform (same index at both points); stop and return v = (x_{t+1}−x₀)/‖·‖ once ‖x_{t+1} − x₀‖ ≥ r, else ⊥ after T steps. ≈ Oja's algorithm on I − η∇²f(x₀) (error O(r²)).
- Theorem 1: w.p. ≥ 1−p: v = ⊥ ⇒ ∇²f(x₀) ⪰ −δI; v ≠ ⊥ ⇒ ‖v‖ = 1 and vᵀ∇²f(x₀)v ≤ −δ/2; Õ(log²(d/p) L²/δ²) stochastic gradients (matches Oja). Lemma 3.2 verification z = (1/m)Σ vᵀ(∇f_{ij}(x+v) − ∇f_{ij}(x)). Neon2det: Chebyshev, Õ(1/√δ) full gradients.
- Visual: same-sample gradient differences on a real finite sum in 3D (iterate x_t − x₀ grows along the negative eigenvector until the radius-r sphere); P(return v) vs λ_min/δ with the "either outcome allowed" band (−δ, −δ/2).

## xu2018first — Xu, Jin, Yang, NeurIPS 2018 (arXiv 1711.01944) [survey 81]
- NEON (Alg. 1): u₀ uniform on sphere radius r; u_{τ+1} = u_τ − η(∇f(x+u_τ) − ∇f(x)) = GD on the centered function f̂ₓ(u) = f(x+u) − f(x) − ∇f(x)ᵀu (Eq. 13); return u_τ minimizing f̂ₓ if min f̂ₓ ≤ −2.5F, else 0. ≈ power method on I − η∇²f(x).
- NEON+ (Alg. 2): Nesterov on f̂ₓ: y_{τ+1} = u_τ − η∇f̂ₓ(u_τ), u_{τ+1} = y_{τ+1} + ζ(y_{τ+1} − y_τ): Lanczos-like iteration count (Theorem 3) vs power-method-like (Theorem 2). IFO O(1/γ³)... combined framework NEON-A (Theorem 4); best time Õ(d/ε^{3.5}) (almost linear in d).
- Visual: exact surface of f̂ₓ at a saddle with the NEON iterate; alignment with e_min vs τ for NEON vs NEON+ and iterations-to-detect vs γ (slopes −1 vs −½) in d = 50.

## reddi2018generic — Reddi, Zaheer, Sra, Póczos, Bach, Salakhutdinov, Smola, AISTATS 2018 [survey 79]
- Algorithm 1: repeat (y_t, z_t) = Gradient-Focused-Optimizer(x_{t−1}); u_t = y_t w.p. p else z_t; (x_{t+1}, τ) = Hessian-Focused-Optimizer(u_t); stop when HFO certifies (τ = ∅).
- G.1 E f(y) ≤ f(x); G.2 E‖∇f(y)‖² ≤ (1/g(n,ε)) E[f(x) − f(z)]. H.1 E f(y) ≤ f(x); H.2 E f(y) ≤ f(x) − h(n,ε,γ) when λ_min(∇²f(x)) ≤ −γ. Theorem 1: rate governed by the slower subroutine; optimal 1/p = 1/(ε²g) + 1/h.
- Instantiation: GFO = nonconvex SVRG (Alg. 2, η = 1/(4Ln^{2/3}), m = n); HFO = HessianDescent (Alg. 3): unit v with vᵀ∇²f v ≤ λ_min + γ/2, α = |vᵀ∇²f v|/M, u = x − α·sign(vᵀ∇f(x))·v, y = argmin{f(u), f(x)}; Lemma 2: h = γ³/(24M²).
- Visual: framework run on ½x² − ½y² + ¼y⁴ from the saddle's stable set (GD alone stalls; HFO steps leave); HessianDescent decrease vs Lemma 2 bound on ½x² − (γ/2)y² + (M/6)|y|³ (Hessian-Lipschitz M; exact decrease γ³/(3M²) with the exact eigenvector).

## tripuraneni2018stochastic — Tripuraneni, Stern, Jin, Regier, Jordan, NeurIPS 2018 (arXiv 1711.02838) [survey 74]
- Alg. 1 (stochastic cubic regularization): each outer step samples S₁ gradients and S₂ Hessians at x_t: g, B; model m(Δ) = gᵀΔ + ½ΔᵀBΔ + (ρ/6)‖Δ‖³; x_{t+1} = x_t + Δ with Δ from Cubic-Subsolver; if the predicted decrease Δm ≥ −(1/100)√(ε³/ρ), call Cubic-Finalsolver (Alg. 2) and stop.
- Cubic-Subsolver (Alg. 3), Hessian-vector products only: if ‖g‖ ≥ ℓ²/ρ take the Cauchy step Δ = −R_c g/‖g‖, R_c = −gᵀBg/(ρ‖g‖²) + √((gᵀBg/(ρ‖g‖²))² + 2‖g‖/ρ); else GD on the model with a perturbed linear term g̃ = g + σζ, step 1/(20ℓ).
- Result: ε-SOSP (‖∇f‖ ≤ ε, λ_min ≥ −√(ρε)) in Õ(ε⁻³·⁵) stochastic gradient + HVP evaluations vs Õ(ε⁻⁴) for perturbed SGD.
- Notes' 1D check reproduces: f = cos s at x = 0 with g = 0.01, B = −0.9, ρ = 1 → r ≈ 1.811043, m ≈ −0.504055, q ≈ −0.649939, true change ≈ −1.237943.
- Visual: sampled vs true cubic model with the subsolver ball (Cauchy vs perturbed GD); outer loop on a real finite-sum landscape vs SGD.

## carmon2018analysis — Carmon & Duchi, NeurIPS 2018 (arXiv 1806.09222) [survey 75]
- Trust-region problem min ½xᵀAx + bᵀx s.t. ‖x‖ ≤ R (also the cubic version, (ρ/3)‖x‖³); Krylov solution s_t = argmin over K_t(A, b) = span{b, Ab, …, A^{t−1}b} (Lanczos).
- Theorem 1: linear rate f(s_t) − f(s*) ≤ 36[f(0) − f(s*)]·exp(−4t√((λ_min + λ*)/(λ_max + λ*))); sublinear rate ∝ (λ_max − λ_min)‖s*‖²/(t − ½)², with an extra log²(‖b‖²/(u_minᵀb)²) term when λ_min < 0. Lower bounds show both are sharp.
- Hard case: u_minᵀb = 0 and λ* = −λ_min, so the Krylov space never reaches u_min. Fix (Corollaries 2–3): perturb b by a small random vector.
- Notes' example reproduces: A = diag(−2, 1), b = (0, 1), R = 1 → the Krylov space span{(0,1)} gives value −1/2; the true minimizers are (±2√2/3, −1/3) with value −7/6.
- Visual: hard-case disk with the Krylov line vs the perturbed fix; Lanczos gap vs t against Theorem 1's two bounds.

## carmon2019gradient — Carmon & Duchi, SIAM J. Optim. 2019 (arXiv 1612.00547) [survey 76]
- f(x) = ½xᵀAx + bᵀx + (ρ/3)‖x‖³ (A indefinite allowed). GD x_{t+1} = x_t − η∇f(x_t) = (I − ηA − ρη‖x_t‖I)x_t − ηb (Eq. 9).
- Assumption A: 0 < η ≤ 1/(4(β + ρR)), β = ‖A‖. Assumption B: x₀ = −r·b/‖b‖ with 0 ≤ r ≤ R_c (Cauchy radius).
- Lemma 2.3: ‖x_t‖ non-decreasing, ‖x_t‖ ≤ R, x_tᵀ∇f(x_t) ≤ 0. Lemma 2.4: in the eigenbasis x_t⁽ⁱ⁾b⁽ⁱ⁾ ≤ 0. Prop. 2.5: if b⁽¹⁾ ≠ 0 then x_t → x* (global min) despite saddles and a bad local min.
- Theorem 3.1: three regimes, roughly O(1/ε), gap-dependent O(1/√(ε·gap)) and O(log 1/ε), plus a growth phase ∝ log(1 + …/|b⁽¹⁾|) that is 0 when the problem is convex. Hard case b⁽¹⁾ = 0: GD stays orthogonal to v₁. Theorem 3.2: GD on b̃ = b + σq (q uniform on the sphere) gives the bound w.h.p. with only log d dependence.
- Paper's Fig. 2: GD stalls near saddles whose norm is close to γ/ρ (weak negative curvature there) and f(x_t) can increase for a while.
- Notes' 1D check reproduces: f = −x²/2 + |x|³/3, η = 0.25, x₀ = 0.2 → 0.24 → 0.2856.
- Visual: 2D instance A = diag(−γ, λ₂): GD from Assumption B; hard case (b⁽¹⁾ = 0) stuck on the axis vs perturbed b̃; ‖x_t‖ monotone chart.

## wang2020cubic — Wang, Zhou, Liang, Lan, UAI 2020 (PMLR 115) [survey 77]
- CRm (Alg. 1): cubic step s_{k+1} = argmin ∇f(x_k)ᵀs + ½sᵀ∇²f(x_k)s + (M/6)‖s‖³, y_{k+1} = x_k + s_{k+1}; momentum β_{k+1} = min{β, ‖∇f(y_{k+1})‖, ‖y_{k+1} − x_k‖} (β < 1), v_{k+1} = y_{k+1} + β_{k+1}(y_{k+1} − y_k); monotone step x_{k+1} = argmin over {y_{k+1}, v_{k+1}} of f.
- Theorem 1: ε-SOSP within O(ε^{−3/2}) iterations (same order as CR, the best possible). Theorem 2: local quadratic convergence under a local error bound. Theorems 3–4: an inexact (subsampled) variant keeps the rate.
- The speed-up over CR is empirical (Section 5), not a better worst-case exponent.
- Notes' checks reproduce: f = −x²/2 + x⁴/4, f(0.8) = −0.2176, f(0.86) ≈ −0.2330.
- Visual: CR vs CRm on a real 2D nonconvex function (exact cubic subproblem solve), β_k and which candidate the monotone step chose; f vs k.

## nguyen2017loss — Nguyen & Hein, ICML 2017 [survey 53]
- Fully connected nets, analytic strictly monotone activations (sigmoid, tanh, softplus; Lemma 3.3), N distinct samples, a C² loss whose critical points are global minima (Assumption 3.2).
- Lemma 3.5: if rank([F_k, 1_N]) = N, W_l has full column rank for l ≥ k+2, and the gradient for layer k+1 vanishes, the point is a global minimum.
- Theorem 3.8: if some layer has n_k ≥ N − 1, every critical point that is non-degenerate on a subset of layers containing k+1 and has full-column-rank W_{k+2..L} is a global minimum (implies a pyramidal tail). Corollary 3.9: non-degenerate local minima. It does NOT exclude degenerate or low-rank suboptimal local minima (the paper says so). Theorem 5.3: linear separability of features for a class-dependent loss.
- Notes' example reproduces: x = −1, 0, 1 with features tanh(x), tanh(2x+1) and a bias → det ≈ −0.98225; readout (3.339, −2.326, 1.772) fits y = (1, 0, 2) exactly.
- Visual: feature rows of [F, 1] in the plane (full rank ⇔ not collinear ⇒ the stationary readout has zero residual); width sweep with GD on a real 1-hidden-layer tanh net with the N − 1 line (empirical, labeled).

## bhojanapalli2016global — Bhojanapalli, Neyshabur, Srebro, NeurIPS 2016 [survey 58]
- PSD matrix sensing y_i = ⟨A_i, X*⟩ + w_i, f(U) = Σ(⟨A_i, UUᵀ⟩ − y_i)², U ∈ R^{n×r}.
- Published NeurIPS version: Theorem 3.1, (4r, δ_{4r})-RIP with δ_{4r} < 1/10 ⇒ every local minimum has ‖UUᵀ − X*‖_F ≤ 20√(log n/m)·σ_w; noiseless ⇒ all local minima global (Theorem 4.1 relaxes to δ_{4r} < 1/5). Theorem 3.2 strict saddle: λ_min((1/m)∇²f(U)) ≤ −(4/5)σ_r(X*) at non-optimal critical points. Theorem 3.3: projected perturbed SGD (Eq. 4) converges from random init in poly time. (arXiv v2 says 2r instead of 4r; we follow the published version, as the notes do.)
- Gaussian measurements give RIP with m = O(nr).
- Notes' example reproduces: X* = diag(1, 0), U = (a, b): f = (a² − 1)² + 2a²b² + b⁴, f(0.1, 0) = 0.9801, ∇²f(0) = diag(−4, 0).
- Visual: n = 2, r = 1 sensing with m random Gaussian measurements: landscape over U ∈ R², GD from many random inits, recovery error vs m.

## ge2017no — Ge, Jin, Zheng, ICML 2017 [survey 60]
- Unified framework for matrix sensing (Theorem 3: (2r, 1/20)-RIP, balancing regularizer ⅛‖UᵀU − VᵀV‖²_F), matrix completion (Theorem 4) and robust PCA (Theorem 5): all local minima are global and the objectives are strict saddle.
- Direction of improvement Δ = U − U*R, R = argmin over orthogonal Z of ‖U − U*Z‖_F (Definition 6). Lemma 6: ‖ΔΔᵀ‖²_F ≤ 2‖M − M*‖²_F. Lemma 7 (main): Δ:∇²f(U):Δ = ΔΔᵀ:H:ΔΔᵀ − 3(M − M*):H:(M − M*) + 4⟨∇f(U), Δ⟩ + [Δ:∇²Q:Δ − 4⟨∇Q, Δ⟩].
- With H = I, Q = 0 at a stationary point: Δ:∇²f:Δ = ‖ΔΔᵀ‖² − 3‖M − M*‖² ≤ −‖M − M*‖², so every non-optimal stationary point is a saddle and Δ is an escape direction.
- Notes' example reproduces: f = ½‖uuᵀ − diag(1,0)‖² = ½(x² − 1)² + x²y² + ½y⁴, ∇²f(0) = diag(−2, 0), f(0.5, 0) = 0.28125.
- Visual: symmetric d = 2, r = 1 landscape with the aligned Δ arrow at any point; Lemma 7 identity checked live (both sides); stationary points where Δ's curvature ≤ −‖M − M*‖².

## daneshmand2018escaping — Daneshmand, Kohler, Lucchi, Hofmann, ICML 2018 [survey 69]
- CNC (Assumption 1): ∃γ > 0 ∀w: E[⟨v_w, ∇f_z(w)⟩²] > γ, with v_w the eigenvector of λ_min(∇²f(w)). Isotropic noise of radius r puts only r²/d along any fixed direction.
- Alg. 1 CNC-PGD: GD steps; if ‖∇f‖² ≤ g_thres and t − t_noise ≥ t_thres, take one stochastic-gradient step with the larger step r. Alg. 2 CNC-SGD: SGD with step η, and every t_thres steps a larger step r > η. No added isotropic noise in either.
- Theorems 1–2: (ε, ε^{2/5})-SOSP. Table 1: CNC-PGD O(ε⁻² log(1/ε)), CNC-SGD O(ε⁻⁴ log²(1/ε)) (second-order O(ε_h⁻¹⁰ log²)), no explicit d dependence.
- Lemma 4: learning half-spaces satisfies CNC, with a bound that scales with the square of the negative eigenvalue. Section 6: empirical CNC evidence in neural nets.
- Visual: escape time vs dimension d for isotropic perturbations vs sampling noise aligned with the escape direction (real Monte Carlo); 3D saddle with GD stuck vs CNC-PGD vs CNC-SGD on a real finite sum.

## fang2019sharp — Fang, Lin, Zhang, COLT 2019 (arXiv 1902.00247) [survey 70]
- Assumptions: component gradients L-Lipschitz, Hessian ρ-Lipschitz (1); finite gap Δ (2); bounded noise ‖∇F − ∇f‖ ≤ σ (3); dispersive noise (4, Definition 3: narrow sets get probability ≤ ¼).
- Algorithm 2: plain SGD x_{k+1} = x_k − η∇F(x_k; ζ); if ‖x_k − x₀‖ > B the anchor x₀ ← x_k and the counter resets; after K₀ consecutive in-ball steps return the average of those K₀ iterates. K₀ ~ ε⁻², η ~ ε^{1.5}, B ~ ε^{0.5} (up to logs).
- Theorem 1 in the arXiv version (the notes call it Theorem 5): w.p. ≥ 1 − (T₁+1)p the output has ‖∇f‖ ≤ 18B²ρ-type and λ_min ≥ −17·(…)-type bounds, after T₀ = Õ(ε⁻³·⁵) stochastic gradients. Algorithm 3 (noise-scheduled SGD) drops the dispersive assumption by injecting Gaussian noise every K_o steps.
- Notes' coupling check reproduces: on q = (−u² + v²)/2 with η = 0.1, two coupled paths 0.002 apart in u separate as 0.002·1.1^k; the first k with distance > 2B = 0.02 is 25 (0.0216694).
- Visual: SGD with the anchor ball (exits move the ball; a full K₀ stay ends in an average); the coupled-paths argument on the saddle; ε⁻⁴ vs ε⁻³·⁵.

## gurbuzbalaban2021heavy — Gürbüzbalaban, Şimşekli, Zhu, ICML 2021 (arXiv 2006.04740) [survey 71]
- Streaming least-squares SGD x_k = M_k x_{k−1} + q_k, M_k = I − (η/b)Σ a_i a_iᵀ, q_k = (η/b)Σ a_i y_i: a random affine (Kesten) recursion.
- Theorem 2: if the Lyapunov exponent ρ < 0 and h(α) = 1 has a unique positive root (h(s) = lim(E‖M_k⋯M_1‖^s)^{1/k}), the stationary law has a power-law tail with index α. Theorem 3 (Gaussian a ~ N(0, σ²I)): ρ = E log‖M e₁‖, h(s) = E‖M e₁‖^s.
- Theorem 4: α increases with b and decreases with η, σ² and d. Proposition 5: η_crit = 2b/(σ²(d + b + 1)); below it α > 2, at it α = 2, slightly above α < 2 (infinite variance). h(2) = 1 − 2ησ² + (η²σ⁴/b)(d + b + 1). Theorem 8: W_p convergence at rate h(p)^{k/p}. Corollary 11: generalized CLT for α < 2.
- Notes' scalar check reproduces: a, y ~ N(0,1), η = b = 1: E|1 − a²| = 4φ(1) ≈ 0.96788 < 1, E(1 − a²)² = 2 ⇒ α ∈ (1, 2); η_crit = 2/3.
- Visual: α(η, b, d) computed live by Monte Carlo + root finding, with η_crit; tail plot P(|x| > t) of real SGD runs with slope −α.

## azizian2025global — Azizian, Iutzeler, Malick, Mertikopoulos, ICML 2025 (arXiv 2503.16398) [survey 72]
- SGD x_{n+1} = x_n − η(∇f(x_n) + Z_n); τ = first time within δ of the global minimizers Q. Large-deviation action (H = log E exp(−⟨p, G⟩), Lagrangian = its conjugate); B(p, q) = least action between critical components; transition forests and prunings define E(Q | x).
- Theorem 1: exp((E(Q|x) − a)/η) ≤ E_x[τ] ≤ exp((E(Q|x) + a)/η) for small η, δ; the lower bound needs Q to be attracting enough. E ≡ 0 iff there are no spurious minima (then τ is subexponential).
- Gaussian noise with covariance σ²I: climbing Δf along a reversed gradient orbit costs 2Δf/σ²; descending is free.
- Fig. 1 test function f = 2x₁⁶/13 + x₁⁵/8 − 91x₁⁴/64 − x₁³/2 + 42x₁²/16 + 5x₂²/4 + x₁x₂. Our computation: minima (−2.533, 1.013) f = −7.248 [global], (0, 0) f = 0, (2.078, −0.831) f = −3.297; saddles (−1.116, 0.447) f = 1.590, (0.894, −0.358) f = 0.823. (The paper prints p₁ ≈ (−2.573, 1.029); the printed formula's minimizer is at x₁ = −2.533. Our visuals use the formula.)
- Notes' 3-state example reproduces: costs AG 2, AB 3, BA 5, BG 8 ⇒ E = 7 − 3 = 4; exact T at η = 0.5: 3027.17 (η log T = 4.0077); at 0.25: 8,889,035.88 (4.00008).
- Notes' proof flag verified: (D.44) M_n = Σ_{k≤n}(1 − γμ)^{n−k}⟨u_k, p_k − y_k⟩ satisfies E[M_{n+1} | F_n] = (1 − γμ)M_n, so it is not a martingale as asserted before Doob's inequality (D.46). We report the theorem as stated and flag the step; we do not certify a repair.
- Visual: the camel landscape with SGD hopping between basins; the transition graph with forest/pruning costs from the Gaussian action; log E[τ] vs 1/η from real simulations against the slope E.
