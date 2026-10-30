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

# Round 3 (2026-10-30): 64 pages built by sessions A, B and C

## haeffele2017global [54] — Haeffele & Vidal, CVPR 2017 (openaccess.thecvf.com) + long version arXiv 1506.07540
- Setting (Eq. 17): $f_r(W)=\ell(Y,\sum_{i\le r}\phi(W_i))+\lambda\sum_i\theta(W_i)$, $\ell$ convex and once differentiable, $\lambda>0$, $(\phi,\theta)$ a nondegenerate pair: both positively homogeneous of the same degree $p$.
- Proposition 1: $\Omega_{\phi,\theta}$ (Eq. 15) is convex, positively homogeneous of degree 1; the infimum is attained with $r\le\mathrm{card}(X)$.
- Theorem 1: any local minimizer with some subnetwork $W_{i_0}=0$ is a global minimizer of $f_r$ and its output solves the convex problem (16). **No size condition.**
- Corollary 1: a local minimizer is global if appending a zero subnetwork gives a local minimizer of $f_{r+1}$.
- Theorem 2: if $r>\mathrm{card}(X)$, from any point with finite objective there is a non-increasing path to a global minimizer (constructive meta-algorithm).
- Degree mismatch: Prop. 19 of arXiv 1506.07540 (the notes cite "Proposition 2 of the CVPR supplement"; that supplement URL now returns 404, so we checked the long version): degree $p'<p$ regularizer ⇒ the origin is always a local minimum.
- Survey [54] text: "if the network is large enough, a local minimum in which some subnetwork is identically zero is a global minimum". Imprecise: Theorem 1 needs no size condition; size ($r>\mathrm{card}(X)$) enters only Theorem 2 (the non-increasing path) and, in practice, the chance that some subnetwork is zero. The survey also omits that the regularizer ($\lambda>0$, same degree) is essential.
- Notes: header says "Ref. [47]" (older numbering; survey number is [54]). Content correct.
- Visuals: (1) matched toy $\tfrac12(1-uv)^2+\tfrac\lambda2(u^2+v^2)$, GD; origin strict saddle with $\lambda_{\min}=\lambda-1$, all 200 runs reach $f^\star=\lambda-\lambda^2/2$. (2) mismatch toy $\tfrac12(1-uv)^2+\lambda(|u|+|v|)$, proximal gradient; origin is a spurious local min (39/400 runs trapped at λ = 0.15). (3) offline: one-hidden-layer ReLU nets with weight decay (bias inside $w$), r ∈ {1,…,12}, 16 seeds; every run with a zero neuron that passes the local-min test $\max_{\|w\|=1}|\langle\rho,\mathrm{ReLU}(Xw)\rangle|/n\le\lambda$ ends at the same value 0.0143253 (none lower); 81 runs with dead zero neurons fail the test and sit higher — Theorem 1 needs a true local minimum, not just a zero neuron.

## islamov2024loss [57] — Islamov, Ajroldi, Orvieto, Lucchi, NeurIPS 2024 (arXiv 2410.12455)
- Definition 1 (Eq. 2), α-β-condition: for every x some $x_p\in\mathrm{Proj}(x,S)$ with $\langle\nabla f_i(x),x-x_p\rangle\ge\alpha(f_i(x)-f_i(x_p))-\beta(f_i(x)-f_i^\ast)$ for all i, $\alpha>\beta>0$. Not unique: $(\alpha+\delta,\beta+\delta)$ also works.
- Examples 1–3 (Eqs. 3–6): unbounded minimizer set, a saddle (Example 2, β = α − 8), local minima. Examples 4–5: matrix factorization, two-layer ReLU net with logistic loss + weight decay on bounded sets.
- Theorem 1 (SGD, $\gamma\le(\alpha-\beta)/2L$, Assumptions 1–2): Eq. (9), $O(K^{-1/2})$ up to a ball $O(\beta\sigma^2_{\rm int})$. Theorems 2–3: SPSmax, NGN. Sec. 5 checks Eq. (2) along training runs with the final iterate standing in for $x_p$.
- Our check of Example 2: minimizers ≈ (0.00067, 0.00067) and mirror, f* = 0.49983, strict saddle at (1,1) (eigs −0.812, 0.271). The condition does hold, but needs α ≥ 8.8×10⁶ on [−1.5, 3.5]² because $f_1(x_p)\approx9\times10^{-7}$; Theorem 1's floor is then ≈ 1.1×10⁶ ≫ max(f − f*) = 0.5, for any split α − β. So the paper's own saddle example only gets a vacuous rate (the paper says it does not aim for tight constants). Worth one sentence in the survey: "convergence guarantees" are up to a neighbourhood whose size can be large.
- Survey text: accurate in substance; "convergence guarantees" should read "convergence up to a neighbourhood of size O(βσ²_int)".
- Notes: header "Ref. [50]" (older numbering; survey [57]). Content correct.
- Visuals: (1) Example 2 surface with SGD (pick one $f_i$ per step, step 0.5), saddle eigen-lines; (2) heat map of the α needed per point + PL ratio along the diagonal (zero at the saddle).

## bhojanapalli2016dropping [59] — Bhojanapalli, Kyrillidis, Sanghavi, COLT 2016 (arXiv 1509.03917)
- Algorithm 1 (FGD): $U^+=U-\eta\nabla f(UU^\top)U$ (factor 2 absorbed), step (8) $\eta=1/(16(M\|X_0\|_2+\|\nabla f(X_0)\|_2))$.
- Assumptions (A1)/(A2): start within $\rho\,\sigma_r(U^\star_r)$ with $\rho=\frac{1}{100}\frac{\sigma_r}{\sigma_1}$ (smooth) or $\frac{1}{100\kappa}\frac{\sigma_r}{\sigma_1}$ (strongly convex); (A3) approximate low rank.
- Theorem 4.1 (smooth): $f(X_k)-f(X^\star_r)=O(1/k)$ (Eq. 9). Theorem 4.2 (restricted strongly convex): Eq. (10), factor $\alpha=1-m\sigma_r(X^\star)/(64(M\|X^\star\|_2+\|\nabla f(X^\star)\|_2))$. Corollary 4.3: linear convergence when rank $X^\star=r$. Theorem 5.2: initialization $X_0\propto P_+(-\nabla f(0))$.
- The paper's Sec. 3 toy $\|uu^\top-Y\|_F^2$, $Y=\alpha^2v_1v_1^\top-\beta^2v_2v_2^\top$, is on the page. Our checks: Hessian at 0 = diag(−4a², 4b²) (saddle created by the factorization), at $U^\star$ = diag(8a², 4a²+4b²) ⇒ constant steps need $\eta<1/(2a^2)$; from the paper's example point $(a/2,b/10)$, $\eta=1/M$ diverges for a ≥ 1.5 while step (8) converges (209 steps at a = 2). From an (A2) start at a = 2: bound factor 0.9875, measured 0.880.
- Survey text: accurate.
- Notes: header "Ref. [52]" (older numbering). Content correct.
- Visuals: 3D surface of the toy with FGD balls (step (8) vs 1/M) and a slider for a; 2D Dist² vs k against the Cor. 4.3 bound.

## anandkumar2016efficient [61] — Anandkumar & Ge, COLT 2016 (arXiv 1602.05908)
- Definition 1: p-th order local minimum ($f(y)\ge f(x)-C\|y-x\|^{p+1}$ near x). Definition 4: third-order necessary condition (∇f = 0, ∇²f ⪰ 0, $\nabla^3f[u,u,u]=0$ on the Hessian null space), checkable in polynomial time (Claim 1).
- Algorithm: cubic regularization (Nesterov–Polyak) alternated with a third-order step in a "competitive subspace" (Def. 6). Theorem 9: converges to a point satisfying Def. 4; Theorem 8 / Corollary 1: approximate version in polynomial time. Theorem 10: finding a fourth-order local minimum is NP-hard (via Theorem 11, non-negativity of quartic forms).
- Survey text: accurate. "Certifying fourth-order local optimality is NP-hard [61]" (Sec. IX) is a fair paraphrase (the paper: "we cannot even verify whether a point is a fourth order local minimum").
- Notes: header "Ref. [54]" (older numbering). Their toys ($x^2-y^4$, $x^2+y^3$, deep-linear $\frac12(1-abc)^2$) are correct and labelled as teaching examples.
- Visuals (our toys, bounded below): $f_3=x^2+y^3+y^4$ (second- but not third-order min at 0; GD stalls there with $y_k\approx1/(3\eta k)$, GD + a simplified third-order step escapes to (0, −3/4)); $f_4=x^2-y^4+y^6$ (passes Def. 4 at 0 but not fourth order; GD escape time ≈ 1/(8ηy₀²): 1013/256/63 steps for y₀ = 0.05/0.1/0.2).

## goodfellow2015qualitatively [63] — Goodfellow, Vinyals, Saxe, ICLR 2015 (arXiv 1412.6544)
- Method (Sec. 2): $\ell(\alpha)=J((1-\alpha)\theta_0+\alpha\theta_f)$. Findings: smooth decrease for seven models (Secs. 3–4: FC sigmoid/ReLU/maxout, adversarial maxout, conv maxout on CIFAR-10, MP-DBM, LSTM). The paper's own exceptions: a shallow barrier right after $\theta_0$ (Fig. 6, conv maxout) and a secondary high-error minimum on the MP-DBM line (Fig. 7). Fig. 5: a barrier between two SGD solutions. Sec. 5 / Fig. 10: same picture analytically for deep linear regression. Fig. 14: factored model $(1-w_1w_2)^2$.
- Survey text ("often decreases monotonically, with no visible barrier"): accurate, "often" is the right word.
- Notes: header "Ref. [56]" (older numbering). Content correct; the notes' warning ("not that SGD encounters no saddles") is right.
- Our own MLP check (2-16-16-1 tanh, two spirals, SGD): run 2 monotone; run 1 has a bump of 0.111 at α = 0.25 (17% of the drop) despite 100% train accuracy; in a 24-run sweep 19 were monotone within 1% of the drop. Between two solutions: barrier 1.83. So monotone interpolation is common but not universal (consistent with later work, e.g. Lucas et al. 2021).
- Visuals: (1) factored model surface, SGD from near the saddle, start→solution line (monotone, computed live) vs the (1,1)→(−1,−1) line over the saddle; (2) offline MLP interpolation curves + projection/residual of the SGD path.

## apidopoulos2026taming [43] — Apidopoulos, Lytras, Mertikopoulos, ICML 2026 (PMLR 306:3194–3227)
- Assumption 1: C², gradient-coercive. Assumption 2: (L0, L1)-smoothness $\|\nabla^2f\|\le L_0+L_1\|\nabla f\|$. Assumption 3: sub-Gaussian noise with state-dependent proxy σ(x). Assumption 4 (ABC_σ): $\sigma^2\le A(f-f^\ast)+B\|\nabla f\|^2+C$.
- (TSGD) $x_{t+1}=x_t-\gamma_t\hat g_t/(1+\lambda_tr(x_t))$, deterministic scale $r$ (Eq. 12). Theorem 1: $\sum\gamma=\infty$, $\sum\gamma^2/\lambda<\infty$, $\gamma/\lambda\to0$ ⇒ ∇f(x_t) → 0 a.s. (DTSGD) adds dissipation; Theorem 2 without Assumption 4. Theorem 3 (with Assumption 5: uniform excitation, bounded noise near crit f): strict saddle manifolds are avoided a.s.; parameter regions (25a)/(25b). Experiments: phase retrieval, deep diagonal net (Figs. 1–2).
- Survey text: accurate. Small omissions: two schemes (TSGD needs the ABC_σ noise bound, DTSGD does not), plus gradient coercivity.
- Notes: header "Ref. [35]" (older numbering). Their "teaching" update $\eta_t/(1+\eta_t\|g_t\|)$ normalizes by the random $\|g_t\|$, which the paper deliberately avoids (Remark 2: biased); the notes do say it is not the paper's update. Fine as a caveat.
- Our toy: $f=x^4/4-x^2/2+y^2/2+y^4/4$ ((L0,L1) with L0 = 4, L1 = 3; not globally Lipschitz; coercive). With p = 0.8, q = 0.5, r = 1 + ‖x‖³, σ = 0.3: plain SGD diverges in 25%/64%/100% of runs at γ₀ = 0.3/1/3; TSGD and clipped SGD stay ≥ 98% at a minimum up to γ₀ = 10; TSGD from the stable manifold x = 0 always leaves the saddle.
- Visuals: 3D tamed vs plain SGD from far starts + a start on the stable manifold; 2D success rate vs γ₀ and the (p, q) region (25a).

## panageas2019first [64] — Panageas, Piliouras, Wang, NeurIPS 2019 (arXiv 1906.07772v3, Sep 2025)
- Theorem 4.1: stable-manifold theorem for time non-homogeneous systems. Corollary 4.2/4.3: $\alpha_k\to0$, $\alpha_k\in\Omega(1/k)$ ⇒ the stable set of a strict saddle has measure zero. Theorems 5.1/5.3/5.4/5.5–5.6: GD, mirror descent, proximal point, manifold GD. Remark 3.1 + Fig. 1: summable steps can converge to a non-critical point (Fig. 1: $x^2-y^2$ with $1/\sqrt k$, $1/k$, $1/k^4$).
- **Correction in arXiv v3**: the acknowledgement says the proof of the main theorem requires the extra assumption that the Hessian at the saddle is invertible (Sec. 2 and Theorem 5.1 now assume it). An invertible Hessian rules out non-isolated saddle manifolds (they have a zero eigenvalue along the manifold), so Theorem 1.1's "isolated and non-isolated" is not covered by the corrected proof.
- Survey text: accurate (it does not claim non-isolated saddles; "decaying no faster than Ω(1/k)" and the summable-step remark are right). Suggest adding "with an invertible Hessian at the saddle" or citing v3.
- Notes: header "Ref. [57]" (older numbering). "including isolated and non-isolated strict-saddle sets" should be qualified by the v3 invertibility assumption.
- Our toy $f=x^2/2-y^2/2+y^4/4$, $\alpha_k=0.8/(k+1)^s$: s = ½, 1 reach (0, 1) from (1.5, 0.05); s = 2 stops at (0.174, 0.146), ‖∇f‖ = 0.225 ($x_\infty=1.5\prod(1-0.8/(k+1)^2)=0.1738$). Grid: for s = ½, 1 only the rows y = 0 and y = ±1.5 (mapped exactly onto y = 0 by the first step) end at the saddle; for s = 2, 92.4% stop at non-critical points.

## mertikopoulos2020almost [65] — Mertikopoulos, Hallak, Kavis, Cevher, NeurIPS 2020 (arXiv 2006.11144)
- Assumptions 1–4: f G-Lipschitz and L-smooth, bounded sublevel and gradient-sublevel sets, noise with finite q-th moment. Theorem 1: bounded APTs a.s.; Theorem 2: $\gamma_n=\Theta(1/n^p)$, $p\in(2/(q+2),1]$ ⇒ a.s. convergence to a connected component of crit f. Theorem 3: $p\in(0,1]$, bounded ($q=\infty$) uniformly exciting noise (Assumption 5) ⇒ $P(X_n\to S)=0$ for every strict saddle manifold S (Fig. 1: a ridge). Theorem 4: local $O(1/n^p)$ rate near a regular minimizer, conditioned on staying in a neighbourhood.
- Survey text: "SGD with Robbins–Monro step sizes ... avoids strict saddles, including nonisolated saddle manifolds ... under suitable noise conditions". Mostly right, but the paper's point is that it goes **beyond** Robbins–Monro: Theorem 2 allows non-square-summable steps ($p>2/(q+2)$) and Theorem 3 allows any $p\in(0,1]$ (with bounded noise). Suggest "with steps $\gamma_n\propto1/n^p$".
- Notes: header "Ref. [58]" (older numbering). Content correct.
- Our toy: radial $\varphi(r)=r^4/4-r^3+r^2$, ridge of strict saddles at r = 1 (eigenvalues −1, 0). GD started exactly on the ridge stays; SGD from 200 ridge points always leaves (≈ 38–44% to the centre, 56–63% to the outer ring for p = 0.5, 0.75, 1). Theorem 4 slopes: −0.509, −0.754, −1.004 for p = 0.5, 0.75, 1. The toy's gradient is unbounded at infinity (Assumption 1 wants bounded); runs stay in r < 2.6.

## antonakopoulos2022adagrad [66] — Antonakopoulos, Mertikopoulos, Piliouras, Wang, ICML 2022 (PMLR 162)
- Deterministic AdaGrad: ADANORM, ADADIAG, ADAFULL with $G_t=\delta_0^2I+\sum_{s\le t}g_sg_s^\top$ (diag or full), $\Gamma_t=G_t^{-1/2}$. Proposition 1: $\Gamma_t\to\Gamma\succ0$, $\sum\|\nabla f(x_t)\|^2<\infty$. Theorem 1: stable-manifold theorem for the non-autonomous system. Proposition 3: local diffeomorphism for $\delta_0\ge L$. Theorem 2: all three avoid strict saddles from almost any initialization.
- Survey text: accurate. Note the analysis is for deterministic gradients ("all the step-size policies and gradients are deterministic").
- Notes: header "Ref. [59]" (older numbering). Content correct.
- Our toy (saddle with curvature axes at 30°, δ₀ = 10): all methods escape every start off their stable set, in time ∝ log(1/d) (no uniform bound). Observation worth a sentence: the preconditioner changes **where** the stable set is: for ADADIAG it crosses our test segment at d* = 0.180, not at the Hessian's stable manifold (d* = 0 for GD, ADANORM, ADAFULL).
- Visuals: 3D GD / AdaGrad-full / AdaGrad-diag from 10⁻³ off the Hessian's stable manifold + AdaGrad-diag from 10⁻⁹ off its own stable set; 2D escape steps vs d for the four methods.

## cheridito2024gradient [67] — Cheridito, Jentzen, Rossmannek, JOTA 2024 (arXiv 2208.02083)
- Theorem 1.1 / 2.2: center-stable manifold theorem for maps that are only $C^1$ with locally Lipschitz derivative on a full-measure open set (Lipschitz center-stable manifold under Assumption 2.1). Prop. 4.5: for an affine target every non-global critical point has $L=f'(\alpha)^2(\beta-\alpha)^3/(12(n+1)^4)$, n even ≤ N, and is a sawtooth; non-global local minima exist (dead networks with $f'(\alpha)v_jw_j<0$). Theorem 4.7: for a.e. step, starts converging to the strict saddles in S are null. Theorem 1.2 (N even): GD limit loss below $f'(\alpha)^2(\beta-\alpha)^3/(12(N-1)^4)$ ⇒ limit loss 0. Sec. 5: leaky ReLU (small leak).
- Survey text: accurate. One nuance: "under a favorable initialization, given by an explicit threshold on the limiting loss" — the condition is on the limit of the run, not on the start (the abstract phrases it the same way).
- Notes: header "Ref. [60]" (older numbering). Content correct.
- Our checks: sawtooth levels 1/12, 1.029e−3, 1.333e−4 (n = 0, 2, 4); the n = 2 sawtooth built as a 4-neuron net with the sign pattern of Prop. 4.5(iii)(b) is critical (|∇L| = 7e−17) and a strict saddle (λ_min = −0.024). 40 GD runs (N = 4) all end below the threshold at loss ≤ 3.5e−8. A slice through a dead neuron shows the flat critical region (L = 1/12) where GD freezes.

## du2017gradient [68] — Du, Jin, Lee, Jordan, Póczos, Singh, NeurIPS 2017 (arXiv 1705.10412)
- Theorem 4.1: f on R^d with B, ℓ, ρ = poly(d), init U[−1,1]^d: GD stays Ω(1) from local minima for $e^{\Omega(d)}$ steps; PGD reaches one in poly(d, 1/ε). Cor. 4.2 (no ε-SOSP visited), Cor. 4.3 (any init concentrated in an ℓ∞ ball, e.g. Gaussian). Construction: "octopus" of d saddles, $2^d$ minima, spline gluing + Whitney extension; waits grow by $(L+\gamma)/\gamma$; experiments d = 5, 10 (Figs. 3–4).
- Survey text: accurate ("under fairly natural random initialization schemes" matches Cor. 4.3). Note the minimizers lie outside the initialization cube (the paper says so).
- Notes: header "Ref. [61]" (older numbering). The $\frac12x^2-\frac12y^2+\frac14y^4$ example is labelled as illustrative. Correct.
- Our chain (smooth, not the paper's construction, same mechanism): successive GD waits grow by 1.95–2.09 (L = 1) and 3.07–3.17 (L = 2), matching (L+γ)/γ; PGD's stay bounded. In 2/5 runs with L = 2, d = 6 the last coordinate underflows to ~3e−323 and double-precision GD never leaves the last saddle.

## wu2025loss [56] — new notes check (49_Wu_Simsek_Ged_ReLU_Stationarity.pdf)
- The replaced notes agree with the existing page: directional stationarity, escape neurons (no escape neuron ⇒ local minimum; converse for scalar output), embedding by splitting a neuron, saddle-to-saddle from vanishing initialization. No contradiction with `papers/wu2025loss/`.
- One stale reference only: the page's "Kink & escape" tab note says "Teaching example from our notes (x = (1, 0), y = 1)". The new notes no longer contain that one-neuron example (they use the generic $\sum_j a_j\sigma(w_j^\top x)$ model and the neuron-splitting identity). Suggest rewording the note to "Teaching example (one neuron, one data point)". Not edited (page belongs to an earlier round). Notes header says "Ref. [49]" (older numbering; survey [56]).

## hochreiter1997flat [25] — Hochreiter & Schmidhuber, Neural Computation 9(1) 1997
- Flat minimum = large connected region in weight space with approximately constant error (tolerable error $E_{\rm tol}$, boxes). MDL / Gibbs-algorithm argument: flat ⇒ low weight precision (few bits) ⇒ less overfitting; prior over input/output functions, no Gaussian weight prior. Flat Minimum Search needs second-order derivatives but keeps backprop's order of complexity; prunes units/weights/inputs; experiments incl. stock-market prediction (reported).
- Survey text: cited only as part of the flat-vs-sharp debate [25]–[27]; fine.
- Notes: correct. Visual: two equally deep minima (Hessian 3.3 vs 69.4), ε-regions 20.9× larger for the flat one (≈ 4.4 bits less for 2 weights), robustness to weight noise.

## keskar2017large [26] — Keskar, Mudigere, Nocedal, Smelyanskiy, Tang, ICLR 2017 (arXiv 1609.04836)
- SB = 256, LB = 10% of data, ADAM; 6 networks (F1 MNIST, F2 TIMIT, C1–C4 CIFAR-10/100). LB generalizes worse (Table 2), LB minimizers sharper by metric (4) (Tables 3–4), parametric plots (Fig. 3), batch-size threshold (Fig. 4); explanation: SB noise pushes out of sharp basins (Sec. 4).
- Survey text: fine. Notes: correct.
- Toy (a model of the Sec. 4 explanation, labelled as such): finite sum of linear tilts on the two-basin landscape; sharp-basin share ≈ 2% for B ≤ 2, 19–25% for B ≥ 3 (a threshold, as in their Fig. 4); B = 1 too noisy to settle.

## dinh2017sharp [27] — Dinh, Pascanu, Bengio, Bengio, ICML 2017 (arXiv 1703.04933)
- Def. 1 volume ε-flatness, Def. 2 ε-sharpness, Def. 5 α-scale transformation. Theorem 2: ε-flat set has infinite volume; Theorem 3: Hessian transforms as $D_\alpha HD_\alpha$; Theorem 4: spectral norm arbitrarily large; Theorem 5: deep ReLU, $r-\min_kn_k$ eigenvalues > M.
- Survey text ("Sharpness can be changed by reparameterization [27]"): accurate. Notes: correct.
- Toy: $L=(\theta_2{\rm ReLU}(\theta_1)-1)^2$, minima $(a,1/a)$ all compute the same function; $\lambda_{\max}=2(a^2+a^{-2})$ (32.1 at a = 4), ε-sharpness 0.021 → 0.161.

## bottou2018optimization [19] — Bottou, Curtis, Nocedal, SIAM Review 60(2) 2018 (arXiv 1606.04838)
- Sec. 4.3 nonconvex SG: Theorem 4.8 (fixed step ᾱ ≤ μ/(LM_G)): $\mathbb E[\frac1K\sum\|\nabla F\|^2]\le\bar\alpha LM/\mu+2(F(w_1)-F_{\inf})/(K\mu\bar\alpha)$; Theorems 4.9/4.10 (diminishing steps) and Cor. 4.12 (lim E‖∇F‖² = 0 with extra smoothness). Secs. 5–6: noise reduction (dynamic sampling, gradient aggregation, iterate averaging) and second-order methods.
- Survey text: accurate. Notes: correct (and they flag that their derivation uses a simplified variance assumption).
- Toy: $F=(x^2+y^2)/4+\sin x\sin y$ (L = 1.5) with N(0, I) noise: running averages 0.073/0.237/0.641 for ᾱ = 0.05/0.2/0.5, always under (4.28b); diminishing steps → last iterate 0.0094.

## jain2017non [20] — Jain & Kar, Foundations and Trends in ML 10(3–4) 2017 (arXiv 1712.07897)
- Ch. 3 gPGD (Algorithm 2), Def. 3.2 RSC/RSS over C, Theorem 3.3 (β/α < 2, η = 1/β ⇒ ε-optimal over C in O(α/(2α−β) log 1/ε)); Theorem 7.4 (IHT with a larger projection sparsity removes the condition-number restriction); Ch. 6 stochastic techniques that escape saddles (noisy GD, Algorithm 6, tensor factorization).
- Survey text ("centered on problems with known structure such as sparse recovery and low-rank matrix estimation, with a chapter on escaping saddle points"): accurate.
- Notes: correct.
- Toy: B₀(1) ⊂ R², quadratic with β/α = 1.86 (gPGD reaches the optimum) vs 19 (stuck at (0, 1.08), f = 0.137).

## jin2021nonconvex [21] — Jin, Netrapalli, Ge, Kakade, Jordan, J. ACM 68(2) 2021 (arXiv 1902.04811)
- Algorithm 1 (PGD) adds $\xi_t\sim\mathcal N(0,(r^2/d)I)$ at **every** step (the paper notes the 2017 version perturbed only under conditions). Theorem 13: ε-SOSP in Õ(ℓΔ_f/ε²), polylog d. Theorem 16: PSGD. Sec. 3: SOSP = global for many problems.
- Survey text: accurate.
- **Notes discrepancy**: section 3 of the notes ("Perturb only when the gradient is small", "x ← x + ξ, ξ ~ Unif(B(0, r))") describes the 2017 (ICML) algorithm, not this JACM paper's Algorithm 1 (Gaussian noise every step). The conclusions are the same in spirit.
- Toy: escape time from the saddle with N(0, r²/d) noise: median 63 (d = 1) → 113 (d = 16384), slope 5.17 per unit log d vs theory 5.25.

## sun2020optimization [22] — R. Sun, J. Oper. Res. Soc. China 8, 2020 (arXiv 1912.08957)
- Sec. 4.1: explosion/vanishing, Eq. (13) chain F(w) = (w₁⋯w_L − 1)², example (w⁷ − 1)² (Fig. 2); Secs. 4.2–4.5 initialization, normalization, architecture; Sec. 5 SGD/adaptive/distributed; Sec. 6 landscape theory.
- Survey text: accurate. Notes: their "teaching example (not a theorem quoted from Sun)" is in fact essentially the paper's own example (2^{L−1}|e| vs 0.5^{L−1}e right after Eq. 13) — harmless.
- Toy: slice (ab^{L−1} − 1)², GD step 0.02: L = 7 plateau start needs 412 steps (113 at L = 2), cliff start (1.25, 1.25) leaves the box at step 1.

## sun2020global [23] — Sun, Li, Liang, Ding, Srikant, IEEE Signal Processing Magazine 37(5) 2020 (arXiv 2007.01429)
- Sec. IV-A single neuron: Prop. IV.1 (no sub-optimal local minima iff no zero of φ is a local extremum), Fig. 2 (φ = −(z−1)²/8, (1, 1) a non-strict sub-optimal local min); Theorem IV.1 (no spurious valley for wide nets); empirical landscape and convergence links.
- Survey text: accurate. Notes: correct (their toy (ab − 1)² is the paper's Fig. 1).
- Toy: Fig. 2 function; GD from a 41×41 grid: 30.4% stuck at F = 1 (quadratic φ) vs only the saddle (0, 0) for tanh.

## danilova2022recent [24] — Danilova, Dvurechensky, Gasnikov, Gorbunov, Guminov, Kamzolov, Shibaev, 2022 (arXiv 2012.06188)
- Survey of complexity of ε-stationarity (incl. lower bounds, SPIDER-type variance reduction), PŁ (Sec. 5.1, Eq. 36, linear rate $e^{-\mu k/L}$), star-convexity, higher-order/tensor and zeroth-order methods.
- Survey text: accurate. Notes: correct.
- Toy: $x^2+3\sin^2x+y^2$ (nonconvex, PŁ with μ = 0.1755 numerically on [−6, 6], L = 8): all 200 GD runs obey $e^{-\mu k/L}$ and reach the global minimum.

## shub1987global [109] — M. Shub, Global Stability of Dynamical Systems, Springer 1987
- Book not available online; checked through Lee et al. 2016 (arXiv 1602.04915, Theorem 4.4 = Shub's Theorem III.7, center-stable manifold). The notes cite Theorem 5.20 for the hyperbolic stable manifold — could not be verified directly.
- Notes were downloaded from Drive (`Ref109_Shub_1987.pdf`, id 1LjCBSKfXTQnHAlgAJr-OKveMW6r0HKMB) into `public/notes/shub1987global.pdf`. Header says "Paper 79" (internal numbering).
- Survey text ("By the center-stable manifold theorem [109] ... a manifold of dimension strictly less than d"): accurate.
- Toy: GD map on $x^2/2-u^2/2+u^4/4$, $u=y-0.8x^2$: stable manifold computed by bisection is a curve through the saddle tangent to the contracting eigenvector ($|y^*|\le0.282x^2$ near 0), $y^*(\pm1.2)=0.533$; escape after 40/89/137/185/233 steps from δ = 10⁻²…10⁻¹⁰.

## nesterov2006cubic [40] — Nesterov & Polyak, Math. Program. 108 (2006); read from the open CORE DP 2003/41 preprint (UCLouvain DIAL)
- Assumption 1: ∇²f is L-Lipschitz on a convex set containing the initial level set. Step T_M(x) = global argmin of ⟨g,s⟩ + ½⟨Hs,s⟩ + (M/6)‖s‖³ (Eq. 2.4); it satisfies g + Hs + (M/2)‖s‖s = 0 (2.5) and H + (M/2)r_M I ⪰ 0 (Prop. 1).
- Lemma 1 (Eq. 2.3): |f(y) − quadratic model| ≤ (L/6)‖y − x‖³, so for M ≥ L the cubic model is an upper bound and f(T_M(x)) ≤ f̄_M(x) (Lemma 4). Scheme (3.3): M_k ∈ [L₀, 2L], monotone.
- Theorem 1: Σ r³ ≤ 12(f(x₀) − f*)/L₀ and min_{i≤k} μ_L(x_i) ≤ (8/3)(3(f(x₀) − f*)/(2kL₀))^{1/3}, i.e. min ‖∇f‖ = O(k^{−2/3}) vs O(k^{−1/2}) for the gradient method. Theorem 2: limit points satisfy ∇f = 0, ∇²f ⪰ 0. Lemma 6: the method cannot be attracted to a non-degenerate saddle or maximum, even started exactly on it. Theorem 3: local quadratic convergence near a non-degenerate minimum. Sec. 5: the subproblem is a 1-D convex problem.
- Notes: say the full article was not available ("Source blocker"); they state only the canonical model, the first-order condition and generic claims. Everything they state matches the paper. They label the paper "Ref. [32]" (old numbering) — survey number is [40].
- Survey reference text: correct (title, authors, Math. Program. 2006).
- Visuals: (1) f = x²/2 − y²/2 + y⁴/4 + 0.3xy² from (1.2, 0.03): pure Newton → saddle, CR (M = L = 10, exact subproblem) → minimum in 12 steps, GD (1/L₁, L₁ = 8) in 167; (2) the cubic model sheet at the current point for M ∈ {0, 3, 10, 25} with "take the step", plus the λ_min slice (model above f on the box iff M ≥ L there); (3) min-so-far ‖∇f‖ of both runs vs Theorem 1 and the GD bound (1.2.13) evaluated with this f's constants (both runs far below the worst case).

## zhang2021escape [82] — Zhang & Li, NeurIPS 2021 (arXiv 2111.14069)
- Setting: f ℓ-smooth, ρ-Hessian Lipschitz; ε-approximate SOSP: ‖∇f‖ ≤ ε, λ_min(∇²f) ≥ −√(ρε) (Eq. 1).
- Algorithm 1 (gradient-only Hessian power method): y₀ ~ Unif(B(r)), y_t = y_{t−1} − ‖y_{t−1}‖/(ℓr)·(∇f(x̃ + r y_{t−1}/‖y_{t−1}‖) − ∇f(x̃)). Proposition 3: if λ_min ≤ −√(ρε), w.p. ≥ 1 − δ₀ the output ê has êᵀHê ≤ −√(ρε)/4 after T = Õ(log n/√ε)-type many iterations. Lemma 6: a step of length ¼√(ε/ρ) along ±ê lowers f by ≥ (1/384)√(ε³/ρ).
- Theorem 7 (Algorithm 2, accelerated): Õ(log n/ε^1.75) iterations; Theorem 9 (Algorithm 3, stochastic): Õ(log² n/ε⁴). Table 1: the gain is in powers of log n (PGD log⁴n/ε², PAGD log⁶n/ε^1.75, PSGD log¹⁵n/ε⁴); in the stochastic setting ε⁻⁴ is worse in ε than nested-loop Neon2 (ε^−3.5) and SPIDER-type (ε⁻³).
- Fig. 1 setup reproduced: f = x₁⁴/16 − x₁²/2 + 9x₂²/8, η = 0.05, r = 0.1, M = 300. Check: 94.7% of Algorithm-1 directions within 5° of ±e₁ at t = 30; PGD at t = 90: 44.0% of paths decrease f by ≤ 0.9 (paper: "over 40%"). Our equal-budget completion of Algorithm 1 (kick of length 1/ρ = 0.267 from Lemma 6, then 60 GD steps) stalls in 0.7% (paper: "less than 5%" for its own descent procedure, not specified in detail).
- Notes: consistent with the paper (Prop. 3, Thm 9 numbering correct; worked example x₁ grows ×1.2/step with step 0.2 on −x²/2 + 2y² is right). The notes call the stochastic rate an improvement "in dimension dependence" — correct, but they do not mention that ε⁻⁴ is worse in ε than Neon2/SPIDER; the page says so.
- Survey reference text: correct.
- Visuals: (1) live replication of Fig. 1 (both clouds, alignment share over time, histogram of decrease at t = 90); (2) Table 1 orders vs dimension n for both settings.

## wang2019spiderboost [84] — Wang, Ji, Zhou, Liang, Tarokh, NeurIPS 2019 (arXiv 1810.10690)
- Assumption 1: Ψ bounded below and **each** ∇f_i L-Lipschitz. Eq. (1) is the SARAH/SPIDER recursive estimator. Algorithm 1: x_{k+1} = x_k − v_k/(2L), q = |S| = √n, output a uniformly random iterate.
- Theorem 1: E‖∇f(x_ξ)‖ ≤ ε with K = O(L(f(x₀) − f*)/ε²) iterations, SFO O(√n ε⁻² + n) (matches the SPIDER lower bound for n ≤ O(ε⁻⁴)). Theorem 2: Prox-SpiderBoost, E‖G_η(x_ξ)‖ ≤ ε with the same SFO and O(ε⁻²) prox calls. Theorem 3: Prox-SpiderBoost-M (momentum) same order; the speed-up is empirical (Sec. 5). Theorem 6/Corollary 1: online version O(ε⁻³).
- Notes: correct (example f₁,₂ = ½(x ∓ 1)² gives v = 2 exactly at x = 2; prox of 0.3|x| at 0.8 = 0.5 verified). They rightly say the paper has no saddle-escape result.
- Survey reference text: correct. Note the survey cites it in Sec. V (escape algorithms) although the paper only proves first-order stationarity — worth a word in the survey if it claims escape.
- Visuals: (1) SPIDER (normalized ε/L steps) vs SpiderBoost (1/(2L)) on a 196-component non-convex finite sum: check run reaches ‖∇f‖ < 0.05 after 2 744 vs 11 984 SFO calls; (2) Ψ = f + 0.3(|x| + |y|): prox step lands on y = 0 exactly, subgradient steps chatter at |y| ~ 10⁻².

## allenzhu2018natasha [85] — Allen-Zhu, NeurIPS 2018 (arXiv 1708.08694 v4)
- Online problem f = E_i f_i; assumptions (A1) bounded variance, (A2) L-smooth, (A4) L₂-Hessian Lipschitz. (ε, δ)-approximate local minimum: ‖∇f‖ ≤ ε, ∇²f ⪰ −δI (Def. 4.2).
- Theorem 1 (informal): Natasha1.5 finds ‖∇f‖ ≤ ε in O(ε⁻³ + σ^{1/3}ε^{−10/3}) for σ-bounded nonconvexity. Theorem 2 (informal): for δ ∈ (0, ε^{1/4}), Natasha2 finds an (ε, δ)-local minimum w.p. ≥ 2/3 in Õ(δ⁻⁵ + δ⁻¹ε⁻³) stochastic gradients + HVPs; Remark 4.4: with δ = ε^{1/4}, Õ(ε^{−3.25}) (Cor. 4.6); δ = ε^{1/2}: Õ(ε^{−3.5}) (Cor. 4.7). Sec. 4.2: the reduction (Carmon et al.) — Oja for curvature below −δ, else solve F_k = f + L·max(0, ‖x − y_k‖ − δ/L₂)² in the "safe zone". Remark 4.5: HVPs replaceable by gradient differences (Neon2).
- Notes: consistent. Their "δ⁻⁵ + δ⁻¹ε⁻³ + ε^{−13/4}" matches Remark 4.4; worked example f = x⁴/4 − x²/2 + y²/2 (Hessian at x = 0.6: diag(0.08, 1)) correct.
- Survey reference text: correct.
- Visuals: (1) the Sec. 4.2 loop on a 200-component finite sum from (0, 1.3) on the saddle's stable manifold: Oja (30 stochastic HVPs) → step δ/L₂ along negative curvature, else minibatch SGD in the safe zone (simplified: the paper uses Natasha1.5 there). Check: 20/20 seeds reach a (0.05, 0.5)-local min (median 30 rounds, 10 NC steps), never closer than 0.87 to the saddle; GD from the same start converges to the saddle. (2) Theorem 2 orders vs ε with a δ slider.

## ge2019stabilized [86] — Ge, Li, Wang, Wang, COLT 2019 (arXiv 1905.00529 v1)
- Algorithm 2 (Perturbed SVRG): SVRG + uniform perturbation in B(δ) when ‖∇f(x_sm)‖ ≤ G, then a super epoch. Theorem 2 (PMLR; Thm 1 in arXiv v1): Õ((n^{2/3}LΔ/ε² + n√ρΔ/ε^{1.5})(1 + (ρ′/(n^{1/3}ρ))²)) where ρ′ is the component Hessian-Lipschitz constant.
- Algorithm 3 (Stabilized SVRG): also subtract v_shift = ∇f(x̃), i.e. run SVRG on f̂ = f − ⟨∇f(x̃), x − x̃⟩, with a much smaller perturbation. Theorem 3 (PMLR; Thm 2 in arXiv v1): Õ(n^{2/3}LΔ/ε² + n√ρΔ/ε^{1.5}), only polylog in ρ′, b = Õ(n^{2/3}), m = n/b, η = Õ(1/L). Sec. 4.3: Phase 1 stays within Õ(δ) of x̃ (PMLR Lemma 12; arXiv Lemma 9) while the projection on the negative eigenspace reaches δ/10 (Lemma 11; arXiv 8); Lemma 10 (arXiv 7): decrease Ω̃(γ³/ρ²) w.p. ≥ 1/8.
- Theorem numbering: COLT/PMLR version has Perturbed SVRG = Theorem 2 and Stabilized SVRG = Theorem 3 (arXiv v1: Theorems 1 and 2). The notes' "Theorem 3" follows PMLR — correct; the page uses PMLR numbers. Worked example (shifting f = x + 2x² − y²/2 at 0) correct.
- Survey reference text: correct.
- Visuals: (1) 125 components with zero-mean cubic terms (ρ′ ≈ 25 vs ρ = 9); same perturbation, Perturbed vs Stabilized SVRG: median ‖x_t − x̃‖ at t = 1/(ηγ) = 10 is 0.041 vs 0.020 (= δ) over 200 seeds; both escape on this easy instance (the benefit is in the variance analysis, stated on the page); (2) Table 1 orders vs n.

## li2019ssrgd [87] — Z. Li, NeurIPS 2019 (arXiv 1904.09265)
- Algorithm 1: SARAH/SPIDER estimator with full gradient every m steps, plus uniform perturbation (radius r) when ‖∇f(x_sm)‖ ≤ g_thres outside a super epoch. Theorem 1: η ≤ (√5 − 1)/(2L), m = b = √n → O(n + LΔ√n/ε²) (first order). Theorem 2: Õ(LΔ√n/ε² + Lρ²Δ√n/δ⁴ + ρ²Δn/δ³) for (ε, δ)-SOSP. Theorems 3–4: online versions (O(σ²/ε² + LΔσ/ε³) first order). Lower bound Ω(√n/ε²) (Fang et al.) → near-optimal.
- Notes: correct (Theorems 1–2 match; both worked examples verified: x₁ = 1.8 recursion; x₁₀ = 0.00349, y₁₀ = 0.02594).
- Survey reference text: correct.
- Visuals: (1) start (0, 1.3) on the stable manifold: plain SARAH keeps x = 0 exactly and converges to the saddle; SSRGD reaches (±1, 0) in 50/50 seeds (super-epoch stop simplified to distance/time); (2) √n vs n^{2/3} orders with Fang's lower bound.

## chen2022faster [88] — Chen, Zhou, Gu, ALT 2022 (PMLR 167; arXiv 2110.13144)
- Notation: (ε, ε_H)-local minimum ‖∇F‖₂ ≤ ε, λ_min ≥ −ε_H. Algorithm 1 (LENA): GD phase with normalized steps η/‖d_t‖ while ‖d_t‖ > ε; escape phase: perturb, η_H steps, and if Σ_{i=m_s}^t η_i²‖d_i‖² > (t − m_s + 1)D shrink the last step to equality and restart; if ℓ_thres steps pass, output x_{m_s}. Eq. (4.1): ‖d_t − ∇F(x_t)‖² = Õ(Σ‖x_{i+1} − x_i‖²) for SPIDER/STORM — why movement is capped. Theorem 5.1 (SPIDER) and 5.3 (STORM): Õ(σLΔε⁻³ + σρ³LΔε_H⁻⁶); Remark 5.2: Õ(ε⁻³) for ε_H = √ε vs Õ(ε^{−3.5}) for SSRGD; lower bound Ω(ε⁻³ + ε_H⁻⁵) (Arjevani et al. 2020) — optimal in ε, not in ε_H.
- Notes: correct; the shrinkage example √(0.003 − 0.03² − 0.04²) = 0.0224 is an analogy (they say so).
- Survey reference text: venue "ALT" correct (PMLR 167, 2022).
- Visuals: (1) LENA-SPIDER, online noise that vanishes in x on the stable manifold (so only the kick starts the escape); budget chart and ‖d_t‖ vs ‖∇F‖. Check: 30/30 seeds output ‖∇F‖ ≤ 0.1 next to a minimum after a median of one shrink; ℓ_thres = 120 (60 failed in 3/30 runs because the kick was too small — the theorem's log factors matter). (2) ε⁻³ + ε_H⁻⁶ vs lower bound and SSRGD with an ε_H = ε^p slider.

## staib2019escaping [89] — Staib, Reddi, Kale, Kumar, Sra, ICML 2019 (arXiv 1901.09149)
- Eq. (1–3): idealized RMSProp x_{t+1} = x_t − ηA_tg_t, A = E[ggᵀ]^{−1/2}; noise ξ = A(g − ∇f) has Cov(ξ) = I − G^{−1/2}∇∇ᵀG^{−1/2} ≈ I near stationary points. Definition 4.2 ((Λ₁, Λ₂, Γ, ν, λ₋)-preconditioner). Theorem 4.1: EMA estimation error with β = 1 − Cη^{2/3} and burn-in W = O(η^{−2/3}). Theorem 4.3: first-order rate with an inexact preconditioner. Theorem 4.4 (Algorithm 4, occasional large step r): (τ, √(ρτ))-stationary point in Õ(τ⁻⁵); Corollary 4.2: same for RMSProp (Appendix B).
- Sec. 5 experiment replicated exactly: f_i = ½xᵀdiag(1, −0.1)x + b_iᵀx + ‖x‖₁₀¹⁰, Cov(b) = diag(1, 0.01), b_i = (±1, ±0.1). Check (20 seeds, 10⁴ steps from the saddle): median steps to |x₂| > 0.4: RMSProp(0.001) 2 994, SGD(0.001) never, SGD(0.01) 2 846; tail mean f: −0.0124 (RMSProp) vs −0.0101 (SGD 0.01) — the paper's Fig. 1 conclusion.
- Notes: correct (Definition 4.2, Theorems 4.3–4.4, Corollary 4.2, Õ(ε⁻⁵) all match; whitening example diag(9,1) verified).
- Survey reference text: correct.
- Visuals: (1) live replication of the experiment with the check-script medians as dashed curves; (2) covariance ellipses of raw vs preconditioned noise at (0, s) from Eq. (3).

## chizat2018global [32] — Chizat & Bach, NeurIPS 2018 (arXiv 1805.09545)
- Problem (1): min over signed measures J(µ) = R(∫φ dµ) + G(µ) (convex in µ); particle discretization J_m is non-convex. Gradient descent on particles ↔ Wasserstein gradient flow as m → ∞ (Theorem 2.6). Theorem 3.3 (2-homogeneous φ, e.g. ReLU) / 3.5 (partially 1-homogeneous, e.g. sigmoid): if the initial support separates the inner and outer spheres (resp. {−r₀}×Θ from {r₀}×Θ) and the flow converges, the limit is a global minimizer; lim_{t,m→∞} J(µ_{m,t}) = min J. Qualitative only (no rates). Prop. 4.3: for ReLU the classical parameterization has discontinuous dΦ; the theorem uses a "signed square" reparameterization.
- Fig. 2: ReLU net, d = 2, Gaussian features, teacher with m₀ = 4 neurons, particles start on a small sphere: failure with 5, success with 10 and 100. Fig. 3: particle complexity only slightly above m₀ vs exponential for convex optimization on a fixed grid.
- Our replication (fixed teacher, 600 samples, step 0.4m per particle, 5 000 steps): m = 5 and m = 10 stall at loss ≈ 3×10⁻⁵ (a plateau: unchanged after 20 000 steps), m = 20 and m = 100 reach ≤ 10⁻⁶ and keep decreasing. Same phenomenon, different threshold (the paper's 10-particle success is instance-dependent) — stated on the page.
- Notes: correct and careful (they say the result is qualitative, many-particle). They label the paper "Ref. [29]" (old numbering) — survey number [32].
- Survey reference text: correct.
- Visuals: (1) target surface vs the student's output sheet replayed during training, m ∈ {5, 10, 20, 100}; (2) the Fig. 2 view, trajectories of |w_j|θ_j with target directions.

## zou2019improved [90] — Zou & Gu, NeurIPS 2019 (arXiv 1906.04688)
- Assumptions 3.1 (‖x_i‖ = 1, (x_i)_d = µ > 0) and 3.2 (‖x_i − x_j‖ ≥ φ). Algorithm 1: hidden rows N(0, 2/m I), output rows N(0, I/d). Theorem 3.3 (GD, L hidden layers): m = Ω(kn⁸L¹² log³m/φ⁴), η = O(k/(L²m)) → ε training loss in O(n²L² log(1/ε)/φ) iterations. Remark 3.4: vs Allen-Zhu et al. Ω̃(kn²⁴L¹²/φ⁸), O(n⁶L² log(1/ε)/φ²). Corollary 3.5: two-layer, m = Ω(kn⁸ log³m/φ⁴). Theorem 3.8 (SGD): m = Ω(kn¹⁷L¹² log³m/(B⁴φ⁸)), T = O(n⁵ log m log²(1/ε)/(Bφ²)). Theorem 3.10: two-layer SGD.
- Proof ingredients: Lemma 4.1 (gradient lower bound ‖∇L‖²_F ≥ Ω(mφL(W)/(kn²)) in a τ-ball), Eq. (4.1)/Lemma 4.2 (telescoping trajectory length, ‖W_t − W_0‖₂ ≤ O(√(kn² log n/(mφ)))), Lemma 4.4 (semi-smoothness, from Allen-Zhu et al.).
- Notes: correct (n⁸ vs n²⁴ width, Theorems 3.3/3.8/3.10; telescoping example). Minor: notes say "Theorem 3.10 gives tighter special-case bounds for two-layer networks" — for GD the two-layer case is Corollary 3.5.
- Survey reference text: correct.
- Visuals (one-hidden-layer net, n = 10, d = 10, η = 1/λ_max(Gram) ∝ 1/m as in the theorem): (1) real loss slice along the GD displacement and a random direction — convex along the training direction at m = 8 192, not at m = 32; (2) linear convergence for widths 32…8 192 and the relative spectral movement 1.83 → 0.10 and activation flips 13.4% → 2.0% (≈ 1/√m); (3) width-condition orders vs n.

## arora2019fine [91] — Arora, Du, Hu, Li, Wang, ICML 2019 (arXiv 1901.08584)
- Model: f(x) = m^{−1/2} Σ a_r relu(w_rᵀx), w_r ~ N(0, κ²I), a_r = ±1 fixed; H∞_ij = E[x_iᵀx_j 1{wᵀx_i ≥ 0, wᵀx_j ≥ 0}] (closed form x_iᵀx_j(π − arccos x_iᵀx_j)/(2π) for unit x). Theorem 4.1: ‖y − u(k)‖₂ = √(Σ(1 − ηλ_i)^{2k}(v_iᵀy)²) ± ε (small κ, large m, η = O(λ₀/n²)). Theorem 5.1 (informal Thm 3.3): population loss ≤ √(2yᵀ(H∞)⁻¹y/n) + lower-order terms, width-independent. Sec. 6: provably learnable function classes.
- Check: closed form vs Monte Carlo 3×10⁻³; n = 60, d = 10, m = 8 000, κ = 1 (we keep u(0) in the prediction because κ is not small): GD residual within 0.021√n of the spectral prediction; structured labels sign(x₁) vs random: yᵀ(H∞)⁻¹y 99.1 vs 351.5, top-10 energy 76% vs 12%, loss at k = 100 0.005 vs 0.023.
- Notes: correct (Theorems 4.1, 5.1; example H = diag(2, 0.2), η = 0.25 → factors 0.5, 0.95, yᵀH⁻¹y = 5.5 verified).
- Survey reference text: correct.
- Visuals: (1) real loss on a weight-space slice spanned by the top/bottom kernel directions (elongated valley, GD on the slice); (2) actual vs predicted loss curves for structured/random labels and the label energy per eigenvalue.

## nguyen2021proof [92] — Q. Nguyen, ICML 2021 (arXiv 2101.09612)
- Theorem 2.2: deep ReLU net (standard parameterization, all layers trained by GD, square loss) with last hidden width n_{L−1} ≥ N. With α₀ = σ_min(F⁰_{L−1}), λ̄_l = ‖W⁰_l‖₂ + C_l: if conditions (7)–(9) hold and η satisfies (10), then Φ(θ_k) ≤ (1 − ηα₀²/8)^k Φ(θ₀) (11). Proof tracks only F_{L−1} via the Lipschitz property of ReLU (no NTK / activation-pattern tracking).
- Sec. 3.1: W_L = 0 and the other layers scaled by large β satisfy (7)–(9) → width N suffices. Sec. 3.2: LeCun init, two layers, width Ω(N²). Corollary 3.2 (Sec. 3.3): LeCun-type init, one layer of width Ω̃(N³ max(m, n₀)³), others Θ(m). Lemma 3.3: σ_min(F_{L−1})² = Θ(n₀ Π n_l β_l²).
- Check (L = 3, widths 4→8→3→1, N = 3): smallest β meeting (7)–(9) is ≈ 212, α₀ ≈ 6 685, and (10) then forces η ≈ 1.7·10⁻¹⁶ — guaranteed rate ≈ 10⁻⁹ per step. Eq. (11) and σ_min ≥ α₀/2 hold at all 4 000 steps. An ordinary run (β = 1, step 0.23) reaches 3·10⁻²⁹ with σ_min growing 0.149 → 0.694 (empirical, no guarantee). Worth stating in the survey if it describes the result as practical: it is an existence/width result.
- Notes: correct (worked example verified). They call the result "Theorem 2.2" correctly.
- Survey reference text: correct.
- Visuals: (1) last-layer least-squares slice with features frozen at a chosen training step (strictly convex bowl, GD on it); (2) theorem run vs bound (11) and ordinary run; σ_min/α₀ for both.

## mei2019mean [93] — Mei, Misiakiewicz, Montanari, COLT 2019 (PMLR 99; arXiv 1902.06015)
- Theorem 1 (A) (noiseless SGD, fixed coefficients): sup_k |R_N(θ^k) − R(ρ_{kε})| ≤ Ke^{KT}(√log N + z)/√N + Ke^{KT}(√(D + log N) + z)√ε (Eq. 9) — the N requirement is dimension-free; D enters only via ε ≪ 1/D. (B): general coefficients, e^{KT³}. Noisy SGD: arXiv Theorem 2 = COLT Theorem 4; part (B) is not dimension-free (arXiv Remark 3.3 = COLT Remark 5). Anisotropic-Gaussian example: arXiv Theorem 3 = COLT Theorem 6 (N = O(1) instead of O(d)). Kernel limit: COLT Theorem 8.
- **Notes error:** they attribute the noiseless approximation bound to "Theorem 3"; it is Theorem 1 in both versions (their "Theorem 4 … Remark 5" for the noisy case follows the COLT numbering).
- Our replication of the Sec. 3.2 example (Δ = 0.6, truncated ReLU 0.5/1.5/−2.5/7.5 — constants are ours, ε = 0.05/d): sup-gap to an N = 3 200 proxy 0.240 → 0.106 → 0.115 (N = 50, 200, 800) at d = 20 and 0.202 → 0.065 → 0.038 at d = 80 — no growth with d; the plateau at d = 20 is the √ε term plus proxy noise. Closed-form R∞(r₁, r₂) verified against a 1 500-neuron network at d = 400 and its d → ∞ limit.
- Survey reference text: correct.
- Visuals: (1) neuron cloud (‖P_V w‖, ‖P_⊥ w‖) replayed on the R∞ surface (ends near its minimum (1.55, 0.70)); (2) risk curves for N = 50/200/800 vs the proxy and the gap vs N for both d.

## lu2020mean [94] — Lu, Ma, Lu, Lu, Ying, ICML 2020 (PMLR 119; arXiv 2003.05508)
- Model (2.1)–(2.2): Ẋ_ρ(x, t) = ∫ f(X_ρ(x, t), θ) ρ(θ, t) dθ, X(0) = w₂x, E(ρ) = E½(⟨w₁, X(1)⟩ − y)². ICML numbering: Theorem 1 (well-posedness, Lipschitz in W₂), Theorem 2 (first variation δE/δρ = E f(X, θ) p(x, t), adjoint p), Lemma 1 (‖p(·, t)‖ ≥ e^{−(C₁+C₂r)}E(ρ)), Theorem 3 (E > 0 ⇒ descent direction ⇒ every local minimizer global), Theorem 4 (informal; positively p-homogeneous blocks, if the Wasserstein flow converges and the support separates two spheres, the limit has E = 0). arXiv numbers: Thm 2.3, 3.1, 3.2, 3.9.
- Notes: consistent with ICML numbering (Theorems 2–4); Euler example (1 + 1/L)^L verified.
- Our model: L = 16 layers × M = 32 tanh blocks, 1/(LM) scaling, 1-D input lifted to ℝ², target sin(πx), GD on all blocks: loss 0.107 → 3.1·10⁻⁵; adjoint ratio ‖p(t)‖/‖p(1)‖ ≥ 0.34 over depth and training; L = 4/16/64 give nearly the same loss curves (depth limit).
- Survey reference text: correct.
- Visuals: (1) forward flow of the data through depth (3-D curves) at a chosen training step; (2) loss vs depth L and the adjoint profile.

## li2020learning [95] — Li, Ma, Zhang, COLT 2020 (PMLR 125; arXiv 2007.04596)
- Setup: x ~ N(0, I), f*(x) = Σ aᵢ|w*ᵢᵀx|, orthonormal W*, aᵢ ∈ [1/(κd), κ/d]; student f_W = (1/m)Σ‖wᵢ‖ReLU(wᵢᵀx) (Eq. 1.2); Algorithm 1 = truncated GD in two stages (λ₀, λ₁). Theorem 1.1: L(Ŵ) ≤ O(1/d^{1+Q}) with poly samples/width/time. Theorem 1.2: any poly(d)-size feature map or kernel (NTK included) has L = Ω(1/d). Eq. (2.1): Hermite expansion into tensor-decomposition terms.
- Notes: correct in substance; they call the lower bound "Theorem 2" (paper: Theorem 1.2). Survey title matches the COLT title ("…Two-Layer Neural Networks beyond NTK"; arXiv adds "ReLU").
- Our d = 3 replication (exact population loss via the arc-cosine kernel, m = 60, truncated GD): loss 0.234 → 3.4·10⁻⁵, neuron mass within 15° of ±eᵢ 4.4% → 97.0%; best fit on the 60 frozen initial features 1.2·10⁻³ (35× worse; crossed after ≈1 240 steps). At d = 8 and 16 (m = 20d, 3 000 steps) the trained loss sat at the random-feature level — the theorem's second stage needs far longer; the page says the d = 3 run is an illustration, not the asymptotic theorem.
- Survey reference text: correct.
- Visuals: (1) neurons in ℝ³ gathering on the six teacher rays; (2) trained loss vs the same-width fixed-feature optimum and the alignment share.

## xu2023over [96] — Xu & Du, COLT 2023 (arXiv 2302.10034)
- Theorem 1 (informal): n ≥ 2 students, d = Ω(log(n/δ)), small init, suitable η ⇒ GD converges globally with L ≤ poly(n, ‖v‖, η⁻¹)·T⁻³. Theorem 2 (informal): gradient flow L(w(t)) ≥ (Γ₁t + Γ₂)⁻³. n = 1: exp(−Ω(T)) (Yehudai & Shamir 2020). Eq. (3): closed-form gradient (Safran & Shamir) — note the w̄ᵢ (normalized) in the sin term.
- Fig. 1 setting reproduced (σ = 0.1, η = 0.05, ‖v‖ = 1, d = 10 our choice): n = 1 reaches L < 10⁻²⁸ in 750 steps; fitted slopes −2.99/−2.99/−2.98 for n = 2/3/5; log L/log(ηT) → −2.96/−3.01/−3.10 at T = 6·10⁴. Mechanism checked on a 2-D slice: two students summing to v have loss ∝ |b|^{3.000} across the minimum segment, and GD there decays as t^{−3.00}.
- Notes: correct (Theorems 1–2, worked example (0.4, 0.6) split verified).
- Survey reference text: correct.
- Visuals: (1) side-by-side n = 1 bowl vs n = 2 cubic trough with GD; (2) Fig. 1 replication and the −3 ratio.

## chizat2019lazy [110] — Chizat, Oyallon, Bach, NeurIPS 2019 (arXiv 1812.07956)
- κ_h(w₀) (Eq. 1) controls laziness; scaling α: κ_{αh} ∝ 1/α; q-homogeneous models: init scale λ acts like α = λ^q. Theorem 2.2 (h(w₀) = 0, horizon T): ‖w_α − w₀‖ = O(1/α), ‖w_α − w̄_α‖ = O(1/α²), outputs O(1/α). Theorem 2.3: square-loss version without exponential T dependence. Theorem 2.4: over-parameterized, linear convergence for all t. Theorem 2.5: under-parameterized, convergence to a local (not necessarily global) minimum. Experiments: lazy regime generalizes worse (two-layer nets, CIFAR-10 CNNs).
- Check: toy h(w) = w₁ + ½w₂ + w₁w₂: slopes −0.96 and −1.97 for the two Theorem 2.2 quantities over α = 2…64. Fig. 1 setting (n = 15, m = 20, d = 2, symmetrized init; τ values ours): τ = 0.1 fits with relative weight change 3.12, τ = 5 with 0.008; change ∝ τ^{−1.94} for τ ≥ 1. (First attempt used a step too large for large τ and blew up early — fixed with η = 1/λ_max of the tangent kernel.)
- Notes: correct (Theorem 2.2 statement; example h = w + w², α = 100 verified).
- Survey reference text: correct.
- Visuals: (1) model vs linearization surfaces on a 1/α window with GD on each and the gap ×α²; (2) Fig. 1(a)–(b) replication and the τ⁻² scan.

## nesterov2004introductory [44] — Y. Nesterov, Introductory Lectures on Convex Optimization, Kluwer 2004 (book; not available for a line-by-line check)
- Used for: L-smoothness, the descent lemma |f(y) − f(x) − ⟨∇f(x), y − x⟩| ≤ (L/2)‖y − x‖², the gradient step decrease h(1 − Lh/2)‖∇f‖² (best h = 1/L), and min_{i≤k}‖∇f(x_i)‖² ≤ 2L(f(x₀) − f*)/(k + 1), i.e. O(ε⁻²) without convexity. Nesterov & Polyak 2006 cite this book's inequality (1.2.13) for the O(k^{−1/2}) gradient rate; we give no other theorem numbers.
- Notes: correct; they say themselves the book was not available and label the derivations pedagogical. They label it "Ref. [36]" (old numbering) — survey number [44].
- Survey reference text: correct (Kluwer 2004).
- Visual (one tab): f = x²/4 − cos(2y)/4 (non-convex, L = 1 exactly): GD with h = 1/L, 1.9/L, 2.2/L; the measured min‖∇f‖² stays under the bound; with 2.2/L f increases.

## carmon2020lower [45] — Carmon, Duchi, Hinder, Sidford, Math. Program. 184 (2020) (arXiv 1710.11606)
- Hard instance f̄_T (Eq. 10) with Ψ, Φ; zero-chain (Def. 3, Observation 1); Lemma 1 (Ψ(x)Φ′(y) > 1 for x ≥ 1, |y| < 1), Lemma 2 (‖∇f̄_T‖ > 1 unless |x_i| ≥ 1 ∀i), Lemma 3 (gap ≤ 12T, ‖∇f̄_T‖ ≤ 23√T, Lipschitz derivatives). Theorem 1 (deterministic) / Theorem 2 (randomized): c_p Δ L_p^{1/p} ε^{−(p+1)/p} queries, sharp (GD p = 1, cubic Newton p = 2, p-th order regularization).
- Check: gradient, zero-chain, Lemmas 1–3 verified numerically; GD from 0 needs 15…146 steps for T = 1…20 (≈ 6.4 per coordinate); for T = 2, x₂ first moves at step 8 when x₁ = 0.624 > ½.
- Notes: correct (exponent table p = 1, 2, 3 verified). They label it "Ref. [37]" (old numbering) — survey number [45].
- Survey reference text: correct.
- Visuals: (1) f̄₂ surface with GD's L-shaped path and ‖∇f‖ vs the Lemma 2 threshold; (2) GD steps vs T and the ε^{−(p+1)/p} bounds.

## robbins1951stochastic [46] — Robbins & Monro, Ann. Math. Statist. 22(3) 1951 (JSTOR copy)
- Rule (7) x_{n+1} − x_n = a_n(α − y_n); b_n = E(x_n − θ)² (9). Conditions: (4) bounded responses, (5) M < α left of θ and > α right, (6) Σa_n² < ∞, (26) (implies (27) Σa_n = ∞); type 1/n (30). Theorem 1 (type 1/n, (4), (5′)) and Theorem 2 (M nondecreasing, M(θ) = α, M′(θ) > 0): b_n → 0, hence x_n → θ in probability (not almost surely — later work).
- Check (M = tanh, α = ½, U(−1, 1) noise, 200 runs): b_n = 0.0077 → 0.00075 → 2.9·10⁻⁵ at n = 10², 10³, 2·10⁴ for a_n = 2/n; ≈ 0.053 plateau for constant 0.2; ≈ 0.51 for 2/n².
- Notes: correct and careful (they flag Σa_n = ∞, Σa_n² < ∞ as a later shorthand). Label "Ref. [38]" (old) — survey [46].
- Survey reference text: correct.
- Visuals: (1) noisy-gradient descent on a 2-D convex function with three gain sequences; (2) the original 1-D root-finding paths and b_n curves.

## arjevani2023lower [48] — Arjevani, Carmon, Duchi, Foster, Srebro, Woodworth, Math. Program. 199 (2023) (arXiv 1912.02365)
- Probability-p zero-chain (Def. 2); oracle (17) on the Carmon chain F_T (15): [g_T]_i = ∇_iF_T (1 + 1{i > prog_{1/4}(x)}(z/p − 1)), z ~ Bernoulli(p). Lemma 3: unbiased, variance ≤ 23²(1 − p)/p. Lemma 1: zero-respecting methods need ≳ (T − log(1/δ))/(2p) rounds. Theorem 1: Ω(ΔLσ²ε⁻⁴) (bounded variance; tight, SGD). Theorem 2: Ω(ΔL̄σε⁻³ + σ²ε⁻²) under mean-squared smoothness (tight, SPIDER/SNVRG). Theorem 3: all randomized algorithms, any batch K.
- Check: unbiasedness, variance bound, zero-chain; SGD (step 0.05) needs 22/44/90 (p = 1), 27/51/98 (p = 0.2), 84/170/336 (p = 0.05) queries for T = 4/8/16.
- Notes: correct (the SGD ε⁻⁴ upper-bound derivation and both lower bounds match). They label it "Ref. [40]" (old numbering) — survey number [48].
- Survey reference text: correct (Math. Program. 199, 2023).
- Visuals: (1) f̄₂ with exact GD vs the probability-p oracle (x₂ stuck until lucky); (2) queries vs T for p ∈ {1, 0.2, 0.05} and the ε⁻⁴ / ε⁻³ bounds.

## cohen2021gradient — Cohen, Kaur, Li, Kolter, Talwalkar, ICLR 2021 (arXiv 2103.00065) [survey 16]
- Empirical paper (no convergence theorem). Sec. 2: on a quadratic GD diverges along any eigenvector with eigenvalue a > 2/η (Prop. 1, App. B); Fig. 2 example eigenvalues 20 and 1 with η = 0.11.
- Eq. (1): maximum stable sharpness MSS_Polyak = (2 + 2β)/η (previously Goh 2017), MSS_Nesterov = (2 + 2β)/(η(1 + 2β)) (new).
- Sec. 3.1 progressive sharpening: while S < 2/η full-batch GD has an "overwhelming tendency" to increase S. Sec. 3.2 edge of stability: S hovers at or just above 2/η, loss non-monotone short-term, decreasing long-term. Sec. 4 / App.: FC, VGG, ResNet on CIFAR-10 subsets, plus Transformer (WikiText-2), deep linear net, 1-d regression; MSE and cross-entropy (with CE the sharpness drops at the end of training, App. C). Fig. 7: with momentum S plateaus at the MSS. Sec. 6: SGD does not flatline at one value.
- Notes: correct; quadratic derivation and "empirical, not a theorem" caveat match the paper.
- Survey (Sec. VII-B): consistent ("hovers at or slightly above", momentum-modified threshold).
- Visuals: (quad, 3D) Fig. 2 quadratic f = ½(20x² + y²), GD with η slider, |x_t| chart; (mlp) offline replication, tanh MLP 10–48–48–1 on 128 synthetic points, full-batch GD, sharpness by power iteration: S₀ = 4.05; η = 0.1 saturates at 7.75 < 20; η = 0.3/0.36/0.42 end at 6.66/5.55/4.75 vs 2/η = 6.67/5.56/4.76 with non-monotone loss; (mom) Eq. (1) curves vs live spectral-radius bisection.

## lewkowycz2020large — Lewkowycz, Bahri, Dyer, Sohl-Dickstein, Gur-Ari, arXiv 2003.02218 (2020) [survey 17]
- Sec. 2.1 warmup model: x = 1, y = 0, f = n^{-1/2} vᵀu, L = f²/2, kernel λ = n⁻¹(‖u‖² + ‖v‖²). Eqs. (6)–(7) are exact ("no higher-order terms were neglected"): f_{t+1} = (1 − ηλ_t + η²f_t²/n) f_t, λ_{t+1} = λ_t + η f_t²/n (ηλ_t − 4).
- Phases: lazy η < η_crit = 2/λ₀ (kernel constant as n → ∞); catapult 2/λ₀ < η < η_max = 4/λ₀ (loss grows for ~log n steps, kernel drops below 2/η, converges to a flatter minimum); divergent η > 4/λ₀. Sec. 2.2 extends to the full model (Eq. 9–11) along the top kernel eigenvector.
- Experiments (MSE, SGD, wide FC ReLU nets on MNIST/CIFAR-10, WRN 28-10 on CIFAR-10): η_crit = 2/λ₀ matches; η_max ≈ c/λ₀ with c ≈ 12 for ReLU nets; best test accuracy often in the catapult phase.
- Notes: correct. Their teaching toy L(w) = ½(1 − w²)² has L'' = −2 + 6w² (checked); the notes correctly flag it as not the paper's model and that the upper edge is model-dependent.
- Survey: consistent ("4/λ₀ in their solvable model and larger for ReLU networks"). Venue: arXiv preprint (as listed).
- Visuals: (spike, 3D) n = 1 model ½(uv)² from (2, 0.35), λ₀ = 4.12, η·λ₀ slider: 2.6 → peak loss 0.484 (start 0.245), ends at λ = 1.21 < 2/η; 4.2 diverges; (width) Eqs. (6)–(7) with n slider, loss peak at steps 2/6/9 for n = 10²/10⁴/10⁶ (log n); (phases) final λ/λ₀ vs ηλ₀ for chosen n, divergence at exactly 4 (3.98 converges, 4.02 diverges).

## foret2021sharpness — Foret, Kleiner, Mobahi, Neyshabur, ICLR 2021 (arXiv 2010.01412) [survey 34]
- Eq. (1): min_w L^SAM_S(w) + λ‖w‖², L^SAM_S(w) = max_{‖ε‖_p ≤ ρ} L_S(w + ε), motivated by a PAC-Bayes bound (Theorem 1 informal, Theorem 2 in App. A). Eq. (2): first-order inner solution ε̂ = ρ sign(∇L)|∇L|^{q−1}/(‖∇L‖_q^q)^{1/p}, = ρ∇L/‖∇L‖ for p = 2. Eq. (3): ∇L^SAM ≈ ∇L|_{w+ε̂} (second-order terms dropped). Algorithm 1; ≈ 2× the gradient cost.
- Reported: error reductions on CIFAR-10/100, ImageNet, fine-tuning; label-noise robustness; m-sharpness (Sec. 4.1: per-subset inner max, smaller m generalizes better and correlates better with the gap); lower Hessian spectrum for WRN-40-10 trained with SAM (Sec. 4.2, Fig. 3).
- Notes: correct (the "max over ‖ε‖ ≤ 0.1" example gives 0.01 vs 0.10 — checked). Header carries old numbering ("Ref 100", "Paper 70"); the Atlas maps it to survey [34].
- Survey (VII-C, Eq. 20): consistent.
- Toy finding worth knowing: on a 2-well toy (sharp λ_max 39.2, flat 3.5, equal depth), descent on the *exact* SAM objective shrinks the sharp basin (25% → 17% of starts at ρ = 0.4), and the practical one-step update shrinks it more (→ 11%) but leaves 23% of starts jittering between the wells instead of reaching the flat one (e.g. from (−1.36, −0.32) at ρ = 0.3 it hovers near (−0.18, 0.14), L ≈ 0.89). The one-step approximation is not the exact minimax.
- Visuals: (objective, 3D) L and the exact SAM objective side by side with GD / one-step SAM / exact-objective descent; (basins) 41×41 start-grid fate maps for the three methods and ρ ∈ {0.1, 0.2, 0.4}.

## damian2023self — Damian, Nichani, Lee, ICLR 2023 (arXiv 2209.15594) [survey 35]
- Def. 1 sharpness; Def. 2 progressive sharpening coefficient α = −∇L·∇S, Assumption 1 α > 0; β = ‖∇S‖². Sec. 4.1 four stages; simplified dynamics Eq. (3): x_{t+1} = −(1 + ηy_t)x_t, y_{t+1} = y_t + η(α − βx_t²/2). ODE Eq. (4) X' = XY, Y' = α − βX²/2 with fixed point (δ, 0), δ = √(2α/β); Lemma 3: g = h(βX²/(2α)) + Y²/α conserved, h(z) = z − log z − 1; Corollary 1 bounds the swings.
- Main result: constrained trajectory θ†_{t+1} = proj_M(θ†_t − η∇L(θ†_t)), M = {S ≤ 2/η, ∇L·u = 0} (Eq. 2); Theorem 1 (Assumptions 1–5) predicts loss, sharpness and deviation for T = O(ε⁻¹) steps.
- App. B toy L(x, y, z) = (2/η + √β y)x²/2 − (α/√β)y − z satisfies all Sec. 4 assumptions; GD on its (x, y) part is exactly Eq. (3) (checked to 1e-13).
- Check-script finding: the discrete map (3) is not conservative — its period-2 orbit (±δ, 0) is a weakly unstable focus (two-step eigenvalues (1 − δ²) ± 2iδ, |·| ≈ 1 + δ²), so iterates follow the ODE loops for a few cycles and then spiral outward (g: 0.38 → 2.0 in 450 steps at α = 0.002). Consistent with Theorem 1 being finite-horizon. Over the shown window the mean sharpness is 1.9992 vs 2/η = 2.
- Notes: correct; they list the source as "Ref. [30]" (old numbering). The notes write the expansion as "L(θ + xu) ≈ L + x∇L·u + x²/2 S" — fine.
- Survey (VII-B, Eq. 19, Fig. 17): consistent with the paper (schematic figure).
- Visuals: (stages, 3D) the App. B toy with GD zig-zagging across the trough, |x_t| and S_t − 2/η charts, stage readout; (orbit) (|x|, S − 2/η) phase plane, ODE orbits vs GD iterates, click to restart.

## kalra2024warmup — Kalra & Barkeshli, NeurIPS 2024 (arXiv 2406.09405) [survey 36]
- Abstract/Sec. 1: the overwhelming benefit of warmup is to let the network tolerate a larger η_trgt by moving it to better-conditioned regions. Sec. 3, Eq. (1): x_{t+1} = (1 − η_t λ^H_t)x_t, λ^H_{t+1} = λ^H_t + η_t(α − βx_t²) (self-stabilization with time-dependent η); four steps (approach, blow-up, sharpness reduction, return). Sec. 4.1: C1 natural progressive sharpening (α > 0: head-on collision, persistent catapult cycle η_t ≈ 2/λ^H_t), C2 natural sharpness reduction (α < 0: separated catapults); which one depends on initialization/parameterization (µP / large-init SP).
- Sec. 4.3 / App. B.1: Adam is GD in θ̃ = P^{1/2}θ, stability set by λ_max(P⁻¹∇²L) with P_t = (1 − β1ᵗ)diag(√v̂_t) + ε (Eq. 9); late-time threshold (2 + 2β1)/((1 − β1)η) (citing Cohen et al. 2022); initial P⁻¹H ~ 10⁵ (µP) to 10⁶ (SP) because P₁ = (1 − β1)diag(|g₀|) + ε. Sec. 6: GI-Adam v₀ = g₀² lowers the initial preconditioned sharpness "by around two orders of magnitude (more precisely by a factor of √(1 − β2))" and equals Adam with automatic warmup η_t = η_trgt√(1 − β2ᵗ); catapult-based choice of η_init.
- Check-script reproduction: the GI/Adam step-1 ratio is 0.0317 vs √(1 − β2) = 0.0316 on a 200-d quadratic. Note: √(1 − β2) ≈ 1/32 is about 1.5 orders of magnitude, so "two orders" is the paper's own rounding; the survey repeats "roughly two orders of magnitude" — acceptable but loose.
- Toy finding: deterministic GD on the n = 1 linear model cannot be rescued by warmup — its residual decays to ~1e-30 before η_t crosses 2/λ, so the catapult comes too late and λ grows once ηλ > 4. With small label noise (labels ±0.02, SGD) warmup works: η_trgt·λ₀ = 6 diverges without warmup (20/20 seeds), trains with ≥ 50 warmup steps (20/20); at η_trgt·λ₀ = 12 warmup ≥ 50 steps trains 10/10, shorter 0/10.
- Notes: correct (pedagogical examples 2/20 vs 2/5 and diag(20,1) rescaled to diag(0.2,1) are labelled as illustrations). Old numbering in header ("Ref 99", "Paper 69").
- Survey (VII-B "Warmup"): consistent.
- Visuals: (warm, 3D) SGD on the linear toy with warmup seg (none/20/50/200/400), λ_t vs 2/η_t; (sweep) success map η_trgt·λ₀ × T_wrm; (adam) λ_max(P⁻¹H) over the first 300 steps for Adam vs GI-Adam with the (2 + 2β1)/((1 − β1)η) line.

## li2023convergence — Li, Rakhlin, Jadbabaie, NeurIPS 2023 (arXiv 2304.13972) [survey 42]
- Definition 3.2: (ρ, L0, Lρ) smooth if ‖∇²f(x)‖ ≤ L0 + Lρ‖∇f(x)‖^ρ a.e.; Assumption 2: 0 ≤ ρ < 2 (ρ = 1 is (L0, L1) smoothness of Zhang et al. [41], Remark 3.3). Assumption 3: bounded noise (Theorem 4.2: sub-Gaussian, Assumption 4). Lemma 3.4: local Lipschitz gradient near any point with bounded gradient.
- Theorem 4.1: with high probability ‖∇f(x_t)‖ ≤ G along the trajectory and (1/T)Σ‖∇f(x_t)‖² ≤ ε² with O(ε⁻⁴) gradients; no condition on β_sq; β, λ = O(ε²); dependence λ⁻² noted as a limitation. Sec. 6: VRAdam, O(ε⁻³). Prop. B.1: rational functions and double exponentials are (ρ, L0, Lρ) smooth with 1 < ρ < 2, not necessarily (L0, L1).
- Notes: correct; the e^x illustration is labelled as teaching material. Header uses "Ref. [34]" (old numbering).
- Survey (VII-D): consistent ("sub-quadratic function of the gradient norm", O(ε⁻⁴), VR O(ε⁻³), bounded gradients along the trajectory).
- Visuals: (adam, 3D) f = cosh(1.5x) + (y² − 1)² from (3.6, 1.9): GD η = 0.05 leaves after 2 steps, GD η = 0.01 converges, Adam η = 0.05 converges with max ‖∇f‖ = 167.2 = its start (no faster than tuned GD — stated on the page); (classes) |f″| vs |f′| log–log for eˣ (slope 1), x⁴ (2/3), 1/x (3/2), e^(eˣ) (1.11 and rising).

## zhang2019lookahead — Zhang, Lucas, Hinton, Ba, NeurIPS 2019 (arXiv 1907.08610) [survey 97]
- Algorithm 1 / Eq. (1): fast weights θ take k inner steps from φ, then φ_{t+1} = φ_t + α(θ_{t,k} − φ_t); unrolled, an exponential average of the inner-loop endpoints. Prop. 1: loss-minimizing α on a quadratic (adaptive α with empirical-Fisher estimate, clipped; a fixed α is used in experiments).
- Sec. 3.1, Eq. (4): noisy quadratic L̂(x) = ½(x − c)ᵀA(x − c), c ~ N(x*, Σ), A, Σ diagonal. Prop. 2: V_SGD = γ²A²σ²/(I − (I − γA)²), V_LA = α²(I − (I−γA)^{2k}) / (α²(I − (I−γA)^{2k}) + 2α(1−α)(I − (I−γA)^k)) · V_SGD; prefactor < 1 for α ∈ (0, 1). Fig. 3: at matched steady-state risk Lookahead converges faster. Sec. 3.2 / Fig. 4: wrapping classical momentum improves the rate in the under-damped regime (k = 20, α = 0.5).
- Notational slip in the paper: Prop. 2 writes γ²A²Σ² with Σ the noise covariance; the variance that matches simulation uses σᵢ² (the diagonal of Σ), i.e. γ²aᵢ²σᵢ² (checked by 200 000-run Monte Carlo, within 2%).
- Table 1 (reported): CIFAR-10/100 95.27 ± .06 / 78.34 ± .05 (Lookahead) vs SGD 95.23 ± .19 / 78.24 ± .18, Polyak 95.26 ± .04 / 77.99 ± .42, Adam 94.84 ± .16 / 76.88 ± .39 — the notes' numbers are right and their "differences are small" caveat is fair.
- Notes: correct (worked φ updates 10 → 9 → 8; independent-endpoint EMA variance ασ²/(2 − α) = σ²/3 at α = ½ is labelled as a teaching model). Header "Ref 94"/"Paper 65" is the old numbering.
- Survey (VII-A): consistent ("φ ← φ + α(θ_k − φ)", variance reduction, robustness to inner hyperparameters).
- Visuals: (noisy, 3D) the noisy quadratic with a = (1, 0.3), σ² = (1, 1), γ = 0.6: SGD vs Lookahead fast/slow weights, live time-averaged excess loss vs Prop. 2 (0.229 vs 0.078 at k = 5, α = 0.5); (ratio) V_LA/V_SGD vs α for k = 1, 5, 20.

## wang2022analyzing — Zhouzi Li, Zixuan Wang, Jian Li, NeurIPS 2022 (arXiv 2207.12678) [survey 98]
- **Survey reference error:** the survey lists the authors as "Z. Wang, Z. Li, and J. Li"; the paper (arXiv and NeurIPS) lists Zhouzi Li, Zixuan Wang, Jian Li — the first author is Zhouzi Li. The Atlas key `wang2022analyzing` is internal and kept. The notes have the correct order.
- Sec. 3.1: four phases (I sharpness < 2/η and rising; II > 2/η, |Dᵀv₁| grows; III > 2/η but falling, loss may rise; IV < 2/η, |Dᵀv₁| decays). Sec. 3.2, Eq. (2): for f = Aᵀh(W, x), d‖A‖²/dt = −(4/n)DᵀF; the Gram matrix is dominated by M_W which scales with ‖A‖², so ‖A‖² and sharpness move together "most of the time" (high-frequency sharpness oscillations unexplained, Sec. 5). Sec. 4: f(x) = AᵀWx/√m, labels ±1; Theorem 1 (progressive sharpening under Assumptions 4.1–4.2 and width/sample conditions), Theorem 2 (under 4.3–4.5: a sharpness above (2/η)(1 + δ) later falls below it; |Dᵀv₁| grows by 1 + c₇ per step above it; the non-top residual tracks a converging sequence).
- Notes: correct and appropriately cautious. Survey (VII-B "Mechanisms"): consistent apart from the author order.
- Replication (check script): two-layer linear net n = 32, d = 48, m = 64, η = 1.973: sharpness 0.49 → peak 1.58 at step 17, 76 excursions above 2/η = 1.014 until step 225, converges at 1.006; ‖A‖² and sharpness agree in direction in 91% of sliding 10-step windows over the first 300 steps (98% over the whole run); Δ‖A‖² has the sign of −DᵀF at every step.
- Visuals: (sign, 3D) scalar model (aw − 1)² with the surface coloured by sign(D·F), GD with η slider, sharpness and a² charts, phase readout; (trace) the replication with shaded above-threshold steps and the loss.

## chen2023beyond — Chen & Bruna, ICML 2023 (arXiv 2206.04172) [survey 99]
- Theorem 1 (1-D, local): f‴(x*) ≠ 0 and 3f‴² − f″f⁗ > 0 ⇒ fixed points of F² for some η slightly above 2/f″(x*). Lemma 1: higher-order cases (sin). Prop. 1–2: squared loss on many base functions and their compositions qualify. Theorem 2: f = ¼(x² − μ)², η = K/μ with 1 < K < √4.5 − 1 ≈ 1.121, x₀ ∈ (0, √μ): converges to a period-2 orbit (except a null set). Theorem 3: f = ½(xy − μ)², 1 < K < 1.5, x, y > 0 throughout ⇒ |x − y| → 0 on the iterates with xy > μ; Prop. 3: for 1 < K < 1.121 the 2-D GD converges to the same period-2 orbit. Theorem 4 / Prop. 4 single neuron; Theorem 6 matrix factorization satisfies the Theorem 1 condition.
- Check-script findings: Theorem 2's range is sufficient, not sharp — the 2-cycle persists until K ≈ 1.237 (period 4), 8 at 1.295, then chaos. Theorem 3's positivity assumption matters: from (1.6, 0.25) y turns negative and the iterates leave the setting; (1.3, 0.5) satisfies it and converges to {0.8729, 1.0911} at K = 1.05 (two-step multiplier 0.595).
- Notes: correct, including the worked numbers (x₁ = 1.1024, x₂ ≈ 0.8532, condition 96 > 0). Old numbering in header ("Ref 96").
- Survey (VII-B): consistent ("local conditions involving third-order derivatives", "stable period-two orbits", teacher–student settings).
- Visuals: (orbit, 3D) ½(xy − 1)² with K slider, orbit markers, |x − y| and residual charts; (bif) bifurcation diagram of the quartic with the Theorem 2 range and the first doubling marked.

## mohtashami2023special — Mohtashami, Jaggi, Stich, ICML 2023 (arXiv 2205.15142) [survey 100]
- Def. 2: μ-one-point-strongly-convex (OPSC) w.r.t. x* over M. Lemma 1: L_global-smooth f with a local minimum x† that is μ†-OPSC over a set M (diameter r), a surrounding shell P(M) that is L-smooth and μ*-OPSC towards the global minimum x*, x† far enough from x*: with 2/μ† < γ < μ*/L², GD that reaches M leaves it to a point closer to x*. Theorem 1 (informal; formal in Sec. 4.1): on a class C_l, large-step GD from random init converges to x* a.s., small-step GD converges to x† with positive probability. Theorem 2: probability of never visiting a region X far from x* is lower-bounded by a quantity increasing in the step and decreasing in vol(X). Experiments: the large-step benefit persists in full-batch GD (noise is not sufficient); gradient norm can grow temporarily.
- Notes: correct; worked multipliers 1 − 0.1·10 = 0 and 1 − 0.25·10 = −1.5 checked. Old numbering in header ("Ref 97").
- Survey (VII-B "Large steps change the destination"): consistent.
- Toy finding: bowl + narrow spurious well (λ 15.46 vs 0.3): η ≤ 0.12 traps ~15% of 600 random starts at the spurious minimum; η = 0.15–0.2 traps none there but leaves 14% oscillating inside the narrow well; η ≥ 0.3 sends 100% to the global minimum; no divergence below 2/0.3. From (−0.55, 0.62): η = 0.05 → spurious, η = 0.3/0.8 → global, with ‖∇f‖ rising 1.41 → 1.75 during the escape.
- Visuals: (escape, 3D) small vs large η from the same start with ‖∇f‖ chart; (odds) outcome fractions vs η (log) with 2/λ lines.

## agarwala2023sam — Agarwala & Dauphin, ICML 2023 (arXiv 2302.08692) [survey 101]
- Unnormalized SAM, Eq. (6): θ_{t+1} = θ_t − η∇L(θ_t + ρ∇L(θ_t)). Quadratic regression model; Eq. (7): residual dynamics governed by (1 + ρJJᵀ)JJᵀ. Theorem 2.1: at initialization (full batch) SAM's first-step change of the singular values is negative but ∝ (z·w_α)², so weak for small learning rates. Eq. (12) on ½xᵀHx: x_{t+1} − x_t = −η(H + ρH²)x_t. **Theorem 2.2** (App. B.1): for a C² model with MSE near a zero-residual point, convergent if ηλ_i(1 + ρλ_i) < 2 − ε for all NTK eigenvalues, divergent if > 2 + ε for one ⇒ SAM EOS ηλ_max(1 + ρλ_max) ≈ 2 (Eq. 13). **Theorem 2.3** (App. A.2): SGD with batch fraction β, first-step averages (stronger regularization). Experiments: FC net on 2-class CIFAR with MSE, WRN 28-10 on CIFAR-10.
- **Notes error:** the notes attribute the SAM stability condition to "Theorem 2.3" and the minibatch effect to "Theorem 2.2"; in the paper it is the other way round (Theorem 2.2 = SAM edge of stability, Theorem 2.3 = SGD). Everything else in the notes is right (worked example η = 0.5, ρ = 0.2 → λ_SAM ≈ 2.62; multipliers −0.5 vs −1.4, checked).
- Survey (VII-C): consistent ("predicted largest eigenvalue that depends on the learning rate and the radius").
- Replication (check script): the Cohen-page tanh MLP at η = 0.3: late sharpness GD 6.657 (2/η = 6.667), SAM ρ = 0.05: 5.271 (edge 5.275), ρ = 0.1: 4.571 (edge 4.574) — the SAM edge is reproduced to 0.1%.
- Visuals: (uv, 3D) GD vs unnormalized SAM on ½(uv − 1)² from (2.2, 0.1), η = 0.3: end sharpness 4.915 (GD) vs 4.196 / 2.956 / 2.000 for ρ = 0.05 / 0.1 / 0.2; (mlp) the replication; (edge) λ_SAM(ρ) vs 2/η.

## shin2025sassha — Shin, Lee, Chung, Lee, ICML 2025 (arXiv 2502.18153) [survey 102]
- Algorithm 1: g = ∇f_B(x); ε = ρg/‖g‖₂; g̃ = ∇f_B(x + ε); m = EMA_β1(g̃); every k steps (lazy, k = 10) D = EMA_β2(|H̃|) with H̃ a diagonal Hessian estimate at x + ε and |H| = Σ|H_ii|e_ie_iᵀ (Eq. 5); update x ← x − η D̂^{−1/2} m̂ − ηλx (square root, Sec. 4.2). Theorem 4.4: convex, smooth, lower-bounded (Assumptions 4.1–4.3) ⇒ liminf ‖∇f(x_t)‖ = 0 — a convex result. Def. 4.5 / Corollary 4.6: necessary conditions for linear stability of a minimum for SASSHA in terms of sharpness a = λ_max(E[H]) and Hessian non-uniformity s_k, tightening with η and ρ; P ≈ H⁻¹ methods are stable at minima of any sharpness.
- Table 1 (ResNet-32, CIFAR-100; reported): λ_max(H) / val. acc. — SGD 265 / 69.32, AdaHessian 11 992 / 68.06, Sophia-H 22 797 / 67.76, Shampoo 436 374 / 64.08, SASSHA 107 / 72.14.
- Notes: correct, including the "don't overstate Theorem 4.4" caveat; the worked |H|^{−1/2} example (100 → 10) is checked.
- Survey (VII-C): consistent ("explicitly reduces the sharpness", "stabilizes the Hessian approximation", "lazy Hessian updates", "comparable to, and often better than, first-order baselines").
- Toy (deterministic, exact diagonal Hessian, β1 = β2 = 0, k = 1) on the SAM page's two-well loss: diagonal Newton (g/|H|) stays in the sharp well; SASSHA at η = 0.5, ρ = 0.05 ends in the flat well. Our 1-D linearization (ignoring the SAM term): error multiplier 1 − η√a, so minima with a > 4/η² are unstable (η√a = 3.13 sharp vs 0.94 flat). At η = 0.1 SASSHA stays sharp for ρ ∈ {0, 0.05}; at η = 0.2, ρ = 0 stays sharp but ρ = 0.05 reaches the flat well — the SAM term matters. This linearization is ours, not the paper's Corollary 4.6 (whose extracted formula was not fully legible; the page cites it only qualitatively).
- Visuals: (wells, 3D) diagonal Newton vs SASSHA with η slider and ρ seg; (table) reported Table 1 bars.

## taniguchi2024adopt — Taniguchi et al., NeurIPS 2024 (arXiv 2411.02853) [survey 103]
- Theorem 3.1: RMSprop/Adam bounds have a non-vanishing term growing with the gradient bound unless β2 is tuned. Fixes: normalize by v_{t−1} (Sec. 3.1, cf. AdaShift); with momentum, normalize before the momentum update (Eqs. 13–14). Algorithm 1 (ADOPT) uses max{√v, ε}. Theorem 4.1: Assumptions 2.1–2.3, 2.5 (bounded second moment, no almost-sure gradient bound 2.6), constant α = Θ(1/√T): min_t E[‖∇f(θ_{t−1})‖^{4/3}]^{3/2} = O(1/√T) for any β2 (diminishing α_t = Θ(1/√t) in the appendix). Algorithm 2 clips to avoid early instability from near-zero v₀ entries.
- Toy (Eq. 17): f(θ) = θ on [−1, 1], gradient k² w.p. 1/k, −k otherwise, η_t = 0.01/√(1 + 0.01t), β1 = 0.9. Reported: Adam fails except β2 = 0.999 at k = 10, fails for all β2 at k = 50; AMSGrad much slower at k = 50.
- Replication (8 seeds, 4·10⁵ steps): k = 10 — Adam ends at +0.99/+1.00/+1.00 for β2 = 0.1/0.5/0.9 but reaches −1.00 for both 0.99 and 0.999 (the paper's Fig. 1 shows success only at 0.999; our horizon/seeds may differ); ADOPT ≤ −0.99 for all β2. k = 50 — Adam ≥ +0.53 for all β2, AMSGrad +0.17 to +0.34, ADOPT −0.28 to −0.73 and still decreasing at 4·10⁵ steps (monotone from 10⁴ to 4·10⁵).
- Notes: correct (Theorem 4.1's 4/3–3/2 form is quoted carefully; worked example 4/√2.5 = 2.53 vs 4 checked). Survey (VII-D): consistent ("removes the current gradient from the second-moment estimate and swaps the order", "optimal O(1/√T) for any β2", "without the bounded-noise assumption").
- Visuals: (toy, 3D) θ₁ = the paper's toy, θ₂ a noiseless bowl, Adam / AMSGrad / ADOPT live with k and β2 segs; (beta2) final θ vs β2 for both k.

## gupta2018shampoo — Gupta, Koren, Singer, ICML 2018 (arXiv 1802.09568) [survey 104]
- Algorithm 1 (matrix): L_t = L_{t−1} + G_tG_tᵀ, R_t = R_{t−1} + G_tᵀG_t (L₀ = εI, R₀ = εI), W_{t+1} = W_t − ηL_t^{−1/4}G_tR_t^{−1/4}; memory m² + n² vs m²n², cost O(m³ + n³) vs O(m³n³). Algorithm 2 (order-k tensor): one statistic per mode, power −1/(2k). Lemma 4: (L ⊗ Rᵀ)vec(G) = vec(LGR). Lemma 8: εI + (1/r)Σg_tg_tᵀ ⪯ (εI + ΣGGᵀ)^{1/2} ⊗ (εI + ΣGᵀG)^{1/2} for rank ≤ r gradients. Theorem 7: regret ≤ √(2r) D Tr(L_T^{1/4}) Tr(R_T^{1/4}) = O(√T) (online convex). Experiments: ResNet-32 / Inception on CIFAR-10, ResNet-55 on CIFAR-100, attention model on LM1B; Table 1 steps/s similar to SGD/Adam/AdaGrad except LM1B (3.5 vs ~4.9).
- Check script: Lemma 4 exact; Lemma 8 in 30 random tests (min eigenvalue of the difference 4.4 > 0); notes' diag(2, 1) example gives exactly I.
- Notes: correct. Survey (VII-D, "W ← W − L^{−1/4}GR^{−1/4}, approximates full-matrix AdaGrad at a fraction of the cost"): consistent.
- Visuals: (rotate, 3D) 1 × 2 weight on a rotated quadratic (curvatures 8, 0.2): diagonal AdaGrad 5e-32 aligned vs 4e-5 at 45°, Shampoo 4.0e-11 at every angle, full AdaGrad ~1e-31; (matrix) 8 × 8 matrix least squares with Kronecker curvature: GD 4.6e-4, diagonal AdaGrad 3.1e-4, Shampoo 1.2e-18, full AdaGrad 1.7e-31 after 1500 steps (best step each), memory 64 / 128 / 4096.

## defazio2024road — Defazio, Yang, Mehta, Mishchenko, Khaled, Cutkosky, NeurIPS 2024 (arXiv 2405.15682) [survey 105]
- Eqs. (3)–(5): y_t = (1 − β)z_t + βx_t, z_{t+1} = z_t − γ∇f(y_t, ζ_t), x_{t+1} = (1 − c_{t+1})x_t + c_{t+1}z_{t+1}, c_{t+1} = 1/(t + 1); β = 0 Polyak–Ruppert, β = 1 primal averaging. Theorem 1: convex, G-Lipschitz, γ = D/(G√T): E[F(x_T) − F*] ≤ DG/√T for any β ∈ [0, 1]. Theorem 2: general online-to-batch conversion unifying schedules and averaging; Theorem 3: large steps γ = D/G optimal under a condition on ⟨g_t, z_t − z_1⟩. Experiments: 28 problems from logistic regression to LLM training; Schedule-Free AdamW won the MLCommons AlgoPerf 2024 self-tuning track.
- Notes: correct (worked example y₂ = 0.6, z₃ = 1.7, x₃ = 0.9 reproduced; they flag that Theorem 1's γ uses T). Survey (VII-D): consistent.
- Check-script experiment (20-D noisy least squares, batch 4, best step per method from the same grid): one Schedule-Free run (γ = 0.08, β = 0.9) vs linear decay tuned for each T: T = 250: 1.5e-2 vs 9.8e-3; T = 1000: 1.4e-3 vs 2.5e-3; T = 4000: 4.2e-4 vs 7.4e-4; constant SGD 2.9e-3 at 4000.
- Visuals: (three, 3D) z / y / x sequences on a noisy quadratic with β seg; (anytime) the comparison above.

## ahn2025general — Ahn, Magakyan, Cutkosky, ICML 2025 (arXiv 2411.07061) [survey 106]
- SF-SGD (x_t = (1 − c_t)x_{t−1} + c_tz_t, y_t = (1 − β_t)z_t + β_tx_t, z_{t+1} = z_t − γg_t). Definition 1: (λ, ε)-stationarity (approximate Goldstein). Rate O(λ^{1/2}ε^{−7/2}), optimal (Zhang & Cutkosky 2024). Proposition 1: L-smooth ⇒ O(ε⁻⁴), H-second-order-smooth ⇒ O(ε^{−3.5}) for ‖∇F‖ ≤ 2ε, both optimal. Algorithm 1 general conversion; Algorithm 5 (Option III) with discounted OMD = SF-SGD with β_t ~ U[β, 1], c_t ≡ 1 − β, base step γ/(1 − β) (Prop. 3, Sec. 5.3: explains β near 1 and larger steps).
- Notes: correct, including the worked example (y₂ = 0.625, z₃ = 1.6875) and the caveat that only SF-SGD (not Schedule-Free AdamW) is covered. Survey (VII-D: "optimal iteration complexity for non-smooth nonconvex, O(ε⁻⁴) smooth, O(ε^{−3.5}) second-order smooth; justifies weights close to one and large base steps"): consistent.
- The paper is theoretical; no performance claim versus tuned SGD is made, and our toy does not make one either (in a quick test, SGD with a small tuned step reached a lower final loss than SF-SGD with the theory-style constant EMA — not shown on the page).
- Visuals: (kink, 3D) f = |x| + |y| + 0.5 sin2x sin2y with noisy gradients, SF-SGD's z / y / x; readout: over the last 300 steps the true gradients at y have mean length ≈ 1.4 while their mean has length ≈ 0.04 (check script, last 1000 steps: 1.41 vs 0.019) — the (λ, ε) idea; (rates) the three complexity curves.

## crawshaw2026exploration — Crawshaw, Modi, Liu, Gower, ICML 2026 (arXiv 2510.09827) [survey 107]
- LMO(v) = argmin_{‖u‖≤1}⟨v, u⟩. Prop. 3.1 CSD: w_{t+1} = w_t + ηLMO(m_t); Prop. 3.2 RSD: w_{t+1} = w_t + η‖m_t‖_*LMO(m_t). Lemma 3.3: LMO and dual of product norms. Props. 3.4–3.5: Adam is CSD/RSD in suitable norms. Prop. 3.6: MuonAdam (spectral norm on hidden matrices, adaptive norm elsewhere, max-combined) is exactly CSD in that product norm. MuonMax: RSD in a hybrid product norm, matrix update ∝ (Σ_j‖M^j‖_nuc)·polar(M^i), stale dual norms (near-identical memory, ~5% wall-clock overhead). Momo (Props. 4.1–4.3): truncated model with a loss lower bound F_*, step min{η, (F̃_t − F_*)/‖m_t‖_*}·LMO. Experiments: GPT-2 Small (124M) on FineWeb 1B tokens, GPT-2 Large on SlimPajama 6B; Muon sensitive to the learning rate, MuonMax(-Momo) much more robust.
- Check script: −polar(M) is the spectral LMO (40 000 random candidates never beat it), ⟨M, polar(M)⟩ = nuclear norm, diagonal case reduces to sign descent; notes' diag(3, 1) example (−√10 vs −4) reproduced.
- Notes: correct (header "Ref 107"). Survey (VII-D): consistent ("non-Euclidean steepest descent under layer-wise norms combined into a product norm", MuonMax, Momo for arbitrary norms, GPT-2 experiments).
- Visuals: (steps, 3D) W = diag(a, b) on an ill-conditioned quadratic: GD, Muon-style CSD (sign steps of length 0.05, ends in a 2-cycle with loss ~1e-3 — which cycle is rounding-sensitive) and spectral RSD (5.7e-10 after 400 steps); (lmo) 2 × 2 gradient with σ₂ and rotation sliders: Frobenius decrease ‖G‖_F vs spectral decrease σ₁ + σ₂.

## zhang2020gradient — Zhang, He, Sra, Jadbabaie, ICLR 2020 (arXiv 1905.11881) [survey 41]
- Definition 1: (L0, L1)-smooth if ‖∇²f‖ ≤ L0 + L1‖∇f‖, motivated by AWD-LSTM on PTB (Fig. 1); Remark 1: positive correlation suffices. Lemma 2: polynomials of degree ≥ 3 are (L0, L1)-smooth but not L-smooth. Theorem 3: clipped GD (h = min{η_c, γη_c/‖∇f‖}) needs O(L0Δ/ε² + L1²Δ/L0) iterations. Assumption 4: M = sup‖∇f‖ on the initial sublevel set. Theorem 4: fixed-step GD needs Ω(L1MΔ/(ε² log M)); Theorem 6: O((ML1 + L0)Δ/ε²) suffices. Theorems 7–8: stochastic versions.
- Notes: correct; header "Ref. [33]" is old numbering. Survey (Assumption 4 and VIII): consistent ("curvature may grow with the gradient norm").
- Toy: f = x²/2 + x⁴/4 + y²/2 is (1, 3)-smooth on [−6, 6]² (ratio ≤ 0.84). Iterations to ‖∇f‖ ≤ 1e-4 from (x₀, 1), x₀ = 1…8 (M = 2.2…520): best fixed-step GD 6, 22, 47, 81, 189, 322; clipped GD (η = 1, γ = 0.5) 4, 6, 8, 10, 14, 18.
- Visuals: (steep, 3D) best fixed step vs clipped GD from x₀ ∈ {2, 4, 6}; (scaling) iterations vs M on log axes.

## polyak1964some — Polyak, USSR Comput. Math. Math. Phys. 4(5), 1964 [survey 49]
- The original article is not openly available (no arXiv/author copy); the page uses the standard heavy-ball formulas (as in the notes and survey Eq. 5): x_{t+1} = x_t − α∇f(x_t) + β(x_t − x_{t−1}); on quadratics with spectrum [μ, L], α = 4/(√L + √μ)², β = ((√L − √μ)/(√L + √μ))² gives contraction (√κ − 1)/(√κ + 1), vs (κ − 1)/(κ + 1) for GD with step 2/(L + μ).
- Check script: worst spectral radius equals (√κ − 1)/(√κ + 1) for κ = 10, 25, 100; κ = 25: GD 126 steps vs heavy ball 35 to f < 1e-8. Notes' saddle remark checked (p(1) = −αa < 0; the notes write "−a", dropping α — trivial).
- Notes header "Ref. [41]" is old numbering. Survey: consistent.
- Visuals: (ball, 3D) GD vs heavy ball with κ seg; (kappa) steps-to-1e-6 vs κ.
- Link: no free official PDF; `link` points to the journal DOI page only.

## duchi2011adaptive — Duchi, Hazan, Singer, JMLR 12, 2011 [survey 50]
- Eq. (1) diagonal AdaGrad x_{t+1} = Π(x_t − η diag(G_t)^{−1/2} g_t); full-matrix variant with G_t^{1/2}. Corollary 1: η = D∞/√2 ⇒ R(T) ≤ √2 D∞ Σᵢ‖g_{1:T,i}‖₂ (= √(2d) D∞ inf over diagonal preconditioners). Theorem 7: full-matrix O(‖x*‖ tr(G_T^{1/2})). Sec. 1.3 sparse example: p_i = min{1, ci^{−α}} ⇒ O(log d √T) for α ≥ 2 vs O(√(dT)) for online GD on the box.
- Check script: Σᵢ√pᵢ = 2.93/5.19/7.49/9.79 for d = 10…10⁴ (log growth); Jensen step by Monte Carlo; toy logistic regression with a 3%-frequency feature: excess loss SGD 0.0050 vs AdaGrad 0.0021, |w₂ − w₂*| 1.37 vs 0.53 (1500 steps, best step each).
- Notes: correct (convex online setting stressed). Header "Ref. [42]" old numbering. Survey: consistent.
- Visuals: (sparse, 3D) SGD vs AdaGrad on the empirical logistic loss with w₂ chart; (regret) Σ√pᵢ vs √d.

## kingma2015adam — Kingma & Ba, ICLR 2015 (arXiv 1412.6980) [survey 51]
- Algorithm 1 with bias correction; Sec. 2.1: |Δ_t| ≤ α(1 − β1)/√(1 − β2) in the most sparse case, ≈ ≤ α otherwise; scale invariance; Sec. 3: E[v_t] = E[g²](1 − β2ᵗ) + ζ; Sec. 6.4 / Fig. 4: without correction β2 near 1 is unstable (VAE). Theorem 4.1 / Corollary 4.2 (O(√T) regret) — their proof was later shown to be flawed (Reddi et al. 2018, survey [111]); the page says so. AdaMax (Sec. 7.1).
- Check script: corrected constant-gradient step = α exactly; uncorrected step = α(1 − β1ᵗ)/√(1 − β2ᵗ), peak 6.57α at t = 12 for (0.9, 0.999); single spike after 100/1000 zeros: 0.98α/2.52α (bound 3.16α); scale invariance; toy late loss SGD 0.018, momentum 0.010, Adam 0.008.
- Notes: correct. Survey (Sec. II): consistent.
- Visuals: (scales, 3D) SGD / momentum / Adam on a noisy, 400:1 quadratic; (bias) step/α curves with β2 seg.

## he2016deep — He, Zhang, Ren, Sun, CVPR 2016 (arXiv 1512.03385) [survey 108]
- Eq. (1) y = F(x, {W_i}) + x; Eq. (2) projection shortcut when shapes change. Degradation problem: deeper plain nets have higher training error (Fig. 1 CIFAR-10 20 vs 56 layers; Fig. 4 ImageNet 18 vs 34), not overfitting; with BN the forward/backward signals do not vanish, so the authors do not attribute it to vanishing gradients. Table 2 (top-1 %, 10-crop): plain-18 27.94, plain-34 28.54, ResNet-18 27.88, ResNet-34 25.03. Up to 152 layers on ImageNet (ILSVRC 2015 1st place), 1202 layers on CIFAR-10.
- Toy (ours): tied scalar chain of depth 2K from (0.02, 0.03): gradient norm plain vs residual 0.07 vs 1.4 (depth 2), 7.8e-8 vs 4.0 (depth 6), 7.3e-30 vs 8.2 (depth 20); at depth 6 and 20 the plain chain stays at loss 2.0 after 2000 steps while the residual one converges. Caveat recorded on the page: the paper itself says its degradation is not explained by vanishing gradients (BN); the toy only illustrates the identity-shortcut mechanism.
- Notes: correct. Survey (III-G): cites ResNets via Li et al. [62] (skip connections smooth landscapes) — consistent.
- Visuals: (chain, 3D) plain vs residual surfaces side by side with depth seg; (imagenet) reported Table 2 bars.
