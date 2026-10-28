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
