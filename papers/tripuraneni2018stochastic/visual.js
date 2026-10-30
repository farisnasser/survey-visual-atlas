// Tripuraneni, Stern, Jin, Regier, Jordan 2018 — stochastic cubic regularization (Alg. 1) with the
// Hessian-vector-product Cubic-Subsolver (Alg. 3). Checked in scripts/check-math/tripuraneni2018stochastic.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng, eig2 } from '../../src/lib/math.js';

const RHO = 1, ELL = 1, SG = .5, SH = .5, EPS = .05;
const f = (x, y) => .5 * x * x + Math.cos(y), gf = (x, y) => [x, -Math.sin(y)], Hf = (x, y) => [1, 0, -Math.cos(y)];
const FS = def({ L: f, g: gf, hess: Hf, h: v => v, dom: [-1.5, 1.5, -4, 4], square: true });
const model = (g, B) => ({ m: (u, v) => g[0] * u + g[1] * v + .5 * (B[0] * u * u + 2 * B[1] * u * v + B[2] * v * v) + RHO / 6 * Math.hypot(u, v) ** 3,
  dm: (u, v) => { const r = Math.hypot(u, v); return [g[0] + B[0] * u + B[1] * v + RHO / 2 * r * u, g[1] + B[1] * u + B[2] * v + RHO / 2 * r * v]; } });
function sample(p, S1, S2, R) {
  const g = gf(...p), H = Hf(...p), s1 = SG / Math.sqrt(S1), s2 = SH / Math.sqrt(S2), z = R.normal() * s2;
  return [[g[0] + s1 * R.normal(), g[1] + s1 * R.normal()], [H[0] + s2 * R.normal(), H[1] + z, H[2] + s2 * R.normal()]];
}
/** Exact global minimizer of a 2D cubic model (secular equation). */
function exactMin(g, B) {
  const ev = eig2(...B), [[l1, v1], [l2, v2]] = ev[0][0] <= ev[1][0] ? ev : [ev[1], ev[0]];
  const gb = [g[0] * v1[0] + g[1] * v1[1], g[0] * v2[0] + g[1] * v2[1]], lo0 = Math.max(0, -2 * l1 / RHO) + 1e-12;
  const nr = r => Math.hypot(gb[0] / (l1 + RHO * r / 2), gb[1] / (l2 + RHO * r / 2));
  let d;
  if (nr(lo0) < lo0) { const a2 = -gb[1] / (l2 + RHO * lo0 / 2); d = [Math.sqrt(Math.max(lo0 * lo0 - a2 * a2, 0)), a2]; }
  else { let lo = lo0, hi = lo0 + 1; while (nr(hi) > hi) hi *= 2; for (let i = 0; i < 80; i++) { const mid = (lo + hi) / 2; if (nr(mid) > mid) lo = mid; else hi = mid; } d = [-gb[0] / (l1 + RHO * hi / 2), -gb[1] / (l2 + RHO * hi / 2)]; }
  return [d[0] * v1[0] + d[1] * v2[0], d[0] * v1[1] + d[1] * v2[1]];
}
/** Alg. 3: Cauchy step if |g| >= ell^2/rho, else GD on the model with g~ = g + sigma*zeta, step 1/(20 ell). */
function subsolve(g, B, R, iters = 400) {
  const gn = Math.hypot(...g);
  if (gn >= ELL * ELL / RHO) { const k = (g[0] * (B[0] * g[0] + B[1] * g[1]) + g[1] * (B[1] * g[0] + B[2] * g[1])) / (RHO * gn * gn), rc = -k + Math.sqrt(k * k + 2 * gn / RHO); return [-rc * g[0] / gn, -rc * g[1] / gn]; }
  const gt = [g[0] + 1e-3 * R.normal(), g[1] + 1e-3 * R.normal()], M = model(gt, B); let d = [0, 0];
  for (let i = 0; i < iters; i++) { const q = M.dm(...d); d = [d[0] - q[0] / (20 * (ELL + 2 * RHO)), d[1] - q[1] / (20 * (ELL + 2 * RHO))]; }
  return d;
}

export default {
  tabs: [
    {
      id: 'model', name: 'Sampled model', title: 'Each step minimizes a cubic model built from a few samples',
      caption: 'At x = (0.6, 0.15) on f = x²/2 + cos y, Algorithm 1 averages |S₁| stochastic gradients and |S₂| stochastic Hessians into g and B, and minimizes m(Δ) = gᵀΔ + ½ΔᵀBΔ + (ρ/6)‖Δ‖³ (surface). The ball is the Cubic-Subsolver: gradient descent on m with a tiny perturbation of g, so it can leave the flat start along negative curvature. Green: minimizer of the noiseless model.',
      note: 'Oracle noise σ = 0.5 per sample, ρ = ℓ = 1. Here ‖g‖ < ℓ²/ρ, so Algorithm 3 takes its gradient-descent branch (step 1/(20ℓ)). Larger samples make the sampled model approach the noiseless one.',
      legend: [[HEX.ball, 'Cubic-Subsolver on m'], [HEX.good, 'minimizer of the noiseless model'], [HEX.white, 'minimizer of the sampled model']],
      action: 'Resample', cam: [3.2, 3.2, 4], tgt: [0, .3, 0],
      controls: [{ id: 'S', type: 'seg', label: 'samples |S₁| = |S₂|', value: 4, options: [[1, '1'], [4, '4'], [25, '25'], [400, '400']] }],
      build(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6), x = [.6, .15]; s.clear();
        const [g, B] = sample(x, ctx.vals.S, ctx.vals.S, R), M = model(g, B), T = model(gf(...x), Hf(...x));
        const d = def({ L: M.m, g: M.dm, h: v => v, dom: [-3, 3, -3, 3] }); s.surface(d);
        const ds = exactMin(g, B), dt = exactMin(gf(...x), Hf(...x));
        s.marker(d, ...dt, C.good, .09); s.marker(d, ...ds, C.white, .07);
        const gt = [g[0] + 1e-3 * R.normal(), g[1] + 1e-3 * R.normal()], Mt = model(gt, B), lr = 1 / (20 * (ELL + 2 * RHO));
        ctx.b = s.ball(d, { start: [0, 0], speed: 3, maxSteps: 3000, step: p => { const q = Mt.dm(...p); return [p[0] - lr * q[0], p[1] - lr * q[1]]; } });
        ctx.info = { M, T, g, x, ds, dt };
      },
      onControl(ctx) { this.drop(ctx); },
      frame(ctx) {
        const b = ctx.b, I = ctx.info; if (!b) return;
        ctx.stat(`sampled g = (${I.g.map(v => v.toFixed(2)).join(', ')}) · subsolver Δ = (${b.p.map(v => v.toFixed(2)).join(', ')}), m(Δ) = ${I.M.m(...b.p).toFixed(3)} · true change f(x + Δ) − f(x) = ${(f(I.x[0] + b.p[0], I.x[1] + b.p[1]) - f(...I.x)).toFixed(3)}`);
      },
    },
    {
      id: 'outer', name: 'Outer loop', title: 'Stochastic cubic steps escape the saddle line and stop at a minimum',
      caption: 'f = x²/2 + cos y has a line of saddles at y = 0 and minima at y = ±π. Coral: Algorithm 1 from (1.2, 0) with |S₁| = |S₂| = 20 noisy samples per step, stopping once the model predicts a decrease smaller than (1/100)√(ε³/ρ). Violet: SGD with step 0.1 and one noisy gradient per step.',
      note: 'σ = 0.5, ρ = ℓ = 1, ε = 0.05. Cubic-Subsolver as in Alg. 3 (300 model-GD steps). The paper’s rate is Õ(ε^{−3.5}) gradient + HVP evaluations vs Õ(ε^{−4}) for perturbed SGD.',
      legend: [[HEX.ball, 'stochastic cubic regularization'], [HEX.opt2, 'SGD'], [HEX.good, 'minima y = ±π']],
      action: 'Run again', cam: [3.6, 3.0, 3.4], tgt: [0, .4, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(FS); s.eigenLines(FS, 1.2, 0, { len: .5 });
        s.marker(FS, 0, Math.PI, C.good); s.marker(FS, 0, -Math.PI, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6); s.clearBalls();
        ctx.a = s.ball(FS, { start: [1.2, 0], speed: .15, maxSteps: 400, stop: b => b.fin, step: (p, b) => {
          b.calls = (b.calls || 0) + 40; const [g, B] = sample(p, 20, 20, R), M = model(g, B), d = subsolve(g, B, R, 300);
          if (M.m(...d) >= -Math.sqrt(EPS ** 3 / RHO) / 100) { b.fin = true; return p; }
          return [p[0] + d[0], p[1] + d[1]];
        } });
        ctx.b = s.ball(FS, { start: [1.2, 0], color: C.opt2, speed: 4, maxSteps: 6000, step: opt.sgd(.1, SG, R) });
      },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        ctx.stat(`cubic: ${a.steps} outer steps (${a.calls || 0} oracle calls)${a.fin ? ' · stopped by the model-decrease test' : ''}, f = ${a.loss().toFixed(3)} · SGD: ${b.steps} steps, f = ${b.loss().toFixed(3)} (min −1)`);
      },
    },
    {
      id: 'oned', name: '1D check', mode: '2d', title: 'The notes’ one-dimensional step, computed',
      caption: 'f(s) = cos s at s = 0 (a maximum: f′ = 0, f″ = −1). With sampled g = 0.01 and B = −0.9, the cubic model is minimized at Δ ≈ −1.811. The noiseless model predicts −0.650 there and the true function drops by −1.238.',
      note: 'ρ = 1. Curves: sampled model m(Δ), noiseless model q(Δ) = −Δ²/2 + |Δ|³/6, and the true change cos Δ − 1.',
      hint: '', legend: [[HEX.ball, 'sampled model m'], [HEX.pos, 'noiseless model q'], [HEX.good, 'true change cos Δ − 1']],
      plot: { xlabel: 'step Δ', ylabel: 'predicted / actual change', xlim: [-3.2, 3.2], ylim: [-2.2, 1] },
      build(ctx) {
        const m = D => .01 * D - .45 * D * D + Math.abs(D) ** 3 / 6, q = D => -.5 * D * D + Math.abs(D) ** 3 / 6, D0 = (-1.8 - Math.sqrt(3.24 + .08)) / 2;   // root of 0.01 - 0.9Δ - Δ²/2 = 0 (Δ < 0)
        ctx.plot.clear().fn(m, { color: HEX.ball, width: 2.2, label: 'm' }).fn(q, { color: HEX.pos, width: 2, label: 'q' }).fn(D => Math.cos(D) - 1, { color: HEX.good, width: 2, label: 'cos Δ − 1' })
          .vline(D0, { color: HEX.muted, dash: [4, 4], text: 'Δ*' }).scatter([[D0, m(D0)], [D0, q(D0)], [D0, Math.cos(D0) - 1]], { color: HEX.white, r: 4 }).draw();
        ctx.stat(`Δ* = ${D0.toFixed(6)} · m(Δ*) = ${m(D0).toFixed(6)} · q(Δ*) = ${q(D0).toFixed(6)} · cos Δ* − 1 = ${(Math.cos(D0) - 1).toFixed(6)}`);
      },
    },
  ],
};
