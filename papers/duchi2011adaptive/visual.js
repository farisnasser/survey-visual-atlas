// Duchi, Hazan, Singer 2011 — AdaGrad: per-coordinate steps η/√(Σ g²), so rarely active features keep large steps.
// Toy data, tuned steps and the sparse-regret scaling are in scripts/check-math/duchi2011adaptive.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

function makeLoss(D) {
  const X = D.X, Y = D.Y, n = Y.length;
  const L = (a, b) => { let s = 0; for (let i = 0; i < n; i++) { const m = Y[i] * (X[i][0] * a + X[i][1] * b); s += m > 0 ? Math.log1p(Math.exp(-m)) : -m + Math.log1p(Math.exp(m)); } return s / n; };
  const g1 = (w, i) => { const c = -Y[i] / (1 + Math.exp(Y[i] * (X[i][0] * w[0] + X[i][1] * w[1]))); return [c * X[i][0], c * X[i][1]]; };
  return { L, g1, n };
}

export default {
  tabs: [
    {
      id: 'sparse', name: 'Rare features', title: 'A rare feature gets big steps only if its step size is its own',
      caption: 'Logistic regression with two features: x₁ appears in every example, x₂ in only 3% of them. Plain SGD (violet) uses one decaying step size, so the rarely updated weight w₂ barely moves before the step has shrunk. AdaGrad (coral) divides each coordinate by the root of its own accumulated squared gradients: w₂, which has seen few gradients, keeps a large step and heads to its optimum.',
      note: 'The surface is the empirical logistic loss of 4000 examples; both balls take one random example per step. SGD: η/√t with η = 3; AdaGrad: η = 1 (each the best of a small grid in the check script). Start (−2.5, −2.5).',
      legend: [[HEX.opt2, 'SGD, η/√t'], [HEX.ball, 'diagonal AdaGrad'], [HEX.good, 'minimizer']],
      action: 'Restart', cam: [0, 4.4, 4.4], tgt: [0, .1, 0],
      charts: [{ id: 'w2', title: 'rare-feature weight w₂', xlabel: 'step', ylabel: 'w₂', ylim: [-3, 4], xlim: [0, 1500], h: 220, legend: false }],
      build(ctx) {
        ctx.M = makeLoss(ctx.data); const d = ctx.d = def({ L: ctx.M.L, dom: [-3, 3, -3, 4], h: v => Math.log1p(v), N: 60 });
        ctx.stage.surface(d); ctx.stage.marker(d, ...ctx.data.wbest, C.good, .1); this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, d = ctx.d, M = ctx.M, R1 = rng(1), R2 = rng(2), eS = ctx.data.etaS, eA = ctx.data.etaA; s.clearBalls(); ctx.H = { s: [], a: [] };
        ctx.bs = s.ball(d, { start: [-2.5, -2.5], speed: 4, maxSteps: 1500, color: C.opt2, step: (p, b) => { const g = M.g1(p, Math.floor(R1() * M.n)), h = eS / Math.sqrt(b.steps + 1); return [p[0] - h * g[0], p[1] - h * g[1]]; } });
        ctx.ba = s.ball(d, { start: [-2.5, -2.5], speed: 4, maxSteps: 1500, step: (p, b) => { const g = M.g1(p, Math.floor(R2() * M.n)); b.G = b.G || [1e-8, 1e-8]; b.G = [b.G[0] + g[0] * g[0], b.G[1] + g[1] * g[1]]; return [p[0] - eA * g[0] / Math.sqrt(b.G[0]), p[1] - eA * g[1] / Math.sqrt(b.G[1])]; } });
        ctx.bs.onStep = b => ctx.H.s.push([b.steps, b.p[1]]); ctx.ba.onStep = b => ctx.H.a.push([b.steps, b.p[1]]);
      },
      frame(ctx) {
        const w2 = ctx.data.wbest[1];
        ctx.charts.w2.clear().hline(w2, { color: HEX.good, dash: [4, 4], text: 'w₂*' }).line(ctx.H.s, { color: HEX.opt2, width: 1.4 }).line(ctx.H.a, { color: HEX.ball, width: 1.6 }).draw();
        const f = b => (ctx.M.L(...b.p) - ctx.data.Lopt).toFixed(4);
        ctx.stat(`step ${ctx.ba.steps} · w₂: SGD ${ctx.bs.p[1].toFixed(2)}, AdaGrad ${ctx.ba.p[1].toFixed(2)} (optimum ${w2.toFixed(2)}) · excess loss: SGD ${f(ctx.bs)}, AdaGrad ${f(ctx.ba)}`);
      },
    },
    {
      id: 'regret', name: 'Sparse regret', mode: '2d', title: 'Regret that scales with log d instead of √d',
      caption: 'The paper’s sparse example: feature i is present with probability pᵢ = min(1, i⁻²). AdaGrad’s regret bound (Corollary 1) is √2·D∞·Σᵢ‖g₁:T,ᵢ‖₂, whose expectation is at most √T·Σᵢ√pᵢ, a harmonic sum that grows like log d. Online gradient descent’s bound grows like √(dT) on the same box. Curves: both factors divided by √T.',
      note: 'Exact sums (α = 2); the check script confirms the Jensen step by Monte Carlo (d = 200, T = 2000: 231.5 ≤ 262.9, vs √(dT) = 632).',
      hint: '', legend: [[HEX.ball, 'AdaGrad: Σᵢ √pᵢ'], [HEX.opt2, 'online GD: √d']],
      plot: { xlabel: 'dimension d (log)', ylabel: 'bound / (D√T) (log)', logx: true, logy: true, xlim: [1, 1e5], ylim: [.8, 400], legend: false },
      build(ctx) {
        const marks = new Set(Array.from({ length: 41 }, (_, k) => Math.round(10 ** (k / 8)))), pts = []; let s = 0;
        for (let i = 1; i <= 1e5; i++) { s += Math.min(1, i ** -2) ** .5; if (marks.has(i)) pts.push([i, s]); }
        ctx.plot.clear().line(pts, { color: HEX.ball, width: 2.2 }).fn(dd => Math.sqrt(dd), { color: HEX.opt2, width: 2.2 }).draw();
        ctx.stat(`d = 10⁴: Σᵢ√pᵢ = ${pts.find(p => p[0] === 1e4)[1].toFixed(2)} vs √d = 100`);
      },
    },
  ],
};
