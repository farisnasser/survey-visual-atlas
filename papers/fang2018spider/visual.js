// Fang, Li, Lin, Zhang 2018 (SPIDER) — path-integrated gradient estimator and SPIDER-SFO on a real finite sum.
// Components and the mean-square smoothness constant come from scripts/check-math/fang2018spider.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => .25 * (x * x - 1) ** 2 + .5 * y * y, g: (x, y) => [x ** 3 - x, y], hess: (x, y) => [3 * x * x - 1, 0, 1], h: v => Math.log1p(v * 3), dom: [-1.8, 1.8, -1.6, 1.6] });
const ETA = .02, Q = 25, S2 = 4;
const gi = (D, p, i) => [p[0] ** 3 - p[0] + D.a[i] + D.c[i] * p[0], p[1] + D.b[i] - D.c[i] * p[1]];
const mb = (D, p, I) => { let gx = 0, gy = 0; for (const i of I) { const g = gi(D, p, i); gx += g[0]; gy += g[1]; } return [gx / I.length, gy / I.length]; };
const nstep = (p, v) => { const n = Math.max(Math.hypot(...v), 1e-12); return [p[0] - ETA * v[0] / n, p[1] - ETA * v[1] / n]; };

/** SPIDER-SFO Option I: full gradient every Q steps, otherwise same-minibatch difference + previous estimate. */
const spider = (D, R) => (p, b) => {
  b.k = (b.k ?? -1) + 1;
  if (b.k % Q === 0) b.v = F.g(...p);
  else { const I = Array.from({ length: S2 }, () => Math.floor(R() * D.a.length)), a = mb(D, p, I), c = mb(D, b.xp, I); b.v = [a[0] - c[0] + b.v[0], a[1] - c[1] + b.v[1]]; }
  b.err.push([b.k, Math.max((b.v[0] - F.g(...p)[0]) ** 2 + (b.v[1] - F.g(...p)[1]) ** 2, 1e-12)]);
  b.xp = p.slice(); return nstep(p, b.v);
};
const sgd = (D, R) => (p, b) => {
  b.k = (b.k ?? -1) + 1; const I = Array.from({ length: S2 }, () => Math.floor(R() * D.a.length)), v = mb(D, p, I);
  b.err.push([b.k, Math.max((v[0] - F.g(...p)[0]) ** 2 + (v[1] - F.g(...p)[1]) ** 2, 1e-12)]);
  return nstep(p, v);
};

export default {
  tabs: [
    {
      id: 'estimator', name: 'Estimator', title: 'Track the gradient by its changes, not by fresh noisy samples',
      caption: 'Both balls take normalized steps of length η on the mean of 200 non-convex components. Coral uses SPIDER: a full gradient every q = 25 steps, otherwise the same minibatch evaluated at x_k and x_{k−1}. Violet uses a fresh minibatch each step. Right: estimator error and Lemma 1’s bound, which grows linearly within each epoch.',
      note: 'f_i = (x² − 1)²/4 + y²/2 + a_i x + b_i y + c_i(x² − y²)/2 with zero-mean a, b, c; |S₂| = 4, η = 0.02 = ε₁ of Lemma 1; L = mean-square smoothness bound on the box.',
      legend: [[HEX.ball, 'SPIDER-SFO'], [HEX.opt2, 'minibatch SGD (normalized)'], [HEX.good, 'minima (±1, 0)']],
      action: 'Run again', cam: [0, 4.2, 4.4], tgt: [0, .3, 0],
      charts: [{ id: 'err', title: '‖v_k − ∇f(x_k)‖²', xlabel: 'step k', ylabel: 'squared error', logy: true, ylim: [1e-7, 30], h: 250 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.marker(F, 1, 0, C.good); s.marker(F, -1, 0, C.good); s.eigenLines(F, 0, 0, { len: .5 });
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, R = rng(Date.now() % 1e6); s.clearBalls();
        ctx.sp = s.ball(F, { start: [-1.6, 1.4], step: spider(D, R), speed: 1, maxSteps: 600 }); ctx.sp.err = [];
        ctx.sg = s.ball(F, { start: [-1.6, 1.4], step: sgd(D, R), color: C.opt2, speed: 1, maxSteps: 600 }); ctx.sg.err = [];
      },
      frame(ctx) {
        const a = ctx.sp, b = ctx.sg; if (!a) return;
        const L = ctx.data.L, xm = Math.max(50, a.steps), bound = [];
        for (let k = 0; k <= xm; k++) bound.push([k, Math.max((k % Q) * L * L * ETA * ETA / S2, 1e-7)]);
        ctx.charts.err.clear().set({ xlim: [0, xm] }).line(bound, { color: HEX.muted, dash: [4, 4], width: 1, label: 'Lemma 1 bound' })
          .line(b.err, { color: HEX.opt2, width: 1.2, label: 'SGD' }).line(a.err, { color: HEX.ball, width: 1.8, label: 'SPIDER' }).draw();
        const avg = e => e.length ? e.reduce((s, r) => s + r[1], 0) / e.length : 0;
        ctx.stat(`step ${a.steps} · mean squared error SPIDER ${avg(a.err).toExponential(1)} vs SGD ${avg(b.err).toExponential(1)} · |∇f|: SPIDER ${Math.hypot(...F.g(...a.p)).toFixed(3)}, SGD ${Math.hypot(...F.g(...b.p)).toFixed(3)}`);
      },
    },
    {
      id: 'rates', name: 'Oracle counts', mode: '2d', title: 'How many stochastic gradients to reach ‖∇f‖ ≤ ε',
      caption: 'Orders of the stochastic-gradient cost for finite sums of n components, constants set to 1. SPIDER matches its own lower bound n^{1/2}ε^{−2} (Theorem 3), so it is near-optimal in that regime; online it needs ε^{−3} versus SGD’s ε^{−4}.',
      note: 'Orders only (no constants, no logs). SVRG line: non-convex SVRG, O(n + n^{2/3}ε^{−2}). The lower bound holds for n ≤ O(ε^{−4}).',
      hint: '', legend: [[HEX.opt3, 'SGD ε⁻⁴'], [HEX.opt2, 'SVRG n + n^{2/3}ε⁻²'], [HEX.ball, 'SPIDER min(n + n^{1/2}ε⁻², ε⁻³)'], [HEX.muted, 'lower bound n^{1/2}ε⁻²']],
      controls: [{ id: 'n', label: 'components n', min: 2, max: 9, step: .5, value: 5, fmt: v => `10^${(+v).toFixed(1)}` }],
      plot: { xlabel: 'accuracy ε', ylabel: 'stochastic gradient calls', logx: true, logy: true, xlim: [1e-4, .5], ylim: [1, 1e18], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const n = 10 ** ctx.vals.n, P = ctx.plot.clear();
        P.fn(e => e ** -4, { color: HEX.opt3, width: 2 }).fn(e => n + n ** (2 / 3) * e ** -2, { color: HEX.opt2, width: 2 })
          .fn(e => Math.min(n + Math.sqrt(n) * e ** -2, e ** -3), { color: HEX.ball, width: 2.6 }).fn(e => Math.sqrt(n) * e ** -2, { color: HEX.muted, dash: [5, 4], width: 1.4 }).draw();
        const e = 1e-3, v = [e ** -4, n + n ** (2 / 3) * e ** -2, Math.min(n + Math.sqrt(n) * e ** -2, e ** -3)];
        ctx.stat(`n = 10^${ctx.vals.n.toFixed(1)}, ε = 10⁻³: SGD ${v[0].toExponential(1)} · SVRG ${v[1].toExponential(1)} · SPIDER ${v[2].toExponential(1)}`);
      },
    },
  ],
};
