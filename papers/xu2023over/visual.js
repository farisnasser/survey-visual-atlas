// Xu & Du 2023 — learning one ReLU neuron: exact parameterization converges exponentially, over-parameterization only like T^−3.
// Exact Gaussian loss via k(u, v) = (‖u‖‖v‖ sin φ + (π − φ) u·v)/(2π). Curves from scripts/check-math/xu2023over.py (data.json).
import { def, opt, C, HEX, statLabels } from '../../src/lib/atlas3d.js';

const k = (u0, u1, v0, v1) => { const nu = Math.hypot(u0, u1), nv = Math.hypot(v0, v1); if (nu < 1e-12 || nv < 1e-12) return 0; const c = Math.max(-1, Math.min(1, (u0 * v0 + u1 * v1) / (nu * nv))), p = Math.acos(c); return (nu * nv * Math.sin(p) + (Math.PI - p) * (u0 * v0 + u1 * v1)) / (2 * Math.PI); };
// teacher v = e₁ in d = 2
const L1 = (a, b) => .5 * (k(a, b, a, b) - 2 * k(a, b, 1, 0) + k(1, 0, 1, 0));                                   // n = 1: w = (a, b)
const L2 = (a, b) => { const p = [a, b], q = [1 - a, -b]; return .5 * (k(...p, ...p) + 2 * k(...p, ...q) + k(...q, ...q) - 2 * k(...p, 1, 0) - 2 * k(...q, 1, 0) + k(1, 0, 1, 0)); };   // n = 2: w₁ = (a, b), w₂ = (1 − a, −b)
const H = v => Math.cbrt(v) * .9;
const A = def({ L: L1, h: H, dom: [-.4, 1.4, -.6, .6], ox: -1.85, size: 3, fit: 0, square: true });
const B = def({ L: L2, h: H, dom: [-.4, 1.4, -.6, .6], ox: 1.85, size: 3, fit: 0, square: true });
const START = [.3, .4], ETA = .5;
const NCOL = { 1: HEX.good, 2: HEX.ball, 3: HEX.opt2, 5: HEX.opt3 };

export default {
  tabs: [
    {
      id: 'split', name: 'Why it slows down', title: 'One student sits in a bowl; two students sit in a flat-bottomed trough that is only cubic',
      caption: 'Left: one student neuron w = (a, b) learning the teacher v = e₁ — a curved bowl with a single minimum, so gradient descent converges exponentially. Right: two students w₁ = (a, b), w₂ = (1 − a, −b) that always add up to v; every point of the segment b = 0, 0 ≤ a ≤ 1 is a global minimum (green), and across it the loss grows only like |b|³, because two ReLUs at a small angle almost coincide. GD therefore creeps in: b ~ 1/t and loss ~ t⁻³.',
      note: 'Exact population loss for x ~ N(0, I₂), heights cube-root scaled. The right surface is a 2-parameter slice of the 4-parameter problem (students constrained to sum to v); both balls run GD with step 0.5 from (0.3, 0.4).',
      legend: [[HEX.good, 'global minima'], [HEX.ball, 'gradient descent']],
      cam: [0, 4.4, 4.4], tgt: [0, .2, 0], action: 'Run again',
      charts: [{ id: 'L', title: 'loss along the two runs', xlabel: 'GD step', ylabel: 'loss', logx: true, logy: true, xlim: [1, 5000], ylim: [1e-14, 1], h: 230 }],
      build(ctx) {
        const s = ctx.stage; s.surface(A); s.surface(B);
        s.marker(A, 1, 0, C.good, .1); s.tube(B, Array.from({ length: 41 }, (_, i) => [i / 40, 0]), C.good, .02);
        ctx.lab = statLabels(s, [['n = 1 (exact)', A.W(.5, .6, .5)], ['n = 2 (over-parameterized)', B.W(.5, .6, .5)]]);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.b = [s.ball(A, { start: START, step: opt.gd(ETA), speed: 2, maxSteps: 5000 }), s.ball(B, { start: START, step: opt.gd(ETA), speed: 2, maxSteps: 5000 })];
        for (const q of ctx.b) { q.h = [[1, q.loss()]]; q.onStep = z => { if (z.steps < 50 || z.steps % 10 === 0) z.h.push([z.steps, Math.max(z.loss(), 1e-14)]); }; }
      },
      frame(ctx) {
        const [p, q] = ctx.b || []; if (!p) return;
        ctx.charts.L.clear().fn(t => .3 * t ** -3, { color: HEX.muted, dash: [4, 4], width: 1, label: '∝ t⁻³' }).line(p.h, { color: HEX.good, width: 2, label: 'n = 1' }).line(q.h, { color: HEX.ball, width: 2, label: 'n = 2' }).draw();
        ctx.lab[0].stat(`loss ${p.loss().toExponential(1)}`); ctx.lab[1].stat(`loss ${q.loss().toExponential(1)}`);
        ctx.stat(`step ${q.steps} · n = 1: loss ${p.loss().toExponential(2)} · n = 2: loss ${q.loss().toExponential(2)}, split |b| = ${Math.abs(q.p[1]).toExponential(2)}`);
      },
    },
    {
      id: 'rates', name: 'T⁻³ vs exp(−T)', mode: '2d', title: 'Gradient descent on the full problem: exponential for n = 1, T⁻³ for every n ≥ 2',
      caption: 'Replication of the paper’s Fig. 1: d = 10, teacher ‖v‖ = 1, students initialized N(0, σ²I) with σ = 0.1, GD with η = 0.05 using the exact gradient (Eq. 3). The exactly-parameterized student (n = 1) reaches machine precision in a few hundred steps; with 2, 3 or 5 students the loss follows a straight line of slope −3 on log–log axes (Theorem 1: O(T⁻³) for GD; Theorem 2: Ω(T⁻³) for gradient flow).',
      note: 'Fitted slopes over T ∈ [15 000, 60 000]: −2.99 (n = 2), −2.99 (n = 3), −2.98 (n = 5). Right: log L / log(ηT), which the paper shows creeping toward −3.',
      hint: '', legend: [[HEX.good, 'n = 1'], [HEX.ball, 'n = 2'], [HEX.opt2, 'n = 3'], [HEX.opt3, 'n = 5'], [HEX.muted, 'slope −3']],
      plot: { xlabel: 'iteration T', ylabel: 'population loss', logx: true, logy: true, xlim: [1, 6e4], ylim: [1e-16, 1], legend: false },
      charts: [{ id: 'r', title: 'log L(T) / log(ηT)', xlabel: 'iteration T', ylabel: 'ratio', logx: true, xlim: [1e3, 6e4], ylim: [-4, -1.5], h: 230 }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear().fn(t => 3e1 * t ** -3, { color: HEX.muted, dash: [5, 4], width: 1.2 });
        for (const n of [1, 2, 3, 5]) P.line(D[`c${n}`].filter(r => r[0] > 0).map(r => [r[0], Math.max(r[1], 1e-16)]), { color: NCOL[n], width: n === 1 ? 2 : 2.2 });
        P.draw();
        const ch = ctx.charts.r.clear().hline(-3, { color: HEX.muted, dash: [4, 4], text: '−3' });
        for (const n of [2, 3, 5]) ch.line(D[`c${n}`].filter(r => r[0] >= 1e3).map(r => [r[0], Math.log(r[1]) / Math.log(D.eta * r[0])]), { color: NCOL[n], width: 1.8 });
        ch.draw();
        const last = n => D[`c${n}`].at(-1);
        ctx.stat(`n = 1: loss < 10⁻²⁸ by T = ${last(1)[0]} · at T = 60 000: n = 2: ${last(2)[1].toExponential(1)}, n = 3: ${last(3)[1].toExponential(1)}, n = 5: ${last(5)[1].toExponential(1)}`);
      },
    },
  ],
};
