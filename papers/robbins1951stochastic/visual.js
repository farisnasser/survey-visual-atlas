// Robbins & Monro 1951 — stochastic approximation with gains a_n: x_{n+1} = x_n + a_n(α − y_n).
// Mean-square curves and sample paths from scripts/check-math/robbins1951stochastic.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const lc = t => { const a = Math.abs(t); return a + Math.log1p(Math.exp(-2 * a)) - Math.LN2; };   // log cosh, stable
const F = def({ L: (x, y) => lc(x) - x / 2 + lc(y) + .3 * y, g: (x, y) => [Math.tanh(x) - .5, Math.tanh(y) + .3], h: v => v, dom: [-2.4, 2.4, -2.4, 2.4] });
/** Noisy gradient (bounded U(−1, 1) noise per coordinate) with gain schedule a(n). */
const rm = (a, R) => (p, b) => { const n = (b.n = (b.n || 0) + 1), g = F.g(...p); return [p[0] - a(n) * (g[0] + R.uniform(-1, 1)), p[1] - a(n) * (g[1] + R.uniform(-1, 1))]; };
const SC = [['2/n', n => 2 / n, HEX.ball, C.ball], ['0.2', () => .2, HEX.opt2, C.opt2], ['2/n²', n => 2 / (n * n), HEX.opt3, C.opt3]];

export default {
  tabs: [
    {
      id: 'gains', name: 'Gain sequences', title: 'Steps must shrink, but not too fast: Σaₙ = ∞ and Σaₙ² < ∞',
      caption: 'All three balls follow noisy gradients of a smooth convex function (bounded noise) — the multivariate reading of the Robbins–Monro rule. Coral uses aₙ = 2/n (“type 1/n”, Eq. 30): it settles on the root. Violet uses a constant gain 0.2 (Σaₙ² = ∞): it keeps jittering in a noise cloud. White uses aₙ = 2/n² (Σaₙ < ∞): the steps die out before it arrives.',
      note: 'F(x, y) = log cosh x − x/2 + log cosh y + 0.3y; root (0.549, −0.310) (green). Noise U(−1, 1) per coordinate; the paper’s own setting is one-dimensional (second tab).',
      legend: [[HEX.ball, 'aₙ = 2/n'], [HEX.opt2, 'aₙ = 0.2'], [HEX.opt3, 'aₙ = 2/n²'], [HEX.good, 'root']],
      cam: [3.4, 3.6, 3.6], tgt: [0, .2, 0], action: 'Run again',
      charts: [{ id: 'e', title: 'squared distance to the root', xlabel: 'n', ylabel: '‖xₙ − θ‖²', logx: true, logy: true, xlim: [1, 3000], ylim: [1e-6, 20], h: 220 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.marker(F, ...ctx.data.root, C.good); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, seed = Date.now() % 1e6, rt = ctx.data.root; s.clearBalls();
        ctx.b = SC.map(([, a, , col], k) => s.ball(F, { start: [-2, 2], step: rm(a, rng(seed + k)), color: col, speed: 3, maxSteps: 3000 }));
        for (const q of ctx.b) { q.h = []; q.onStep = z => { if (z.steps < 30 || z.steps % 5 === 0) z.h.push([z.steps, Math.max((z.p[0] - rt[0]) ** 2 + (z.p[1] - rt[1]) ** 2, 1e-6)]); }; }
      },
      frame(ctx) {
        const B = ctx.b; if (!B) return; const rt = ctx.data.root;
        const ch = ctx.charts.e.clear(); B.forEach((q, k) => ch.line(q.h, { color: SC[k][2], width: 1.6, label: SC[k][0] })); ch.draw();
        const d = q => Math.hypot(q.p[0] - rt[0], q.p[1] - rt[1]).toFixed(3);
        ctx.stat(`n = ${B[0].steps} · distance to the root: 2/n ${d(B[0])} · 0.2 ${d(B[1])} · 2/n² ${d(B[2])}`);
      },
    },
    {
      id: 'original', name: 'The 1951 setting', mode: '2d', title: 'Find the level θ where an unknown response curve equals α, from noisy responses only',
      caption: 'The paper’s problem: at level x an experiment returns a random Y(x) with unknown mean M(x); find θ with M(θ) = α. Here M(x) = tanh x, α = ½, θ = 0.549, and Y = M(x) + U(−1, 1). Three sample paths of xₙ₊₁ = xₙ + aₙ(α − yₙ) with aₙ = 2/n home in on θ. Right: b_n = E(xₙ − θ)² over 200 runs for the three gain sequences — only the type-1/n gains send it to 0 (Theorems 1–2), here like 1/n.',
      note: 'Assumptions used: bounded responses (4), M below α left of θ and above it right of θ (5), nondecreasing M with M′(θ) > 0 (Theorem 2), Σaₙ² < ∞ (6) and condition (26), which forces Σaₙ = ∞.',
      hint: '', legend: [[HEX.ball, 'sample paths, aₙ = 2/n'], [HEX.good, 'θ']],
      plot: { xlabel: 'n', ylabel: 'xₙ', xlim: [1, 400], ylim: [-2.2, 2], legend: false },
      charts: [{ id: 'b', title: 'b_n = E(xₙ − θ)², 200 runs', xlabel: 'n', ylabel: 'b_n', logx: true, logy: true, ylim: [1e-5, 5], h: 240 }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear().hline(D.theta, { color: HEX.good, dash: [5, 4], text: 'θ' });
        D.paths.forEach((p, k) => P.line(p.map((v, i) => [i + 1, v]), { color: HEX.ball, width: 1.4, alpha: .55 + .2 * k }));
        P.draw();
        const ch = ctx.charts.b.clear().fn(n => 2 / n, { color: HEX.muted, dash: [4, 4], width: 1, label: '∝ 1/n' });
        SC.forEach(([k, , col]) => ch.line(D.bn.map((n, i) => [n + 1, D.b[k.replace('²', '^2')][i]]), { color: col, width: 2, label: k }));
        ch.draw();
        const last = k => D.b[k].at(-1).toExponential(1);
        ctx.stat(`b_n at n = 20 000: 2/n ${last('2/n')} · 0.2 ${last('0.2')} · 2/n² ${last('2/n^2')}`);
      },
    },
  ],
};
