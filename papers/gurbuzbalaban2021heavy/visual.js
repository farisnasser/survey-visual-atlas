// Gürbüzbalaban, Şimşekli, Zhu 2021 — SGD on light-tailed Gaussian data produces heavy-tailed (power-law) iterates.
// Streaming least squares, x_k = M_k x_{k-1} + q_k. α(η, b, d) curves: scripts/check-math/gurbuzbalaban2021heavy.py (data.json).
import { def, C, HEX, ramp } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

/** Tail index α solving h(α) = E‖M e₁‖^α = 1 (Theorem 3), by Monte Carlo with ‖Me₁‖² = 1 − 2(η/b)w₁ + (η/b)²‖w‖², w = Σ a_i(a_i·e₁). */
function tailIndex(eta, b, d, n = 120000, seed = 1) {
  const R = rng(seed), lq = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    const w = new Float64Array(d);
    for (let i = 0; i < b; i++) { const a = Array.from({ length: d }, () => R.normal()); for (let j = 0; j < d; j++) w[j] += a[j] * a[0]; }
    let ww = 0; for (let j = 0; j < d; j++) ww += w[j] * w[j];
    lq[k] = .5 * Math.log(Math.max(1 - 2 * (eta / b) * w[0] + (eta / b) ** 2 * ww, 1e-300));
  }
  const rho = lq.reduce((s, v) => s + v, 0) / n;
  if (rho >= 0) return { alpha: NaN, rho };
  const logh = s => { let m = -Infinity; for (const v of lq) m = Math.max(m, s * v); let t = 0; for (const v of lq) t += Math.exp(s * v - m); return m + Math.log(t / n); };
  let lo = 1e-3, hi = 12; if (logh(hi) < 0) return { alpha: 12, rho };
  for (let i = 0; i < 50; i++) { const mid = (lo + hi) / 2; if (logh(mid) < 0) lo = mid; else hi = mid; }
  return { alpha: (lo + hi) / 2, rho };
}
const etaCrit = (b, d) => 2 * b / (d + b + 1);   // σ = 1
/** One streaming SGD step for every chain (d-dim, batch b, fresh Gaussian data, y independent of a so x* = 0). */
function stepChains(X, d, b, eta, R) {
  for (const x of X) {
    const gr = new Float64Array(d);
    for (let i = 0; i < b; i++) { const a = Array.from({ length: d }, () => R.normal()), y = R.normal(); let r = -y; for (let j = 0; j < d; j++) r += a[j] * x[j]; for (let j = 0; j < d; j++) gr[j] += r * a[j]; }
    for (let j = 0; j < d; j++) x[j] -= eta / b * gr[j];
  }
}
const BOWL = def({ L: (x, y) => .5 * (x * x + y * y + 1), g: (x, y) => [x, y], h: v => Math.log(v), dom: [-4, 4, -4, 4], N: 60 });

export default {
  tabs: [
    {
      id: 'cloud', name: 'Iterate cloud', title: 'A Gaussian problem, a heavy-tailed cloud of SGD iterates',
      caption: '2000 independent SGD runs on streaming linear regression in d = 2 with Gaussian inputs and targets, after many steps. The loss is a perfect bowl and the data have no outliers, yet the stationary cloud of iterates has a power-law tail: most runs sit near the minimum, a few wander very far. Raise the step or shrink the batch and the tail gets heavier.',
      note: 'a ~ N(0, I₂), y ~ N(0, 1) independent, so F(x) = ½(‖x‖² + 1) and x* = 0. Constant step, batch b, fresh data every step (the theorem’s streaming setting). Points beyond the drawn box are clamped to its edge.',
      legend: [[HEX.ball, 'one SGD run (colour = distance from x*)'], [HEX.good, 'minimizer x* = 0']],
      cam: [0, 4.4, 4.6], tgt: [0, .4, 0],
      controls: [{ id: 'eta', label: 'step η', min: .1, max: .9, step: .01, value: .55, digits: 2, live: false }, { id: 'b', type: 'seg', label: 'batch b', value: 1, options: [[1, '1'], [2, '2'], [4, '4']] }],
      build(ctx) {
        const s = ctx.stage, { eta } = ctx.vals, b = +ctx.vals.b; s.surface(BOWL); s.marker(BOWL, 0, 0, C.good, .12);
        ctx.R = rng(3); ctx.X = Array.from({ length: 2000 }, () => new Float64Array(2)); ctx.t = 0;
        const A = tailIndex(eta, b, 2); ctx.A = A;
        ctx.stat(`computing… α = ${isNaN(A.alpha) ? 'none (no contraction: E log‖Me₁‖ ≥ 0)' : A.alpha.toFixed(2)}`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.X || ctx.t >= 400) return;
        const s = ctx.stage, b = +ctx.vals.b;
        for (let k = 0; k < 4; k++) { stepChains(ctx.X, 2, b, ctx.vals.eta, ctx.R); ctx.t++; }
        if (ctx.pts) s.remove(ctx.pts);
        const pos = [], col = [];
        for (const x of ctx.X) { const cx = Math.max(-4, Math.min(4, x[0])), cy = Math.max(-4, Math.min(4, x[1])), v = BOWL.W(cx, cy, .04); pos.push([v.x, v.y, v.z]); col.push(ramp(Math.min(1, Math.hypot(x[0], x[1]) / 4) * .8 + .2)); }
        ctx.pts = s.points(pos, col, .07);
        const r = ctx.X.map(x => Math.hypot(x[0], x[1])).sort((p, q) => p - q), A = ctx.A, ec = etaCrit(b, 2);
        ctx.stat(`step ${ctx.t} · tail index α = ${isNaN(A.alpha) ? 'none (divergent)' : A.alpha.toFixed(2)} (η_crit = ${ec.toFixed(3)} gives α = 2) · median ‖x‖ = ${r[1000].toFixed(2)}, 99.9% quantile = ${r[1997].toFixed(1)}, max = ${r[1999].toExponential(1)}`);
      },
    },
    {
      id: 'tail', name: 'Power-law tail', mode: '2d', title: 'The survival function falls like t^{−α}',
      caption: 'Empirical P(|x₁| > t) over 6000 independent runs after 500 steps, on log–log axes, against the slope −α predicted by Theorem 3 (α computed from h(α) = 1). Below η_crit the tail index is above 2 (finite variance); just above it, α < 2 and the stationary variance is infinite although every data moment is finite.',
      note: 'Same streaming regression as the cloud, now with selectable d. Dashed: c·t^{−α} anchored at the 5% quantile. Finite samples bend the far tail.',
      hint: '', legend: [[HEX.ball, 'empirical P(|x₁| > t)'], [HEX.muted, 'slope −α']],
      controls: [{ id: 'eta', label: 'step η', min: .05, max: 1.4, step: .01, value: .7, digits: 2, live: false }, { id: 'b', type: 'seg', label: 'batch b', value: 1, options: [[1, '1'], [2, '2'], [4, '4']] }, { id: 'd', type: 'seg', label: 'dimension d', value: 1, options: [[1, '1'], [3, '3'], [5, '5']] }],
      plot: { xlabel: 't', ylabel: 'P(|x₁| > t)', logx: true, logy: true, xlim: [.01, 1e4], ylim: [1e-4, 1] },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const { eta } = ctx.vals, b = +ctx.vals.b, d = +ctx.vals.d, R = rng(9), N = 6000, X = Array.from({ length: N }, () => new Float64Array(d));
        for (let k = 0; k < 500; k++) stepChains(X, d, b, eta, R);
        const v = X.map(x => Math.abs(x[0])).filter(isFinite).sort((p, q) => q - p), pts = [];
        for (let i = 0; i < v.length; i += Math.max(1, Math.floor(i / 40))) pts.push([Math.max(v[i], 1e-3), (i + 1) / v.length]);
        const A = tailIndex(eta, b, d), t5 = v[Math.floor(.05 * v.length)], P = ctx.plot.clear();
        if (!isNaN(A.alpha)) P.fn(t => .05 * (t / t5) ** -A.alpha, { color: HEX.muted, dash: [5, 4], width: 1.6 });
        P.line(pts, { color: HEX.ball, width: 2.2 }).draw();
        const ec = etaCrit(b, d);
        ctx.stat(isNaN(A.alpha) ? `η = ${eta.toFixed(2)}: E log‖Me₁‖ = ${A.rho.toFixed(3)} ≥ 0, no stationary law (iterates diverge)` : `α = ${A.alpha.toFixed(2)} · η_crit = 2b/(d + b + 1) = ${ec.toFixed(3)} · ${A.alpha > 2 ? 'finite variance' : A.alpha > 1 ? 'finite mean, infinite variance' : 'infinite mean'}`);
      },
    },
    {
      id: 'alpha', name: 'α(η, b, d)', mode: '2d', title: 'Bigger steps and smaller batches mean heavier tails',
      caption: 'Tail index α against the step η for four batch sizes. α decreases with η and increases with b (Theorem 4); it equals 2 exactly at η_crit = 2b/(σ²(d + b + 1)) (dots, Proposition 5), and curves stop where E log‖Me₁‖ ≥ 0 and no stationary law exists.',
      note: 'Offline Monte Carlo (120 000 samples per curve, σ = 1). Capped at α = 12.',
      hint: '', legend: [[HEX.pos, 'b = 1'], [HEX.good, 'b = 2'], [HEX.flat, 'b = 4'], [HEX.ball, 'b = 8']],
      controls: [{ id: 'd', type: 'seg', label: 'dimension d', value: 5, options: [[1, '1'], [5, '5'], [20, '20']] }],
      plot: { xlabel: 'step η', ylabel: 'tail index α', xlim: [0, 2.6], ylim: [0, 12.5], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data.curves, d = +ctx.vals.d, P = ctx.plot.clear(), cols = [HEX.pos, HEX.good, HEX.flat, HEX.ball];
        P.hline(2, { color: HEX.neg, dash: [5, 4], text: 'α = 2' });
        let xm = 0;
        [1, 2, 4, 8].forEach((b, i) => { const c = D[`${d}-${b}`].filter(r => r[1] != null); xm = Math.max(xm, ...c.map(r => r[0])); P.line(c, { color: cols[i], width: 2.2 }); P.scatter([[etaCrit(b, d), 2]], { color: cols[i], r: 5 }); });
        P.set({ xlim: [0, xm * 1.05] }).draw();
        ctx.stat(`d = ${d}: η_crit = ${[1, 2, 4, 8].map(b => `${etaCrit(b, d).toFixed(3)} (b = ${b})`).join(', ')}`);
      },
    },
  ],
};
