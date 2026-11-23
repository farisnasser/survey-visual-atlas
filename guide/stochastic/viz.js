// Chapter 5 (survey Section II-F). Visuals 'bound' (the T-step sum of Eq. 11; numbers in scripts/check-math/guide3_numbers.py)
// and 'batchvar' (guide3_notes.py); reused crash-course visuals 'cloud' and 'floor' (course_noise.py).
import { HEX } from '../../src/lib/atlas3d.js';
import nz from '../../course/noise/viz.js';

const D0 = 1, BETA = 1, S2 = 1;
const term1 = (eta, T) => 2 * D0 / (eta * T), term2 = (eta, B) => BETA * eta * S2 / B;
const ebal = (B, T) => Math.sqrt(2 * D0 * B / (BETA * S2 * T));
const etaUse = (B, T) => Math.min(1 / BETA, ebal(B, T));
const bound = (B, T) => { const e = etaUse(B, T); return term1(e, T) + term2(e, B); };

const boundViz = {
  title: 'The T-step SGD bound: two terms pulling the step size apart',
  caption: 'The average expected squared gradient over T steps is at most 2Δ₀/(ηT) + βησ²/B. A bigger step shrinks the first term (progress) and grows the second (noise). The balanced step η_bal sits where they cross, capped at 1/β.',
  note: 'Illustrative constants Δ₀ = L(θ₀) − L* = 1, β = 1, σ² = 1. These are worst-case upper bounds, not training curves.',
  legend: [[HEX.pos, '2Δ₀/(ηT): progress term'], [HEX.neg, 'βησ²/B: noise term'], ['#e8ecf8', 'their sum, the bound (76)'], [HEX.flat, 'η = min(1/β, η_bal)']],
  mode: '2d',
  plot: { xlabel: 'step size η (log scale)', ylabel: 'bound on the average E‖g_t‖²', logx: true, logy: true, xlim: [1e-4, 1], ylim: [1e-5, 100], legend: false },
  controls: [
    { id: 'lb', label: 'batch size B', min: 0, max: 10, step: 1, value: 5, fmt: v => String(2 ** v) },
    { id: 'lt', label: 'iterations T', min: 1, max: 6, step: .05, value: 3, fmt: v => Math.round(10 ** v).toLocaleString('en') },
  ],
  charts: [{ id: 't', title: 'the bound at the chosen η, against T (log-log)', xlabel: 'iterations T', ylabel: 'bound', logx: true, logy: true, xlim: [10, 1e6], ylim: [1e-6, 10], h: 200, legend: false }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const B = 2 ** ctx.vals.lb, T = 10 ** ctx.vals.lt, P = ctx.plot.clear();
    const es = Array.from({ length: 300 }, (_, i) => 10 ** (-4 + 4 * i / 299));
    P.line(es.map(e => [e, term1(e, T)]), { color: HEX.pos, width: 1.6, dash: [5, 4] });
    P.line(es.map(e => [e, term2(e, B)]), { color: HEX.neg, width: 1.6, dash: [5, 4] });
    P.line(es.map(e => [e, term1(e, T) + term2(e, B)]), { color: '#e8ecf8', width: 2.2 });
    const eb = ebal(B, T), eu = etaUse(B, T), bu = bound(B, T);
    P.vline(Math.min(eb, 1), { color: HEX.flat, dash: [3, 3], text: eb <= 1 ? 'η_bal' : 'η_bal > 1/β' });
    P.vline(1 / BETA, { color: 'rgba(255,255,255,.3)', dash: [2, 4], text: '1/β' });
    P.scatter([[eu, bu]], { color: HEX.flat, r: 5 });
    P.draw();
    const Ts = Array.from({ length: 200 }, (_, i) => 10 ** (1 + 5 * i / 199));
    const C = ctx.charts.t.clear();
    C.line(Ts.map(t => [t, bound(B, t)]), { color: '#e8ecf8', width: 2 });
    C.line(Ts.map(t => [t, 2 * D0 * BETA / t]), { color: HEX.pos, width: 1.6 });
    C.scatter([[T, bu]], { color: HEX.flat, r: 5 });
    C.draw();
    ctx.stat(`B = ${B}, T = ${Math.round(T).toLocaleString('en')} · η_bal = √(2Δ₀B/(βσ²T)) = ${eb.toPrecision(3)} → η = ${eu.toPrecision(3)} · bound = ${bu.toPrecision(3)} (progress ${term1(eu, T).toPrecision(3)} + noise ${term2(eu, B).toPrecision(3)}) · deterministic GD with η = 1/β: ${(2 * D0 * BETA / T).toPrecision(3)} · side chart: white = this bound against T (slope −1/2 once η_bal < 1/β), blue = the noiseless 2Δ₀β/T (slope −1)`);
  },
};

// ---------- 'batchvar': the exact distribution of a size-B minibatch mean of 2, 4, 6, 8, 10 (with replacement) ----------
const SAMPLES = [2, 4, 6, 8, 10], MEAN = 6, POPVAR = 8;
function batchMeans(B) {   // all 5^B equally likely batches -> {mean: count}
  const counts = new Map(); const idx = new Array(B).fill(0);
  for (;;) {
    const m = idx.reduce((s, i) => s + SAMPLES[i], 0) / B, k = m.toFixed(6);
    counts.set(k, (counts.get(k) || 0) + 1);
    let j = 0; while (j < B && ++idx[j] === SAMPLES.length) { idx[j] = 0; j++; }
    if (j === B) break;
  }
  return [...counts.entries()].map(([k, c]) => [+k, c / SAMPLES.length ** B]).sort((a, b) => a[0] - b[0]);
}
const batchvar = {
  title: 'Averaging more examples: the minibatch estimate stays centred and narrows like σ²/B',
  caption: 'Five per-example gradients 2, 4, 6, 8, 10 (true gradient 6, variance σ² = 8). The bars are the exact probabilities of every possible minibatch average of size B, drawing examples with replacement. The estimate is unbiased at every B, and its variance is exactly σ²/B.',
  note: 'Exact enumeration of all 5^B batches, no sampling. Drawing without replacement from a finite set would shrink the variance a little more.',
  legend: [[HEX.ball, 'probability of each minibatch average'], [HEX.good, 'true gradient 6']],
  mode: '2d',
  plot: { xlabel: 'minibatch gradient estimate ĝ', ylabel: 'probability', xlim: [1, 11], ylim: [0, .45], legend: false },
  controls: [{ id: 'B', label: 'batch size B', min: 1, max: 5, step: 1, value: 1, digits: 0 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const B = ctx.vals.B, dist = batchMeans(B), P = ctx.plot.clear(), w = .9 / B;
    P.custom((g, plot) => {
      g.fillStyle = HEX.ball;
      for (const [m, p] of dist) { const x0 = plot.sx(m - w / 2), x1 = plot.sx(m + w / 2), y0 = plot.sy(0), y1 = plot.sy(p); g.fillRect(x0, y1, Math.max(1, x1 - x0), y0 - y1); }
    });
    P.vline(MEAN, { color: HEX.good, width: 1.6, text: 'true gradient 6' });
    P.draw();
    const mean = dist.reduce((s, [m, p]) => s + m * p, 0), v = dist.reduce((s, [m, p]) => s + p * (m - mean) ** 2, 0);
    ctx.stat(`B = ${B} · ${dist.length} possible averages from ${(5 ** B).toLocaleString('en')} equally likely batches · mean of the estimate = ${mean.toFixed(4)} (unbiased) · variance = ${v.toFixed(4)} = σ²/B = 8/${B} · standard deviation = ${Math.sqrt(v).toFixed(3)}`);
  },
};

export default { cloud: nz.cloud, floor: nz.floor, batchvar, bound: boundViz };
