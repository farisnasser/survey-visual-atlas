// Chapter 15 (survey Sections V-D, V-E). New visual 'hutch' (survey Section V-E); numbers in scripts/check-math/guide3_numbers.py.
// Reused: crash-course 'clip' (course_adaptive.py).
import { HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';
import ad from '../../course/adaptive/viz.js';

const D = 12, K = 1000;
const R = rng(65);
const Hm = (() => { const A = Array.from({ length: D }, () => Array.from({ length: D }, () => R.normal())); const H = A.map((r, i) => r.map((_, j) => .35 * (A[i][j] + A[j][i]))); for (let i = 0; i < D; i++) H[i][i] += 1.2 * Math.sin(1.3 * i + .4); return H; })();
const DIAG = Hm.map((r, i) => r[i]);
const OFF2 = Hm.reduce((s, r, i) => s + r.reduce((t, v, j) => t + (i === j ? 0 : v * v), 0), 0);   // sum of squared off-diagonal entries
const NRM = Math.hypot(...DIAG);
// one fixed probe sequence: running mean of z ⊙ (Hz)
const RUN = (() => { const Z = rng(66), sum = new Array(D).fill(0), out = []; for (let k = 1; k <= K; k++) { const z = Array.from({ length: D }, () => Z() < .5 ? -1 : 1); const Hz = Hm.map(r => r.reduce((s, v, j) => s + v * z[j], 0)); for (let i = 0; i < D; i++) sum[i] += z[i] * Hz[i]; out.push(sum.map(v => v / k)); } return out; })();
const relErr = k => Math.hypot(...RUN[k - 1].map((v, i) => v - DIAG[i])) / NRM;

const hutch = {
  title: 'Hutchinson\'s estimator: the Hessian diagonal from random ±1 probes',
  caption: 'For a random vector z with independent ±1 entries, z ⊙ (Hz) is an unbiased estimate of diag(H) (survey Section V-E); each probe costs one Hessian-vector product. Bars: the true diagonal (white outline) and the average of the first k probes (coral, or red where the estimate is negative). The chart shows the relative error as k grows.',
  note: 'A fixed, seeded 12 × 12 symmetric H with some negative diagonal entries; one fixed probe sequence. Dashed: the exact expected error √(Σ_{i≠j} H²ᵢⱼ / k)/‖diag H‖, since one probe\'s error in entry i has variance Σ_{j≠i} H²ᵢⱼ.',
  legend: [['#e8ecf8', 'true diag(H)'], [HEX.ball, 'estimate, positive'], [HEX.neg, 'estimate, negative']],
  mode: '2d',
  plot: { xlabel: 'coordinate i', ylabel: 'H_ii', xlim: [-.7, 11.7], ylim: [-3, 3], legend: false },
  controls: [{ id: 'lk', label: 'number of probes k', min: 0, max: 3, step: .01, value: 1, fmt: v => String(Math.round(10 ** v)) }],
  charts: [{ id: 'e', title: 'relative error of the estimate (log-log)', xlabel: 'probes k', ylabel: 'error', logx: true, logy: true, xlim: [1, K], ylim: [3e-3, 5], h: 200, legend: false }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const k = Math.max(1, Math.round(10 ** ctx.vals.lk)), est = RUN[k - 1], P = ctx.plot.clear();
    P.hline(0, { color: 'rgba(255,255,255,.3)' });
    P.custom((g, p) => {
      for (let i = 0; i < D; i++) {
        const w = (p.sx(1) - p.sx(0)) * .32, x = p.sx(i);
        g.fillStyle = est[i] >= 0 ? 'rgba(255,122,89,.85)' : 'rgba(255,93,115,.85)';
        const y0 = p.sy(0), y1 = p.sy(Math.max(-3, Math.min(3, est[i]))); g.fillRect(x - w, Math.min(y0, y1), w, Math.abs(y1 - y0));
        g.strokeStyle = '#e8ecf8'; g.lineWidth = 1.4; const t1 = p.sy(DIAG[i]); g.strokeRect(x, Math.min(y0, t1), w, Math.abs(t1 - y0));
      }
    });
    P.draw();
    const E = ctx.charts.e.clear();
    E.line(Array.from({ length: K }, (_, j) => [j + 1, relErr(j + 1)]), { color: HEX.ball, width: 1.6 });
    E.line(Array.from({ length: 60 }, (_, j) => { const kk = 10 ** (3 * j / 59); return [kk, Math.sqrt(OFF2 / kk) / NRM]; }), { color: '#e8ecf8', width: 1, dash: [4, 4] });
    E.scatter([[k, relErr(k)]], { color: HEX.flat, r: 5 });
    E.draw();
    const neg = DIAG.filter(v => v < 0).length, wrong = est.filter((v, i) => Math.sign(v) !== Math.sign(DIAG[i])).length;
    ctx.stat(`k = ${k} probes (= ${k} Hessian-vector products) · relative error ${relErr(k).toFixed(3)} (expected ${(Math.sqrt(OFF2 / k) / NRM).toFixed(3)}) · true diagonal has ${neg} negative entries; the estimate has the wrong sign in ${wrong} of ${D} coordinates`);
  },
};

export default { hutch, clip: ad.clip };
