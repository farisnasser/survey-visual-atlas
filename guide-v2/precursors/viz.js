// Chapter 9 (survey Sections IV-A and IV-B). New visual 'sgdstab' (checked in scripts/check-math/guide_precursors.py);
// reused paper tabs from lewkowycz2020large (scripts/check-math/lewkowycz2020large.py).
import { HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';
import { draggable } from '../../src/learn/plot2d.js';
import { paperTab } from '../lib.js';

// Stability of SGD along one direction: N examples with curvatures h_i (mean lam, population sd s); a batch of S
// examples drawn without replacement multiplies psi by (1 - eta * mean_B h). Mean square factor (survey Eq. 11):
// (1 - eta lam)^2 + (eta s)^2 (N - S) / (S (N - 1)).
const N = 100, SIZES = [1, 4, 16, 64, 100], COL = { 1: HEX.neg, 4: HEX.ball, 16: HEX.flat, 64: HEX.opt2, 100: HEX.good };
const factor = (x, y, S) => (1 - x) ** 2 + y * y * (N - S) / (S * (N - 1));
const edgeY = (x, S) => S === N ? Infinity : Math.sqrt(Math.max(0, 1 - (1 - x) ** 2) * S * (N - 1) / (N - S));
const Z = (() => { const r = rng(7), z = Array.from({ length: N }, () => r.normal()); const m = z.reduce((a, b) => a + b) / N, sd = Math.sqrt(z.reduce((a, b) => a + (b - m) ** 2, 0) / N); return z.map(v => (v - m) / sd); })();
const T = 60, RUNS = 250;

const sgdstab = {
  title: 'Which curvatures can SGD stay in? The stability region shrinks with the batch size',
  caption: 'Each training example has its own curvature along the top direction: mean λ, spread s. SGD with batch S is stable in mean square exactly inside the colored region of its batch size (survey Eq. 11). Drag the point; the chart runs real SGD there.',
  note: 'N = 100 examples with curvatures h_i = λ + s·z_i (z standardized, fixed seed). Batches drawn without replacement. Chart: average of ψ_t² over 250 SGD runs (dots) against the exact mean-square law (line), η = 1.',
  legend: [[HEX.neg, 'S = 1'], [HEX.ball, 'S = 4'], [HEX.flat, 'S = 16'], [HEX.opt2, 'S = 64'], [HEX.good, 'S = N = 100 (full batch)']],
  mode: '2d',
  plot: { xlabel: 'ηλ (step × mean curvature)', ylabel: 'ηs (step × spread of curvatures)', xlim: [0, 2.4], ylim: [0, 5], legend: false },
  controls: [{ id: 'S', type: 'seg', label: 'batch size S', value: 4, options: SIZES.map(s => [s, s === N ? '100 (full)' : String(s)]) }],
  charts: [{ id: 'ms', title: 'mean square of the distance ψ_t² (log)', xlabel: 'step t', ylabel: 'E ψ_t²', logy: true, xlim: [0, T], ylim: [1e-8, 1e8], h: 220, legend: false }],
  build(ctx) {
    ctx.pt = [1.5, 1.0];
    draggable(ctx, p => { ctx.pt = [Math.min(2.4, Math.max(0, p[0])), Math.min(5, Math.max(0, p[1]))]; this.draw(ctx); });
    this.draw(ctx);
  },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const S = +ctx.vals.S, [x, y] = ctx.pt, P = ctx.plot.clear();
    // shaded region of the chosen batch size
    P.custom((g, p) => {
      g.fillStyle = 'rgba(255,255,255,.06)'; g.beginPath();
      g.moveTo(p.sx(0), p.sy(0));
      for (let i = 0; i <= 200; i++) { const xx = 2 * i / 200; g.lineTo(p.sx(xx), p.sy(Math.min(5, edgeY(xx, S)))); }
      g.lineTo(p.sx(2), p.sy(0)); g.closePath(); g.fill();
    });
    for (const s of SIZES) {
      const pts = [];
      if (s === N) { P.vline(0, { color: COL[s], width: s === S ? 2.6 : 1.2 }); P.vline(2, { color: COL[s], width: s === S ? 2.6 : 1.2, text: 'full batch: ηλ = 2' }); continue; }
      for (let i = 0; i <= 200; i++) { const xx = 2 * i / 200; pts.push([xx, edgeY(xx, s)]); }
      P.line(pts, { color: COL[s], width: s === S ? 2.6 : 1.1, alpha: s === S ? 1 : .55 });
    }
    const f = factor(x, y, S), ok = f <= 1;
    P.scatter([[x, y]], { color: ok ? HEX.good : HEX.neg, r: 7 });
    P.text(x, y, ok ? ' stable' : ' unstable', { color: ok ? HEX.good : HEX.neg, dx: 9 });
    P.draw();
    // real SGD at this point (eta = 1): lam = x, s = y
    const r = rng(11), h = Z.map(z => x + y * z), ms = new Array(T + 1).fill(0), idx = Array.from({ length: N }, (_, i) => i);
    for (let k = 0; k < RUNS; k++) {
      let psi = 1; ms[0] += 1;
      for (let t = 1; t <= T; t++) {
        for (let i = 0; i < S; i++) { const j = i + Math.floor(r() * (N - i)); [idx[i], idx[j]] = [idx[j], idx[i]]; }
        let hb = 0; for (let i = 0; i < S; i++) hb += h[idx[i]]; hb /= S;
        psi *= 1 - hb; ms[t] += psi * psi;
      }
    }
    const A = ctx.charts.ms.clear();
    A.hline(1, { color: 'rgba(255,255,255,.3)', dash: [3, 4] });
    A.fn(t => f ** t, { color: '#e8ecf8', width: 1.6 });
    A.scatter(ms.map((v, t) => [t, v / RUNS]), { color: COL[S], r: 2.6 });
    A.draw();
    const verdict = s => factor(x, y, s) <= 1 ? 'stable' : 'unstable';
    ctx.stat(`ηλ = ${x.toFixed(2)}, ηs = ${y.toFixed(2)} · factor per step (1 − ηλ)² + (ηs)²(N − S)/(S(N − 1)) = ${f.toFixed(3)} for S = ${S} → ${ok ? 'stable: ψ² shrinks on average' : 'unstable: ψ² grows on average'} · by batch size: ${SIZES.map(s => `S=${s} ${verdict(s)}`).join(', ')}`);
  },
};

export default {
  sgdstab,
  spike: await paperTab('lewkowycz2020large', 'spike', { title: 'A catapult: the loss spikes, the curvature drops, training converges somewhere flatter' }),
  phases: await paperTab('lewkowycz2020large', 'phases', { title: 'Three phases of the learning rate: lazy, catapult, divergent' }),
};
