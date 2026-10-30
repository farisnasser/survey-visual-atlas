// Chen & Bruna 2023 — beyond the edge of stability GD can converge to a stable period-2 orbit (fixed points of the two-step
// map). Theorems 2–3 / Prop. 3 on f = (xy − μ)²/2 are checked in scripts/check-math/chen2023beyond.py
import { def, C, HEX, hyperbola, segPts } from '../../src/lib/atlas3d.js';

const MU = 1, Z0 = [1.3, .5];
const fxy = def({ L: (x, y) => (x * y - MU) ** 2 / 2, g: (x, y) => { const r = x * y - MU; return [r * y, r * x]; }, dom: [-.1, 2.1, -.1, 2.1], h: v => Math.log1p(v * 4) });
const F1 = (x, K) => x - (K / MU) * x * (x * x - MU);                 // GD on the diagonal = GD on (x^2 - mu)^2 / 4
const dF1 = (x, K) => 1 - (K / MU) * (3 * x * x - MU);
/** Late iterates of the 1-D map from x0 = 0.5. */
function orbit1(K) { let x = .5; for (let i = 0; i < 6000; i++) { x = F1(x, K); if (Math.abs(x) > 1e6) return null; } const o = []; for (let i = 0; i < 64; i++) { x = F1(x, K); o.push(x); } return o; }
function periodOf(o) { if (!o) return -1; for (const p of [1, 2, 4, 8, 16, 32]) { let ok = true; for (let i = 0; i + p < o.length; i++) if (Math.abs(o[i + p] - o[i]) > 1e-8) { ok = false; break; } if (ok) return p; } return 0; }

export default {
  tabs: [
    {
      id: 'orbit', name: 'Period-2 orbit', title: 'Too sharp to settle, GD settles into a two-step bounce',
      caption: 'f(x, y) = ½(xy − 1)². Every minimum on the hyperbola has sharpness (x − y)² + 2, so for η > 1 (K > 1) no minimum is stable. GD does not diverge: from an unbalanced start the iterates are squeezed onto the diagonal x = y (Theorem 3) and end bouncing between two points on either side of the valley (white rings), a stable fixed pair of the two-step map.',
      note: 'η = K/μ with μ = 1, start (1.3, 0.5); x, y stay positive as Theorem 3 assumes. Theorem 2 / Prop. 3 guarantee the 2-cycle for 1 < K < √4.5 − 1 ≈ 1.121; in this run it persists until K ≈ 1.237, then doubles.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'minima xy = 1'], [HEX.white, 'period-2 orbit']],
      action: 'Restart', cam: [3.2, 3.6, 3.8], tgt: [0, .2, 0],
      controls: [{ id: 'K', label: 'K = ημ', min: .8, max: 1.3, step: .005, value: 1.05, digits: 3 }],
      charts: [
        { id: 'gap', title: '|x − y| (log)', xlabel: 'step', ylabel: '|x − y|', logy: true, ylim: [1e-16, 10], xlim: [0, 400], h: 170, legend: false },
        { id: 'res', title: 'xy − 1 (residual)', xlabel: 'step', ylabel: 'xy − 1', ylim: [-.4, .4], xlim: [0, 400], h: 170, legend: false },
      ],
      build(ctx) {
        const s = ctx.stage; s.surface(fxy); s.tube(fxy, hyperbola(1, 2.1, 1), C.good); s.line(fxy, segPts(0, 0, 2.1, 2.1), C.muted, .5);
        const o = orbit1(ctx.vals.K), p = periodOf(o);
        if (o && p >= 1 && p <= 8) for (const v of o.slice(0, p)) s.marker(fxy, v, v, C.white, .07);
        this.drop(ctx);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, K = ctx.vals.K; s.clearBalls(); ctx.hist = [];
        ctx.b = s.ball(fxy, { start: Z0, speed: .5, maxSteps: 400, step: p => { const [gx, gy] = fxy.g(p[0], p[1]), q = [p[0] - K / MU * gx, p[1] - K / MU * gy]; ctx.hist.push([ctx.hist.length + 1, Math.abs(q[0] - q[1]), q[0] * q[1] - MU]); return q; } });
      },
      frame(ctx) {
        const K = ctx.vals.K, H = ctx.hist;
        ctx.charts.gap.clear().line(H.map(h => [h[0], Math.max(h[1], 1e-16)]), { color: HEX.ball, width: 1.4 }).draw();
        ctx.charts.res.clear().hline(0, { color: HEX.good, dash: [4, 4] }).line(H.map(h => [h[0], h[2]]), { color: HEX.pos, width: 1.2, dots: 1.2 }).draw();
        const o = orbit1(K), p = periodOf(o);
        const desc = K < 1 ? `K < 1: converges to a minimum (sharpness < 2/η)` : p === 2 ? `period 2: (${Math.min(o[0], o[1]).toFixed(4)}, ${Math.max(o[0], o[1]).toFixed(4)}) on the diagonal, two-step multiplier F′(a)F′(b) = ${(dF1(o[0], K) * dF1(o[1], K)).toFixed(3)}` : p > 2 ? `period ${p} (beyond the first doubling)` : 'no short period (chaotic band)';
        ctx.stat(`K = ${K.toFixed(3)} · step ${ctx.b.steps} · ${desc}`);
      },
    },
    {
      id: 'bif', name: 'Bifurcation', mode: '2d', title: 'Stable 2-cycle, then period doubling',
      caption: 'Late iterates of GD on the quartic ¼(x² − 1)² (equivalently the diagonal of the 2-D model) for each K = ημ. Below K = 1 it converges to the minimum; just above the edge of stability it converges to a period-2 orbit, which Theorem 2 guarantees up to K ≈ 1.121 and which survives here until K ≈ 1.237; then periods 4, 8, … and chaos.',
      note: 'Start x₀ = 0.5, 3000 steps, then 48 steps plotted (precomputed). Theorem 1’s condition at x* = 1: 3f‴² − f″f⁗ = 96 > 0, which a quadratic (f‴ = f⁗ = 0) fails.',
      hint: '', legend: [[HEX.ball, 'late iterates'], [HEX.flat, 'Theorem 2 range'], [HEX.neg, 'first doubling']],
      plot: { xlabel: 'K = ημ', ylabel: 'late x', xlim: [.8, 1.45], ylim: [0, 1.5], legend: false },
      build(ctx) {
        const D = ctx.data;
        ctx.plot.clear().band(1, D.Kc, { color: 'rgba(255,200,87,.12)' }).vline(1, { color: HEX.muted, dash: [3, 4], text: 'EoS: K = 1' }).vline(D.Kc, { color: HEX.flat, dash: [3, 4], text: '√4.5 − 1' })
          .vline(D.Kpd, { color: HEX.neg, dash: [3, 4], text: `${D.Kpd}` }).hline(1, { color: HEX.good, dash: [2, 4], alpha: .6 })
          .scatter(D.bif, { color: 'rgba(255,122,89,.8)', r: 1 }).draw();
        ctx.stat(`converges for K < 1 · period 2 for 1 < K < ${D.Kpd} (guaranteed by Theorem 2 for K < ${D.Kc.toFixed(3)}) · period 4 from K ≈ ${D.Kpd}`);
      },
    },
  ],
};
