// Nesterov 2004 (Introductory Lectures) — the background the survey uses: L-smooth functions, the descent lemma and the
// gradient method's guarantee, which holds without convexity. Checked in scripts/check-math/nesterov2004introductory.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';

const F = def({ L: (x, y) => x * x / 4 - Math.cos(2 * y) / 4, g: (x, y) => [x / 2, Math.sin(2 * y) / 2], hess: (x, y) => [.5, 0, Math.cos(2 * y)], h: v => v, dom: [-2.4, 2.4, -.8, 3.9] });
const X0 = [1.8, 1.3], FS = -.25;
const RUNS = [['h = 1/L', 1, C.ball, HEX.ball], ['h = 1.9/L', 1.9, C.opt2, HEX.opt2], ['h = 2.2/L', 2.2, C.opt3, HEX.opt3]];

export default {
  tabs: [
    {
      id: 'descent', name: 'Step 1/L', title: 'With an L-Lipschitz gradient, a step of 1/L always goes down — convex or not',
      caption: 'f(x, y) = x²/4 − cos(2y)/4 is non-convex (negative curvature where cos 2y < 0, saddle at (0, π/2)) but its gradient is exactly 1-Lipschitz. The descent lemma f(y) ≤ f(x) + ⟨∇f(x), y − x⟩ + (L/2)‖y − x‖² guarantees that a step h < 2/L lowers f by h(1 − Lh/2)‖∇f‖², most at h = 1/L. Coral (h = 1/L) slides monotonically to a minimum; violet (1.9/L) still descends but zig-zags; white (2.2/L) has no guarantee and f goes up.',
      note: 'Start (1.8, 1.3). Right: for h = 1/L, the smallest squared gradient so far stays under 2L(f(x₀) − f*)/(k + 1) — an O(1/k) rate for ‖∇f‖², i.e. O(ε⁻²) steps to ‖∇f‖ ≤ ε. Convexity is never used; it would only add that ∇f = 0 means a global minimum.',
      legend: [[HEX.ball, 'h = 1/L'], [HEX.opt2, 'h = 1.9/L'], [HEX.opt3, 'h = 2.2/L'], [HEX.good, 'minima']],
      cam: [3.6, 3.4, 3.6], tgt: [0, .2, 0], action: 'Run again',
      charts: [{ id: 'g', title: 'min_{i≤k} ‖∇f(x_i)‖² (h = 1/L) and the bound', xlabel: 'k', ylabel: '‖∇f‖²', logx: true, logy: true, xlim: [1, 200], ylim: [1e-12, 2], h: 220 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.marker(F, 0, 0, C.good); s.marker(F, 0, Math.PI, C.good); s.eigenLines(F, 0, Math.PI / 2, { len: .6 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.b = RUNS.map(([, h, col]) => s.ball(F, { start: X0, step: opt.gd(h), color: col, speed: .2, maxSteps: 200 }));
        for (const q of ctx.b) { q.ups = 0; q.prev = q.loss(); q.onStep = z => { const v = z.loss(); if (v > z.prev + 1e-12) z.ups++; z.prev = v; }; }
      },
      frame(ctx) {
        const B = ctx.b, D = ctx.data; if (!B) return;
        let m = Infinity; const mins = D.G2.map((v, i) => [i + 1, (m = Math.min(m, v))]);
        ctx.charts.g.clear().fn(k => 2 * D.L * (F.L(...X0) - FS) / k, { color: HEX.muted, dash: [5, 4], width: 1.2, label: '2L(f₀ − f*)/(k+1)' }).line(mins, { color: HEX.ball, width: 2, label: 'measured' }).draw();
        ctx.stat(RUNS.map(([n], k) => `${n}: f = ${B[k].loss().toFixed(4)}, went up ${B[k].ups}×`).join(' · '));
      },
    },
  ],
};
