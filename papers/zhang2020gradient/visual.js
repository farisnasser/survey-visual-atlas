// Zhang, He, Sra, Jadbabaie 2020 — (L0, L1)-smoothness and why clipping accelerates: fixed-step GD must pick a step for the
// steepest point it visits, clipped GD does not. Checked in scripts/check-math/zhang2020gradient.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const F = (x, y) => x * x / 2 + x ** 4 / 4 + y * y / 2, G = (x, y) => [x + x ** 3, y];
const surf = def({ L: F, g: G, dom: [-6.5, 6.5, -2, 2], h: v => Math.log1p(v), square: true });
const gd = h => p => { const g = G(...p); return [p[0] - h * g[0], p[1] - h * g[1]]; };
const clip = (eta, gam) => p => { const g = G(...p), h = Math.min(eta, gam * eta / Math.hypot(...g)); return [p[0] - h * g[0], p[1] - h * g[1]]; };
const BEST_H = { 2: .3577, 4: .1111, 6: .0482 };   // best fixed steps from the check script (start (x0, 1))

export default {
  tabs: [
    {
      id: 'steep', name: 'Clipped vs fixed step', title: 'A steep start forces fixed-step GD to crawl everywhere',
      caption: 'f = x²/2 + x⁴/4 + y²/2: its curvature 1 + 3x² grows with the gradient, so it is (L0, L1)-smooth but not L-smooth. Starting far out, fixed-step GD (violet) can only use a step small enough for the steepest point it will visit (the best such step, found by search), and then crawls through the flat middle. Clipped GD (coral) uses step 1 but caps each move at length 0.5, so it walks down the wall and then converges fast.',
      note: 'Deterministic gradients, target ‖∇f‖ ≤ 10⁻⁴. Fixed step = the best of 60 values for this start (check script). Clipped: h = min(1, 0.5/‖∇f‖).',
      legend: [[HEX.opt2, 'GD, best fixed step'], [HEX.ball, 'clipped GD'], [HEX.good, 'minimum']],
      action: 'Restart', cam: [0, 4.2, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'x0', type: 'seg', label: 'start x₀', value: 4, options: [[2, '2'], [4, '4'], [6, '6']] }],
      charts: [{ id: 'g', title: '‖∇f‖ (log)', xlabel: 'step', ylabel: '‖∇f‖', logy: true, ylim: [1e-5, 1e3], xlim: [0, 200], h: 230, legend: false }],
      build(ctx) { ctx.stage.surface(surf); ctx.stage.marker(surf, 0, 0, C.good, .1); this.drop(ctx); },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, x0 = +ctx.vals.x0; s.clearBalls(); ctx.H = { g: [], c: [] };
        const stop = b => Math.hypot(...G(...b.p)) <= 1e-4;
        ctx.bg = s.ball(surf, { start: [x0, 1], speed: .5, maxSteps: 400, step: gd(BEST_H[x0]), color: C.opt2, stop }); ctx.bg.onStep = b => ctx.H.g.push([b.steps, Math.hypot(...G(...b.p))]);
        ctx.bc = s.ball(surf, { start: [x0, 1], speed: .5, maxSteps: 400, step: clip(1, .5), stop }); ctx.bc.onStep = b => ctx.H.c.push([b.steps, Math.hypot(...G(...b.p))]);
      },
      frame(ctx) {
        ctx.charts.g.clear().hline(1e-4, { color: HEX.muted, dash: [3, 4], text: 'ε' }).line(ctx.H.g, { color: HEX.opt2, width: 1.6 }).line(ctx.H.c, { color: HEX.ball, width: 1.8 }).draw();
        const done = b => b.done ? `${b.steps} steps` : 'running';
        ctx.stat(`start (${ctx.vals.x0}, 1), M = ‖∇f(x₀)‖ = ${Math.hypot(...G(+ctx.vals.x0, 1)).toFixed(0)} · fixed step h = ${BEST_H[ctx.vals.x0]}: ${done(ctx.bg)} · clipped: ${done(ctx.bc)}`);
      },
    },
    {
      id: 'scaling', name: 'Cost vs start', mode: '2d', title: 'Fixed-step GD pays for a bad start; clipping barely does',
      caption: 'Iterations to reach ‖∇f‖ ≤ 10⁻⁴ from (x₀, 1) as the starting gradient M grows. The best fixed step has to shrink like 1/M, so GD’s cost grows roughly linearly in M (Theorem 4: Ω(L1·M/log M) times the usual ε⁻² factor; Theorem 6: O(M·L1 + L0) suffices). One clipped-GD setting stays at a handful of steps (Theorem 3: O(L0/ε² + L1²/L0), no M).',
      note: 'Same toy; fixed step tuned separately for every start from 60 values in [10⁻⁴, 1.9]; clipped GD η = 1, γ = 0.5 for all starts.',
      hint: '', legend: [[HEX.opt2, 'GD, best fixed step'], [HEX.ball, 'clipped GD']],
      plot: { xlabel: 'M = ‖∇f(x₀)‖ (log)', ylabel: 'iterations (log)', logx: true, logy: true, xlim: [1, 1000], ylim: [1, 1000], legend: false },
      build(ctx) {
        const R = ctx.data.rows;
        ctx.plot.clear().line(R.map(r => [r[1], r[2]]), { color: HEX.opt2, width: 2.2, dots: 4 }).line(R.map(r => [r[1], r[4]]), { color: HEX.ball, width: 2.2, dots: 4 }).draw();
        ctx.stat(R.map(r => `x₀ = ${r[0]} (M ${r[1]}): GD ${r[2]} (h ${r[3]}), clipped ${r[4]}`).join(' · '));
      },
    },
  ],
};
