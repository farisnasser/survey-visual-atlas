// Li, Rakhlin, Jadbabaie 2023 — Adam converges under (ρ, L0, Lρ) smoothness: curvature may grow sub-quadratically with the
// gradient norm, and gradients stay bounded along Adam's own trajectory. Checked in scripts/check-math/li2023convergence.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const F = (x, y) => Math.cosh(1.5 * x) + (y * y - 1) ** 2;
const Gf = (x, y) => [1.5 * Math.sinh(1.5 * x), 4 * y * (y * y - 1)];
const surf = def({ L: F, g: Gf, dom: [-4, 4, -2.2, 2.2], h: v => Math.log1p(v), square: true });
const START = [3.6, 1.9], ETA = .05;
const adamStep = (lr, b1 = .9, b2 = .999, eps = 1e-8) => (p, b) => {
  const g = Gf(p[0], p[1]); b.m = b.m || [0, 0]; b.v = b.v || [0, 0]; b.t = (b.t || 0) + 1;
  const q = p.slice();
  for (let i = 0; i < 2; i++) { b.m[i] = b1 * b.m[i] + (1 - b1) * g[i]; b.v[i] = b2 * b.v[i] + (1 - b2) * g[i] * g[i]; q[i] -= lr * (b.m[i] / (1 - b1 ** b.t)) / (Math.sqrt(b.v[i] / (1 - b2 ** b.t)) + eps); }
  return q;
};
const gdStep = lr => p => { const g = Gf(p[0], p[1]); return [p[0] - lr * g[0], p[1] - lr * g[1]]; };
const edge = b => Math.abs(b.p[0]) >= 3.999 || Math.abs(b.p[1]) >= 2.199;
// 1D examples (Def. 3.2, Prop. B.1): [name, f'(x), f''(x), x-range, rho]
const EX = {
  exp: ['eˣ', x => Math.exp(x), x => Math.exp(x), [-2, 9], 'ρ = 1: (L0, L1) smooth'],
  quart: ['x⁴', x => 4 * x ** 3, x => 12 * x * x, [.05, 30], 'ρ = 2/3'],
  inv: ['1/x', x => 1 / (x * x), x => 2 / x ** 3, [.02, 5], 'ρ = 3/2: not (L0, L1) smooth'],
  dexp: ['e^(eˣ)', x => Math.exp(x + Math.exp(x)), x => Math.exp(x + Math.exp(x)) * (1 + Math.exp(x)), [-3, 2.4], 'any ρ > 1, not (L0, L1)'],
};

export default {
  tabs: [
    {
      id: 'adam', name: 'Steep walls', title: 'Curvature that grows with the gradient: GD overshoots, Adam does not',
      caption: 'f = cosh(1.5x) + (y² − 1)². Its curvature grows with the gradient (no global Lipschitz constant), but stays below L0 + L1‖∇f‖, which is the paper’s relaxed smoothness. From a steep start, GD with step 0.05 overshoots further each step and leaves the plot; GD only works with a step small enough for the steepest point it visits (0.01, white). Adam with step 0.05 (coral) takes bounded steps from the start, its gradient norm never exceeds the starting value, and it settles near a minimum (no faster than the well-tuned GD here: the theorem is about not needing that tuning, not about speed).',
      note: 'Deterministic gradients (the theorem allows bounded or sub-Gaussian noise). Adam: β1 = 0.9, β2 = 0.999, bias-corrected. Minima at (0, ±1).',
      legend: [[HEX.ball, 'Adam, η = 0.05'], [HEX.opt2, 'GD, η = 0.05'], [HEX.opt3, 'GD, η = 0.01'], [HEX.good, 'minima']],
      action: 'Restart', cam: [0, 4.6, 4.4], tgt: [0, .2, 0],
      charts: [{ id: 'g', title: '‖∇f(x_t)‖ (log)', xlabel: 'step', ylabel: '‖∇f‖', logy: true, ylim: [1e-6, 1e3], xlim: [0, 1000], h: 230, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(surf); s.marker(surf, 0, 1, C.good, .1); s.marker(surf, 0, -1, C.good, .1);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); ctx.H = { a: [], g: [], s: [] };
        const rec = k => b => ctx.H[k].push([b.steps, Math.hypot(...Gf(...b.p))]);
        ctx.ba = s.ball(surf, { start: START, speed: 3, maxSteps: 1000, step: adamStep(ETA) }); ctx.ba.onStep = rec('a');
        ctx.bg = s.ball(surf, { start: START, speed: 3, maxSteps: 1000, step: gdStep(ETA), color: C.opt2, stop: edge }); ctx.bg.onStep = rec('g');
        ctx.bs = s.ball(surf, { start: START, speed: 3, maxSteps: 1000, step: gdStep(.01), color: C.opt3 }); ctx.bs.onStep = rec('s');
      },
      frame(ctx) {
        const g0 = Math.hypot(...Gf(...START)), P = ctx.charts.g.clear().hline(g0, { color: HEX.muted, dash: [3, 4], text: 'start' });
        P.line([[0, g0], ...ctx.H.s], { color: HEX.opt3, width: 1.4 }).line([[0, g0], ...ctx.H.g], { color: HEX.opt2, width: 1.4, dots: 2 }).line([[0, g0], ...ctx.H.a], { color: HEX.ball, width: 1.8 }).draw();
        const mx = Math.max(g0, ...ctx.H.a.map(h => h[1]));
        ctx.stat(`step ${ctx.ba.steps} · Adam ‖∇f‖ = ${Math.hypot(...Gf(...ctx.ba.p)).toExponential(1)} (max along the path ${mx.toFixed(1)} = start) · GD η = 0.05: ${edge(ctx.bg) ? 'left the plot after ' + ctx.bg.steps + ' steps' : 'running'} · GD η = 0.01: ‖∇f‖ = ${Math.hypot(...Gf(...ctx.bs.p)).toExponential(1)}`);
      },
    },
    {
      id: 'classes', name: '(ρ, L0, Lρ) smoothness', mode: '2d', title: 'How fast may curvature grow with the gradient?',
      caption: 'Definition 3.2 asks for ‖∇²f‖ ≤ L0 + Lρ‖∇f‖^ρ with ρ < 2. On log–log axes a function’s points (|f′|, |f″|) eventually follow a slope ρ. eˣ has slope 1, Zhang et al.’s (L0, L1) case; 1/x has slope 3/2 and the double exponential bends ever steeper than 1, so neither is (L0, L1) smooth, but both are covered here. Slope 2 is excluded.',
      note: 'Exact derivatives of the 1D examples from Prop. B.1 and Remark 3.3. Dashed: reference slopes 1, 3/2 and 2 through (1, 1).',
      hint: '', legend: [[HEX.ball, 'selected function'], [HEX.muted, 'slopes 1, 3/2, 2']],
      controls: [{ id: 'f', type: 'seg', label: 'function', value: 'inv', options: [['exp', 'eˣ'], ['quart', 'x⁴'], ['inv', '1/x'], ['dexp', 'e^(eˣ)']] }],
      plot: { xlabel: '|f′(x)|', ylabel: '|f″(x)|', logx: true, logy: true, xlim: [1e-2, 1e6], ylim: [1e-2, 1e8], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const [name, d1, d2, [a, b], tag] = EX[ctx.vals.f], pts = [];
        for (let i = 0; i <= 300; i++) { const x = a + (b - a) * i / 300, u = Math.abs(d1(x)), w = Math.abs(d2(x)); if (isFinite(u) && isFinite(w) && u > 0 && w > 0) pts.push([u, w]); }
        const P = ctx.plot.clear();
        [[1, '1'], [1.5, '3/2'], [2, '2 (excluded)']].forEach(([s, l]) => P.fn(u => u ** s, { color: s === 2 ? HEX.neg : HEX.muted, dash: [5, 4], width: 1.2 }).text(1e5, Math.min(5e7, 1e5 ** s), ` ${l}`, { color: s === 2 ? HEX.neg : HEX.muted }));
        P.line(pts, { color: HEX.ball, width: 2.4 }).draw();
        const k = pts.length - 1, j = Math.floor(pts.length * .8), slope = Math.log(pts[k][1] / pts[j][1]) / Math.log(pts[k][0] / pts[j][0]);
        ctx.stat(`${name}: ${tag} · slope over the last 20% of the curve ${slope.toFixed(2)}`);
      },
    },
  ],
};
