// Ahn, Magakyan, Cutkosky 2025 — schedule-free SGD reaches (λ, ε)-stationary points of nonsmooth nonconvex losses at the
// optimal rate O(λ^(1/2) ε^(−7/2)), with β_t ~ U[β, 1] near 1 and an EMA x (c_t = 1 − β). Checked in scripts/check-math/ahn2025general.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const SIG = .5, START = [1.3, -1.1];
const F = (x, y) => Math.abs(x) + Math.abs(y) + .5 * Math.sin(2 * x) * Math.sin(2 * y);
const G = (x, y) => [Math.sign(x) + Math.cos(2 * x) * Math.sin(2 * y), Math.sign(y) + Math.sin(2 * x) * Math.cos(2 * y)];
const surf = def({ L: F, g: G, dom: [-1.6, 1.6, -1.6, 1.6], h: v => v, checkDom: [.05, 1.5, .05, 1.5] });

export default {
  tabs: [
    {
      id: 'kink', name: 'Nonsmooth landscape', title: 'At a kink the gradients never shrink, but their average does',
      caption: 'A nonsmooth, nonconvex loss whose minimum sits on a kink. Schedule-free SGD with the paper’s nonconvex settings: the base iterate z (violet) takes stochastic gradient steps, x (coral) is an exponential average of z, and the gradient is taken at y (white), a random point between them with β_t drawn near 1. Near the kink each gradient still has length about 1.4, yet the gradients collected around x average to almost zero: that is what (λ, ε)-stationarity measures.',
      note: 'f = |x| + |y| + 0.5 sin 2x sin 2y, Gaussian gradient noise σ = 0.5, step γ = 0.1, β_t ~ U[0.9, 1], c_t = 1 − β = 0.1. Readout over the last 300 steps.',
      legend: [[HEX.opt2, 'z: base SGD'], [HEX.opt3, 'y: gradient point'], [HEX.ball, 'x: EMA of z (output)'], [HEX.good, 'minimum (on the kinks)']],
      action: 'Restart', cam: [3, 3.8, 3.8], tgt: [0, .1, 0],
      build(ctx) { ctx.stage.surface(surf); ctx.stage.marker(surf, 0, 0, C.good, .1); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(3), st = { z: START.slice(), x: START.slice(), y: START.slice(), t: 0, G: [] }; s.clearBalls(); ctx.st = st;
        ctx.bx = s.ball(surf, { start: START, speed: 2, maxSteps: 4000, maxTrail: 4000, step: () => {
          st.t++; st.x = [.9 * st.x[0] + .1 * st.z[0], .9 * st.x[1] + .1 * st.z[1]];
          const b = .9 + .1 * R(); st.y = [(1 - b) * st.z[0] + b * st.x[0], (1 - b) * st.z[1] + b * st.x[1]];
          const g = G(...st.y); st.G.push(g); if (st.G.length > 300) st.G.shift();
          st.z = [st.z[0] - .1 * (g[0] + SIG * R.normal()), st.z[1] - .1 * (g[1] + SIG * R.normal())]; return st.x.slice();
        } });
        ctx.bz = s.ball(surf, { start: START, speed: 2, maxSteps: 4000, maxTrail: 500, color: C.opt2, step: () => st.z.slice() });
        ctx.by = s.ball(surf, { start: START, speed: 2, maxSteps: 4000, trail: false, r: .04, color: C.opt3, step: () => st.y.slice() });
      },
      frame(ctx) {
        const Gs = ctx.st.G, n = Gs.length || 1, m = [Gs.reduce((a, g) => a + g[0], 0) / n, Gs.reduce((a, g) => a + g[1], 0) / n], each = Gs.reduce((a, g) => a + Math.hypot(...g), 0) / n;
        ctx.stat(`step ${ctx.st.t} · |x| = ${Math.hypot(...ctx.st.x).toFixed(3)} · last ${Gs.length} true gradients at y: mean length ${each.toFixed(2)}, length of their mean ${Math.hypot(...m).toFixed(3)}`);
      },
    },
    {
      id: 'rates', name: 'Optimal rates', mode: '2d', title: 'One conversion, the optimal rate in three settings',
      caption: 'Stochastic-gradient calls needed (up to constants) against the target ε. Schedule-free SGD finds a (λ, ε)-stationary point with O(λ^(1/2) ε^(−7/2)) calls, the optimal rate for nonsmooth nonconvex problems. Proposition 1 turns this into ‖∇F‖ ≤ 2ε with O(ε⁻⁴) calls for smooth losses and O(ε^(−3.5)) for losses with a Lipschitz Hessian, both optimal for their classes.',
      note: 'Exponents only (constants set to 1); λ = 1 for the nonsmooth curve. Smooth: λ = L²/ε; second-order smooth: λ = H/2.',
      hint: '', legend: [[HEX.ball, 'nonsmooth: λ^(1/2) ε^(−7/2)'], [HEX.opt2, 'smooth: ε⁻⁴'], [HEX.opt3, 'second-order smooth: ε^(−3.5)']],
      plot: { xlabel: 'target ε (log)', ylabel: 'gradient calls (log)', logx: true, logy: true, xlim: [1e-3, 1], ylim: [1, 1e13], legend: false },
      build(ctx) {
        ctx.plot.clear().fn(e => e ** -4, { color: HEX.opt2, width: 2.2 }).fn(e => e ** -3.5, { color: HEX.opt3, width: 2, dash: [6, 3] }).fn(e => e ** -3.5, { color: HEX.ball, width: 1.4 }).draw();
        ctx.stat('at ε = 10⁻²: ε⁻⁴ = 10⁸ calls (smooth), ε^(−3.5) = 10⁷ (second-order smooth or nonsmooth with λ = 1)');
      },
    },
  ],
};
