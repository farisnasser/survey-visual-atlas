// Polyak 1964 — the heavy-ball method x_{t+1} = x_t − α∇f(x_t) + β(x_t − x_{t−1}); with the optimal α, β on a quadratic it
// contracts by (√κ − 1)/(√κ + 1) per step instead of GD's (κ − 1)/(κ + 1). Checked in scripts/check-math/polyak1964some.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const MU = .2, START = [1.4, 1.8];
const mk = L => def({ L: (x, y) => .5 * (L * x * x + MU * y * y), g: (x, y) => [L * x, MU * y], dom: [-2, 2, -2, 2], h: v => Math.log1p(v) });
const params = L => ({ a: 4 / (Math.sqrt(L) + Math.sqrt(MU)) ** 2, b: ((Math.sqrt(L) - Math.sqrt(MU)) / (Math.sqrt(L) + Math.sqrt(MU))) ** 2, h: 2 / (L + MU) });
const hbRho = k => (Math.sqrt(k) - 1) / (Math.sqrt(k) + 1), gdRho = k => (k - 1) / (k + 1);

export default {
  tabs: [
    {
      id: 'ball', name: 'Heavy ball', title: 'Remembering the last step speeds up a narrow valley',
      caption: 'A quadratic valley with condition number κ = L/μ. Gradient descent with its best step (violet) zig-zags across the steep direction and creeps along the flat one. The heavy ball (coral) adds a fraction β of its previous displacement, so motion along the flat direction builds up speed while the cross-valley oscillation is damped; with Polyak’s optimal α and β it needs about √κ times fewer steps.',
      note: 'f = ½(L x² + 0.2 y²). GD: step 2/(L + μ). Heavy ball: α = 4/(√L + √μ)², β = ((√L − √μ)/(√L + √μ))². Deterministic gradients.',
      legend: [[HEX.opt2, 'GD, best step'], [HEX.ball, 'heavy ball, optimal α, β'], [HEX.good, 'minimum']],
      action: 'Restart', cam: [0, 4.4, 4.2], tgt: [0, .1, 0],
      controls: [{ id: 'L', type: 'seg', label: 'κ = L/μ', value: 5, options: [[1, '5'], [5, '25'], [20, '100']] }],
      charts: [{ id: 'f', title: 'f(x_t) (log)', xlabel: 'step', ylabel: 'f', logy: true, ylim: [1e-12, 10], xlim: [0, 300], h: 230, legend: false }],
      build(ctx) { ctx.d = mk(+ctx.vals.L); ctx.stage.surface(ctx.d); ctx.stage.marker(ctx.d, 0, 0, C.good, .1); this.drop(ctx); },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, d = ctx.d, P = params(+ctx.vals.L); s.clearBalls(); ctx.H = { g: [], h: [] };
        ctx.bg = s.ball(d, { start: START, speed: .5, maxSteps: 300, color: C.opt2, step: p => { const g = d.g(...p); return [p[0] - P.h * g[0], p[1] - P.h * g[1]]; } });
        ctx.bh = s.ball(d, { start: START, speed: .5, maxSteps: 300, step: (p, b) => { const g = d.g(...p), q = b.prev || p; b.prev = p.slice(); return [p[0] - P.a * g[0] + P.b * (p[0] - q[0]), p[1] - P.a * g[1] + P.b * (p[1] - q[1])]; } });
        ctx.bg.onStep = b => ctx.H.g.push([b.steps, Math.max(d.L(...b.p), 1e-12)]); ctx.bh.onStep = b => ctx.H.h.push([b.steps, Math.max(d.L(...b.p), 1e-12)]);
      },
      frame(ctx) {
        const k = +ctx.vals.L / MU, f0 = ctx.d.L(...START);
        ctx.charts.f.clear().fn(t => f0 * gdRho(k) ** (2 * t), { color: HEX.opt2, dash: [3, 4], alpha: .6 }).fn(t => f0 * hbRho(k) ** (2 * t), { color: HEX.ball, dash: [3, 4], alpha: .6 })
          .line(ctx.H.g, { color: HEX.opt2, width: 1.6 }).line(ctx.H.h, { color: HEX.ball, width: 1.8 }).draw();
        const first = H => H.find(h => h[1] < 1e-8)?.[0] ?? '>300';
        ctx.stat(`κ = ${k.toFixed(0)} · per-step contraction: GD ${gdRho(k).toFixed(3)}, heavy ball ${hbRho(k).toFixed(3)} · steps to f < 10⁻⁸: GD ${first(ctx.H.g)}, heavy ball ${first(ctx.H.h)}`);
      },
    },
    {
      id: 'kappa', name: '√κ vs κ', mode: '2d', title: 'Iterations grow like √κ instead of κ',
      caption: 'Steps needed to shrink the error by a factor 10⁶ on a quadratic with condition number κ, from the worst-case contraction factors: (κ − 1)/(κ + 1) for GD with its best step, (√κ − 1)/(√κ + 1) for the optimally tuned heavy ball. The gap grows like √κ. These rates hold for quadratics (and, locally, near strongly convex minima); for general non-convex losses the heavy ball has no such guarantee.',
      note: 'Exact spectral radii (verified in the check script for κ = 10, 25, 100); steps = ln 10⁶ / (−ln ρ).',
      hint: '', legend: [[HEX.opt2, 'GD ~ κ'], [HEX.ball, 'heavy ball ~ √κ']],
      plot: { xlabel: 'κ (log)', ylabel: 'steps to 10⁻⁶ (log)', logx: true, logy: true, xlim: [2, 1e5], ylim: [1, 1e7], legend: false },
      build(ctx) {
        const n = rho => Math.log(1e6) / -Math.log(rho);
        ctx.plot.clear().fn(k => n(gdRho(k)), { color: HEX.opt2, width: 2.2 }).fn(k => n(hbRho(k)), { color: HEX.ball, width: 2.2 }).draw();
        ctx.stat([10, 100, 1e4].map(k => `κ = ${k}: GD ${n(gdRho(k)).toFixed(0)}, heavy ball ${n(hbRho(k)).toFixed(0)}`).join(' · '));
      },
    },
  ],
};
