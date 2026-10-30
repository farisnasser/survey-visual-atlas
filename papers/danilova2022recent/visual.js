// Danilova et al. 2022 — survey of non-convex optimization theory; Sec. 5.1: under the Polyak–Łojasiewicz condition every
// stationary point is global and GD converges linearly. Toy f = x² + 3 sin²x + y²: scripts/check-math/danilova2022recent.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => x * x + 3 * Math.sin(x) ** 2 + y * y, g: (x, y) => [2 * x + 3 * Math.sin(2 * x), 2 * y], dom: [-5, 5, -5, 5] });
const L = 8;

export default {
  tabs: [
    {
      id: 'pl', name: 'PL condition', title: 'Non-convex, yet gradient descent always converges fast to the global minimum',
      caption: 'f = x² + 3 sin²x + y² is not convex (its curvature along x dips to −4, visible as the ripples) but it satisfies the Polyak–Łojasiewicz inequality: wherever the loss is above its minimum, the gradient is proportionally large. So the only stationary point is the global minimum, and GD with step 1/L closes the gap f − f* at least by the factor e^{−μ/L} every step. Right: the worst of 200 runs against that bound.',
      note: 'L = 8; μ = 0.1755 is the smallest ratio ‖∇f‖²/(2(f − f*)) found numerically on [−6, 6] (Karimi et al. give the safe value 1/32). 12 balls from random starts.',
      legend: [[HEX.ball, 'GD, step 1/L'], [HEX.good, 'global minimum (0, 0)'], [HEX.muted, 'bound e^{−μk/L}(f(x₀) − f*)']],
      action: 'Release balls', cam: [0, 4.4, 4.6], tgt: [0, .2, 0],
      charts: [{ id: 'gap', title: 'max f(x_k) − f* over 200 runs', xlabel: 'k', ylabel: 'f − f*', logy: true, xlim: [0, 400], ylim: [1e-30, 100], h: 240, legend: false }],
      build(ctx) {
        const s = ctx.stage, D = ctx.data; s.surface(F); s.marker(F, 0, 0, C.good);
        ctx.charts.gap.clear().fn(k => D.f0max * Math.exp(-D.mu * k / L), { color: HEX.muted, dash: [5, 4] }).line(D.gaps.map(q => [q[0], Math.max(q[1], 1e-31)]), { color: HEX.ball, width: 2.2 }).draw();
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(4); s.clearBalls();
        ctx.bs = Array.from({ length: 12 }, () => s.ball(F, { start: [R.uniform(-4.8, 4.8), R.uniform(-4.8, 4.8)], step: (p, b, d) => { const g = d.g(...p); return [p[0] - g[0] / L, p[1] - g[1] / L]; }, speed: .5, maxSteps: 200, r: .06 }));
      },
      frame(ctx) {
        if (!ctx.bs) return;
        const D = ctx.data, worst = Math.max(...ctx.bs.map(b => b.loss())), k = ctx.bs[0].steps;
        ctx.stat(`step ${k} · worst f − f* of these balls ${worst.toExponential(2)} · offline worst of 200 runs: ${D.gaps[10][1].toExponential(1)} at k = 50, ${D.gaps[20][1].toExponential(1)} at k = 100 · μ/L = ${(D.mu / L).toFixed(4)}`);
      },
    },
  ],
};
