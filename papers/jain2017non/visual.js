// Jain & Kar 2017 — monograph on non-convex optimization for ML; its central tool is projected gradient descent onto
// non-convex sets (Ch. 3, Theorem 3.3). Toy: sparsity set B₀(1) in R² = the two axes. scripts/check-math/jain2017non.py
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';

const WS = [1.2, 0];
const mk = c => def({ L: (x, y) => { const a = x - WS[0], b = y - WS[1]; return .5 * (a * a + 2 * c * a * b + b * b); }, g: (x, y) => { const a = x - WS[0], b = y - WS[1]; return [a + c * b, c * a + b]; }, dom: [-2, 2, -2, 2] });
const proj = z => Math.abs(z[0]) >= Math.abs(z[1]) ? [z[0], 0] : [0, z[1]];

export default {
  tabs: [
    {
      id: 'gpgd', name: 'Projected GD on a sparse set', title: 'Gradient step, then snap back to the nearest sparse point',
      caption: 'Find the best 1-sparse vector for a quadratic loss: the feasible set is the two coordinate axes, a non-convex set. Projected gradient descent takes a gradient step and keeps only the larger coordinate (the exact projection). Theorem 3.3 guarantees convergence to the constrained optimum when the loss is restricted strongly convex and smooth with β/α < 2. With well-conditioned curvature (β/α = 1.86) every ball reaches the optimum (1.2, 0) on the x-axis; with β/α = 19 the ball started on the y-axis gets stuck there, at a point that is not optimal.',
      note: 'f(w) = ½(w − w*)ᵀA(w − w*), A = [[1, c], [c, 1]], w* = (1.2, 0); c = 0.3 (α = 0.7, β = 1.3) or c = 0.9 (α = 0.1, β = 1.9). Step 1/β. White: the feasible set.',
      legend: [[HEX.ball, 'projected GD'], ['#ffffff', 'feasible set B₀(1)'], [HEX.good, 'optimum over the set']],
      action: 'Release balls', cam: [0, 4.4, 4.4], tgt: [0, .2, 0],
      controls: [{ id: 'c', type: 'seg', label: 'conditioning', value: .3, options: [[.3, 'β/α = 1.86'], [.9, 'β/α = 19']] }],
      charts: [{ id: 'f', title: 'f(x_t), ball started at (0, 1.5)', xlabel: 't', ylabel: 'f', logy: true, xlim: [0, 60], ylim: [1e-16, 10], h: 200, legend: false }],
      build(ctx) {
        const c = +ctx.vals.c, d = ctx.d = mk(c), s = ctx.stage; s.surface(d);
        s.tube(d, segPts(-2, 0, 2, 0), C.white, .014); s.tube(d, segPts(0, -2, 0, 2), C.white, .014); s.marker(d, ...WS, C.good);
        const D = ctx.data[c === .3 ? 'good' : 'bad']; ctx.D = D;
        ctx.charts.f.clear().line(D.f.map((v, i) => [i, Math.max(v, 1e-17)]), { color: HEX.ball, width: 2.2, dots: 2 }).draw();
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, be = ctx.D.beta; s.clearBalls();
        ctx.bs = [[0, 1.5], [0, -1.8], [-1.6, 0], [1.9, 0], [0, .6]].map(p => s.ball(ctx.d, { start: p, speed: .25, maxSteps: 60, r: .06, step: (q, b, d) => { const g = d.g(...q); return proj([q[0] - g[0] / be, q[1] - g[1] / be]); } }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const D = ctx.D, ok = ctx.bs.filter(b => Math.hypot(b.p[0] - WS[0], b.p[1] - WS[1]) < 1e-6).length;
        ctx.stat(`α = ${D.alpha.toFixed(2)}, β = ${D.beta.toFixed(2)}, β/α = ${(D.beta / D.alpha).toFixed(2)} · step ${ctx.bs[0].steps} · balls at the optimum: ${ok}/5 · ball from (0, 1.5) ends at (${D.end.map(v => v.toFixed(2)).join(', ')}), f = ${D.f[D.f.length - 1].toExponential(2)}`);
      },
    },
  ],
};
