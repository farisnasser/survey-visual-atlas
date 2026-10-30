// Wang, Zhou, Liang, Lan 2020 — cubic regularization with momentum (CRm, Alg. 1): cubic step, adaptive momentum
// β = min{β, ‖∇f(y)‖, ‖y − x‖}, then a monotone choice. Checked in scripts/check-math/wang2020cubic.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { eig2 } from '../../src/lib/math.js';

/** Exact global minimizer of gᵀs + ½sᵀHs + (M/6)‖s‖³ in 2D (secular equation; hard case included). */
function cubicStep(g, H, M) {
  const ev = eig2(...H), [[l1, v1], [l2, v2]] = ev[0][0] <= ev[1][0] ? ev : [ev[1], ev[0]];
  const gb = [g[0] * v1[0] + g[1] * v1[1], g[0] * v2[0] + g[1] * v2[1]], lo0 = Math.max(0, -2 * l1 / M) + 1e-14;
  const nr = r => Math.hypot(gb[0] / (l1 + M * r / 2), gb[1] / (l2 + M * r / 2));
  let d;
  if (nr(lo0) < lo0) { const a2 = -gb[1] / (l2 + M * lo0 / 2); d = [Math.sqrt(Math.max(lo0 * lo0 - a2 * a2, 0)), a2]; }
  else { let lo = lo0, hi = lo0 + 1; while (nr(hi) > hi) hi *= 2; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (nr(m) > m) lo = m; else hi = m; } d = [-gb[0] / (l1 + M * hi / 2), -gb[1] / (l2 + M * hi / 2)]; }
  return [d[0] * v1[0] + d[1] * v2[0], d[0] * v1[1] + d[1] * v2[1]];
}
const cr = M => (p, b, d) => { const s = cubicStep(d.g(...p), d.hess(...p), M); return [p[0] + s[0], p[1] + s[1]]; };
const crm = (M, beta) => (p, b, d) => {
  const s = cubicStep(d.g(...p), d.hess(...p), M), y = [p[0] + s[0], p[1] + s[1]];
  b.yp ??= p.slice();
  const bk = Math.min(beta, Math.hypot(...d.g(...y)), Math.hypot(s[0], s[1])), v = [y[0] + bk * (y[0] - b.yp[0]), y[1] + bk * (y[1] - b.yp[1])];
  b.yp = y; const useV = d.L(...v) < d.L(...y); b.log.push([b.steps + 1, bk, useV ? 1 : 0]);
  return useV ? v : y;
};
const ROS = def({ L: (x, y) => .5 * (1 - x) ** 2 + 5 * (y - x * x) ** 2, g: (x, y) => [-(1 - x) - 20 * x * (y - x * x), 10 * (y - x * x)],
  hess: (x, y) => [1 - 20 * y + 60 * x * x, -20 * x, 10], h: v => Math.log1p(v * 4), dom: [-1.3, 1.5, -.6, 2.1] });
const SAD = def({ L: (x, y) => .5 * x * x - .5 * y * y + .25 * y ** 4, g: (x, y) => [x, -y + y ** 3], hess: (x, y) => [1, 0, -1 + 3 * y * y], h: v => v, dom: [-1.4, 1.4, -1.5, 1.5] });

export default {
  tabs: [
    {
      id: 'race', name: 'CR vs CRm', title: 'Momentum on top of cubic Newton steps, kept safe by a monotone choice',
      caption: 'Both balls start where the Hessian is indefinite on f = ½(1 − x)² + 5(y − x²)². White: cubic regularization (CR) takes the exact cubic step every iteration. Coral: CRm also forms v = y + β_k(y − y_prev) with β_k = min{β, ‖∇f(y)‖, ‖y − x‖} and keeps whichever of y, v is lower, so f never goes up.',
      note: 'β = 0.9; exact cubic subproblem solves. M = 180 exceeds the Hessian-Lipschitz constant on |x| ≤ 1.5 (Alg. 1 needs M > L₂); with M = 60 CR is not guaranteed monotone here. Speed-up on this instance is modest (the paper reports larger gains on its benchmarks).',
      legend: [[HEX.opt3, 'CR'], [HEX.ball, 'CRm'], [HEX.good, 'minimum (1, 1)']],
      action: 'Run again', cam: [-.6, 3.8, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'M', type: 'seg', label: 'cubic weight M', value: 180, options: [[60, '60'], [180, '180 (> L₂)']] }],
      charts: [{ id: 'f', title: 'f(x_k) − f*', xlabel: 'iteration k', ylabel: 'gap', logy: true, ylim: [1e-14, 30], h: 190 },
        { id: 'b', title: 'β_k (dot = momentum candidate accepted)', xlabel: 'iteration k', ylabel: 'β_k', logy: true, ylim: [1e-8, 1], h: 170, legend: false }],
      build(ctx) { const s = ctx.stage; s.surface(ROS); s.marker(ROS, 1, 1, C.good); s.eigenLines(ROS, -.4, 1.4, { len: .4 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, M = +ctx.vals.M; s.clearBalls();
        const stop = b => Math.hypot(...ROS.g(...b.p)) < 1e-9;
        ctx.a = s.ball(ROS, { start: [-.4, 1.4], step: cr(M), color: C.opt3, speed: .12, maxSteps: 200, stop });
        ctx.b = s.ball(ROS, { start: [-.4, 1.4], step: crm(M, .9), speed: .12, maxSteps: 200, stop }); ctx.b.log = [];
        for (const q of [ctx.a, ctx.b]) { q.h = [[0, q.loss()]]; q.onStep = z => z.h.push([z.steps, Math.max(z.loss(), 1e-14)]); }
      },
      onControl(ctx) { this.drop(ctx); },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        const xm = Math.max(10, a.steps, b.steps);
        ctx.charts.f.clear().set({ xlim: [0, xm] }).line(a.h, { color: HEX.opt3, width: 1.8, label: 'CR' }).line(b.h, { color: HEX.ball, width: 1.8, label: 'CRm' }).draw();
        ctx.charts.b.clear().set({ xlim: [0, xm] }).line(b.log.map(r => [r[0], Math.max(r[1], 1e-8)]), { color: HEX.ball, width: 1.4 }).scatter(b.log.filter(r => r[2]).map(r => [r[0], Math.max(r[1], 1e-8)]), { color: HEX.ball, r: 3 }).draw();
        const ups = h => h.filter((r, i) => i && r[1] > h[i - 1][1] * (1 + 1e-12)).length;
        ctx.stat(`CR: ${a.steps} iterations${a.done ? '' : '…'}, f went up ${ups(a.h)}× · CRm: ${b.steps}${b.done ? '' : '…'}, momentum accepted ${b.log.filter(r => r[2]).length}/${b.log.length}, f went up ${ups(b.h)}×`);
      },
    },
    {
      id: 'saddle', name: 'Zero gradient', title: 'At an exact saddle the cubic step still moves',
      caption: 'Both balls sit exactly on the saddle of f = x²/2 − y²/2 + y⁴/4, where ∇f = 0. Gradient descent (white) never moves. The cubic model ½sᵀ∇²f s + (M/6)‖s‖³ is minimized off the saddle, along the negative-curvature direction, at distance 2|λ_min|/M, so CR (coral) leaves at once and lands in a minimum.',
      note: 'M = 2, so the first cubic step has length 2·1/2 = 1 and lands exactly on the minimum (0, ±1). This is what “second-order” buys: no noise is needed.',
      legend: [[HEX.opt3, 'gradient descent'], [HEX.ball, 'cubic regularization'], [HEX.good, 'minima (0, ±1)']],
      action: 'Run again', cam: [3.2, 3.2, 4], tgt: [0, .3, 0],
      build(ctx) { const s = ctx.stage; s.surface(SAD); s.eigenLines(SAD, 0, 0, { len: .6 }); s.marker(SAD, 0, 1, C.good); s.marker(SAD, 0, -1, C.good); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.a = s.ball(SAD, { start: [0, 0], step: opt.gd(.1), color: C.opt3, speed: .2, maxSteps: 60 });
        ctx.b = s.ball(SAD, { start: [0, 0], step: cr(2), speed: .05, maxSteps: 12 });
      },
      frame(ctx) { const { a, b } = ctx; if (a) ctx.stat(`GD: ${a.steps} steps, still at (${a.p.map(v => v.toFixed(2)).join(', ')}) · CR: ${b.steps} steps, at (${b.p.map(v => v.toFixed(3)).join(', ')}), f = ${b.loss().toFixed(4)}`); },
    },
  ],
};
