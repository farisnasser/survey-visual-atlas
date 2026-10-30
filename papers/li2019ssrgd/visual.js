// Li 2019 — SSRGD: SARAH/SPIDER recursive gradients plus a uniform perturbation when the full gradient is small.
// Components, η = (√5 − 1)/(2L), m = b = √n and thresholds from scripts/check-math/li2019ssrgd.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const fL = (x, y) => .25 * (x * x - 1) ** 2 + .5 * y * y;
const fG = (x, y) => [x ** 3 - x, y];
const F = def({ L: fL, g: fG, hess: (x, y) => [3 * x * x - 1, 0, 1], h: v => Math.log1p(v * 3), dom: [-1.6, 1.6, -1.6, 1.6] });
const X0 = [0, 1.3];
const gi = (D, p, i) => [p[0] ** 3 - p[0] + D.a[i] + D.c[i] * p[0], p[1] + D.b[i] - D.c[i] * p[1]];

/** Algorithm 1, one inner step per call. Epoch start: full gradient (n calls), optional perturbation; then x ← x − ηv, v ← v + mean_b(∇f_i(x) − ∇f_i(x_prev)). */
const ssrgd = (D, R, perturb) => (p, b) => {
  const n = D.a.length; b.t = (b.t ?? -1) + 1; let x = p;
  if (b.t % D.m === 0) {
    b.sfo += n;
    if (perturb && !b.inside && Math.hypot(...fG(...x)) <= D.gthres) {
      const th = 2 * Math.PI * R(), rr = D.r * Math.sqrt(R()); x = [x[0] + rr * Math.cos(th), x[1] + rr * Math.sin(th)];
      b.inside = true; b.xs = x.slice(); b.ts = b.t; b.kicks.push(x.slice());
    }
    b.v = fG(...x);
  }
  const q = [x[0] - D.eta * b.v[0], x[1] - D.eta * b.v[1]];
  let dx = 0, dy = 0;
  for (let k = 0; k < D.m; k++) { const i = Math.floor(R() * n), u = gi(D, q, i), w = gi(D, x, i); dx += u[0] - w[0]; dy += u[1] - w[1]; }
  b.v = [b.v[0] + dx / D.m, b.v[1] + dy / D.m]; b.sfo += 2 * D.m;
  if (b.inside && (Math.hypot(q[0] - b.xs[0], q[1] - b.xs[1]) > D.Rstop || b.t - b.ts > D.tthres)) b.inside = false;
  return q;
};
const done = b => Math.hypot(...fG(...b.p)) < .02 && Math.abs(Math.abs(b.p[0]) - 1) < .05;

export default {
  tabs: [
    {
      id: 'escape', name: 'Perturb once', title: 'A recursive gradient estimator plus an occasional kick',
      caption: 'Both balls start at (0, 1.3), on the line x = 0 that flows straight into the saddle, and use the same SARAH/SPIDER estimator (full gradient every m = √n steps, same-minibatch differences in between). On this line every difference has x-part exactly 0, so without help (violet) the iterates converge to the saddle. SSRGD (coral) adds one rule: when the full gradient is below g_thres, jump to a uniform point in a ball of radius r (amber ring) and start a super epoch. It escapes.',
      note: 'n = 196 noisy components; η = (√5 − 1)/(2L) with L = 8.3, m = b = 14, g_thres = 0.02, r = 0.01. Super-epoch stop simplified to “moved 0.3 or 400 steps”; the paper also tracks the function decrease.',
      legend: [[HEX.ball, 'SSRGD'], [HEX.opt2, 'SARAH/SPIDER, no perturbation'], [HEX.flat, 'perturbation'], [HEX.good, 'minima (±1, 0)']],
      action: 'Run again', cam: [3.2, 3.6, 3.6], tgt: [0, .2, 0],
      charts: [{ id: 'x', title: '|x_t| (escape coordinate)', xlabel: 'stochastic gradient calls', ylabel: '|x|', logy: true, ylim: [1e-6, 2], h: 220 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .55 }); s.marker(F, 1, 0, C.good); s.marker(F, -1, 0, C.good); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, seed = Date.now() % 1e6; s.clearBalls(); (ctx.rings || []).forEach(o => s.remove(o)); ctx.rings = [];
        ctx.a = s.ball(F, { start: X0, step: ssrgd(D, rng(seed), true), speed: 1, maxSteps: 2500, stop: done });
        ctx.c = s.ball(F, { start: X0, step: ssrgd(D, rng(seed), false), color: C.opt2, speed: 1, maxSteps: 2500 });
        for (const q of [ctx.a, ctx.c]) { Object.assign(q, { sfo: 0, kicks: [], h: [] }); q.onStep = z => z.h.push([z.sfo, Math.max(Math.abs(z.p[0]), 1e-6)]); }
      },
      frame(ctx) {
        const { a, c } = ctx; if (!a) return;
        while (ctx.rings.length < a.kicks.length) { const k = a.kicks[ctx.rings.length]; ctx.rings.push(ctx.stage.marker(F, k[0], k[1], C.flat, .07)); }
        ctx.charts.x.clear().set({ xlim: [0, Math.max(4000, a.sfo)] }).line(c.h.filter((_, i) => i % 3 === 0), { color: HEX.opt2, width: 1.4, label: 'no perturbation' })
          .line(a.h, { color: HEX.ball, width: 1.8, label: 'SSRGD' }).draw();
        ctx.stat(`SSRGD: ${a.sfo} stochastic gradients, ${a.kicks.length} perturbation${a.kicks.length === 1 ? '' : 's'}, at (${a.p.map(v => v.toFixed(3)).join(', ')})${a.done ? ' = minimum' : ''} · no perturbation: x = ${c.p[0].toFixed(3)}, y = ${c.p[1].toExponential(1)}`);
      },
    },
    {
      id: 'rates', name: '√n vs n^{2/3}', mode: '2d', title: 'Near-optimal dependence on the number of components',
      caption: 'Stochastic gradients for a finite sum of n components (orders, Theorems 1–2). First order: SSRGD needs O(n + √n/ε²), matching Fang et al.’s lower bound √n/ε², where perturbed or stabilized SVRG pays n^{2/3}/ε². Second order (δ = √ε): SSRGD Õ(√n/ε² + √n/δ⁴ + n/δ³).',
      note: 'Orders only (L, ρ, Δf = 1, logs dropped). With δ = √ε the second-order terms are √n/ε² + n/ε^{1.5}.',
      hint: '', legend: [[HEX.ball, 'SSRGD (second-order)'], ['#ff9b80', 'SSRGD (first-order)'], [HEX.opt2, 'Stabilized SVRG (second-order)'], [HEX.muted, 'lower bound √n/ε²']],
      controls: [{ id: 'e', label: 'accuracy ε', min: -4, max: -1, step: .5, value: -2, fmt: v => `10^${(+v).toFixed(1)}` }],
      plot: { xlabel: 'number of components n', ylabel: 'stochastic gradients', logx: true, logy: true, xlim: [10, 1e10], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const e = 10 ** ctx.vals.e, d = Math.sqrt(e);
        const rows = [[n => Math.sqrt(n) / e ** 2, HEX.muted, 1.4, [5, 4]], [n => n + Math.sqrt(n) / e ** 2, '#ff9b80', 1.8], [n => n ** (2 / 3) / e ** 2 + n / e ** 1.5, HEX.opt2, 1.8],
          [n => Math.sqrt(n) / e ** 2 + Math.sqrt(n) / d ** 4 + n / d ** 3, HEX.ball, 2.6]];
        const P = ctx.plot.clear(); rows.forEach(([fn, c, w, dash]) => P.fn(fn, { color: c, width: w, dash })); P.draw();
        const n = 1e8, v = rows.map(([fn]) => fn(n).toExponential(1));
        ctx.stat(`n = 10⁸, ε = 10^${ctx.vals.e}: lower bound ${v[0]} · SSRGD first-order ${v[1]} · Stabilized SVRG ${v[2]} · SSRGD second-order ${v[3]}`);
      },
    },
  ],
};
