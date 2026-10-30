// Allen-Zhu 2018 — Natasha 2: swing by saddle points. Each ball step below is one round of the Sec. 4.2 reduction:
// Oja's algorithm (stochastic Hessian-vector products) tests for curvature below −δ; if found, step δ/L₂ along it,
// otherwise minibatch SGD on F_k = f + L·max(0, ‖x − y_k‖ − δ/L₂)². Components, L, L₂ from scripts/check-math/allenzhu2018natasha.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const fL = (x, y) => .25 * (x * x - 1) ** 2 + .5 * y * y;
const fG = (x, y) => [x ** 3 - x, y];
const F = def({ L: fL, g: fG, hess: (x, y) => [3 * x * x - 1, 0, 1], h: v => Math.log1p(v * 3), dom: [-1.6, 1.6, -1.6, 1.6] });
const X0 = [0, 1.3];

const natasha = (D, R) => (p, b) => {
  const n = D.a.length, rad = D.delta / D.L2;
  let w = [R.normal(), R.normal()], nw = Math.hypot(...w); w = [w[0] / nw, w[1] / nw];
  for (let t = 0; t < 30; t++) { const i = Math.floor(R() * n), h0 = 3 * p[0] * p[0] - 1 + D.c[i], h1 = 1 - D.c[i]; w = [w[0] - .08 * h0 * w[0], w[1] - .08 * h1 * w[1]]; nw = Math.hypot(...w); w = [w[0] / nw, w[1] / nw]; }
  b.T += 30;
  const q = (3 * p[0] * p[0] - 1) * w[0] * w[0] + w[1] * w[1];
  if (q <= -D.delta / 2) {
    const u = [p[0] + rad * w[0], p[1] + rad * w[1]], v = [p[0] - rad * w[0], p[1] - rad * w[1]];
    b.nc.push(p.slice()); b.log.push([b.steps + 1, q, 1]); return fL(...u) <= fL(...v) ? u : v;
  }
  let x = p.slice();
  for (let k = 0; k < 12; k++) {
    let gx = 0, gy = 0;
    for (let s = 0; s < 4; s++) { const i = Math.floor(R() * n); gx += x[0] ** 3 - x[0] + D.a[i] + D.c[i] * x[0]; gy += x[1] + D.b[i] - D.c[i] * x[1]; }
    const d = [x[0] - p[0], x[1] - p[1]], nd = Math.hypot(...d), pen = nd > rad ? 2 * D.L * (nd - rad) / nd : 0;
    x = [x[0] - .04 * (gx / 4 + pen * d[0]), x[1] - .04 * (gy / 4 + pen * d[1])]; b.T += 4;
  }
  b.log.push([b.steps + 1, q, 0]); return x;
};
const lmin = p => Math.min(3 * p[0] * p[0] - 1, 1);

export default {
  tabs: [
    {
      id: 'swing', name: 'Swing by', title: 'Use negative curvature when it is there, a safe zone when it is not',
      caption: 'Both balls start at (0, 1.3), on the line that flows straight into the saddle. Gradient descent (violet) slides into the saddle and stops. Coral runs the Natasha 2 loop: each step, Oja’s algorithm looks for curvature below −δ using stochastic Hessian-vector products; if found it steps along it (red dots), otherwise it runs minibatch SGD on f plus a penalty for leaving a ball of radius δ/L₂. It never goes near the saddle.',
      note: 'f = mean of 200 noisy components with saddle (0, 0); δ = 0.5, L₂ = 9.6, safe radius 0.052. Simplified: the paper solves the safe-zone problem with Natasha 1.5 (variance reduced), here with plain minibatch SGD.',
      legend: [[HEX.ball, 'Natasha 2 loop'], [HEX.opt2, 'gradient descent'], [HEX.neg, 'negative-curvature step'], [HEX.good, 'minima (±1, 0)']],
      action: 'Run again', cam: [3.2, 3.6, 3.6], tgt: [0, .2, 0],
      charts: [{ id: 'l', title: 'curvature found by Oja, vᵀ∇²f v', xlabel: 'round', ylabel: 'vᵀHv', ylim: [-1.1, 1.1], h: 200 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .55 });
        s.marker(F, 1, 0, C.good); s.marker(F, -1, 0, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data; s.clearBalls();
        (ctx.dots || []).forEach(m => s.remove(m)); ctx.dots = [];
        ctx.a = s.ball(F, { start: X0, step: natasha(D, rng(Date.now() % 1e6)), speed: .12, maxSteps: 400, stop: b => Math.hypot(...fG(...b.p)) <= D.eps && lmin(b.p) >= -D.delta });
        Object.assign(ctx.a, { T: 0, nc: [], log: [] });
        ctx.g = s.ball(F, { start: X0, step: opt.gd(.1), color: C.opt2, speed: .5, maxSteps: 300 });
      },
      frame(ctx) {
        const a = ctx.a, D = ctx.data; if (!a) return;
        while (ctx.dots.length < a.nc.length) { const p = a.nc[ctx.dots.length]; ctx.dots.push(ctx.stage.dot(F.W(p[0], p[1], .05), C.neg, .035)); }
        ctx.charts.l.clear().set({ xlim: [0, Math.max(20, a.steps)] }).hline(-D.delta / 2, { color: HEX.neg, dash: [4, 4], text: '−δ/2' })
          .line(a.log.map(r => [r[0], r[1]]), { color: HEX.ball, width: 1.4 }).scatter(a.log.filter(r => r[2]).map(r => [r[0], r[1]]), { color: HEX.neg, r: 3 }).draw();
        const gn = Math.hypot(...fG(...a.p));
        ctx.stat(`round ${a.steps}${a.done ? ' (done)' : ''} · ${a.nc.length} negative-curvature steps · ${a.T} stochastic gradients + HVPs · ‖∇f‖ = ${gn.toFixed(3)}, λ_min = ${lmin(a.p).toFixed(2)} · GD at (${ctx.g.p.map(v => v.toFixed(3)).join(', ')})`);
      },
    },
    {
      id: 'rates', name: 'ε^−3.25', mode: '2d', title: 'Stochastic gradients needed for an approximate local minimum',
      caption: 'Theorem 2 (informal): T = Õ(δ⁻⁵ + δ⁻¹ε⁻³) for ‖∇f‖ ≤ ε and ∇²f ⪰ −δI, valid for δ < ε^{1/4}; with δ = ε^{1/4} this is Õ(ε⁻³·²⁵), versus ε⁻⁴ for SGD (Table 1). Move δ to see the trade-off: smaller δ (a stricter Hessian test) costs δ⁻⁵.',
      note: 'Orders only, constants and logs dropped. Grey: other online methods from the paper’s Table 1 (SGD5 and stochastic cubic Newton ε⁻³·⁵, NEON+SCSG ε⁻³·³³³).',
      hint: '', legend: [[HEX.ball, 'Natasha 2, δ = ε^{1/4}'], ['#b28cff', 'Natasha 2, fixed δ'], [HEX.opt3, 'SGD ε⁻⁴'], [HEX.muted, 'other methods']],
      controls: [{ id: 'd', label: 'fixed δ', min: -3, max: -.5, step: .25, value: -1.5, fmt: v => `10^${(+v).toFixed(2)}` }],
      plot: { xlabel: 'accuracy ε', ylabel: 'stochastic gradients T', logx: true, logy: true, xlim: [1e-6, .1], ylim: [1, 1e22], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const dl = 10 ** ctx.vals.d, P = ctx.plot.clear();
        P.fn(e => e ** -3.5, { color: HEX.muted, width: 1.2, dash: [4, 4] }).fn(e => e ** (-10 / 3), { color: HEX.muted, width: 1.2, dash: [2, 3] })
          .fn(e => e ** -4, { color: HEX.opt3, width: 2 })
          .fn(e => (dl < e ** .25 ? dl ** -5 + 1 / (dl * e ** 3) : NaN), { color: '#b28cff', width: 2 })
          .fn(e => { const d = e ** .25; return d ** -5 + 1 / (d * e ** 3); }, { color: HEX.ball, width: 2.6 })
          .vline(dl ** 4, { color: '#b28cff', dash: [3, 3], text: 'δ = ε^{1/4}' }).draw();
        const e = 1e-4, d = e ** .25;
        ctx.stat(`ε = 10⁻⁴: SGD ${(e ** -4).toExponential(1)} · Natasha 2 with δ = ε^{1/4} ${(d ** -5 + 1 / (d * e ** 3)).toExponential(1)} · with δ = 10^${ctx.vals.d.toFixed(2)}: ${dl < d ? (dl ** -5 + 1 / (dl * e ** 3)).toExponential(1) : 'δ ≥ ε^{1/4}, use δ = ε^{1/4}'}`);
      },
    },
  ],
};
