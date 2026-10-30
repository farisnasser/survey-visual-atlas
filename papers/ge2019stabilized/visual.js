// Ge, Li, Wang, Wang 2019 — Stabilized SVRG: SVRG on the shifted function f̂(x) = f(x) − ⟨∇f(x̃), x − x̃⟩ during a super epoch.
// Components (with large component Hessian-Lipschitz ρ′) and parameters from scripts/check-math/ge2019stabilized.py (data.json).
import { def, C, HEX, circlePts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const G0 = .08;
const fL = (x, y) => G0 * x + x * x - .5 * y * y + .25 * y ** 4;
const fG = (x, y) => [G0 + 2 * x, -y + y ** 3];
const F = def({ L: fL, g: fG, hess: (x, y) => [2, 0, -1 + 3 * y * y], h: v => Math.log1p(Math.max(v + .26, 0) * 4), dom: [-.3, .3, -1.3, 1.3], square: true });
const gi = (D, p, i) => [G0 + 2 * p[0] + D.a[i] + D.c[i] * p[0] + D.d[i] * p[0] * p[0], -p[1] + p[1] ** 3 + D.b[i] - D.c[i] * p[1]];

/** One SVRG iteration (Alg. 2 line 8 / Alg. 3 line 9); snapshot every m steps; stabilized runs subtract v_shift until ‖x − x̃‖ > dist. */
const svrg = (D, R, stab) => (p, b) => {
  const n = D.a.length;
  b.t = (b.t ?? -1) + 1;
  if (b.t === 0) b.shift = stab ? fG(0, 0) : [0, 0];
  if (b.t % D.m === 0) { b.snap = p.slice(); b.gs = fG(...p); }
  let vx = 0, vy = 0;
  for (let k = 0; k < D.bs; k++) { const i = Math.floor(R() * n), u = gi(D, p, i), w = gi(D, b.snap, i); vx += u[0] - w[0]; vy += u[1] - w[1]; }
  const q = [p[0] - D.eta * (vx / D.bs + b.gs[0] - b.shift[0]), p[1] - D.eta * (vy / D.bs + b.gs[1] - b.shift[1])];
  if (stab && b.shift[0] !== 0 && Math.hypot(...q) > D.dist) { b.shift = [0, 0]; b.endT = b.t + 1; }
  return q;
};

export default {
  tabs: [
    {
      id: 'stabilize', name: 'Shift the gradient', title: 'Subtracting the snapshot gradient keeps the first phase local, then curvature takes over',
      caption: 'At x̃ = 0 the gradient is small (0.08, below the threshold G) and the Hessian is diag(2, −1). Both balls are perturbed by the same small ξ (radius δ = 0.02) and run minibatch SVRG. Violet (Perturbed SVRG) first drifts along the leftover gradient; coral (Stabilized SVRG) runs on f̂ = f − ⟨∇f(x̃), x − x̃⟩, whose gradient at x̃ is exactly zero, so it stays within O(δ) while the negative-curvature direction grows (Phase 1), then escapes and drops the shift.',
      note: 'n = 125 components with zero-mean cubic terms d_i x³/3, so each f_i has Hessian-Lipschitz ρ′ ≈ 25 vs ρ = 9 for f; b = n^{2/3} = 25, m = 5, η = 0.1. The x-axis is stretched ×4.3 for display.',
      legend: [[HEX.ball, 'Stabilized SVRG'], [HEX.opt2, 'Perturbed SVRG'], [HEX.flat, 'perturbation ball δ'], [HEX.good, 'minima']],
      action: 'Run again', cam: [3.6, 3.0, 3.2], tgt: [0, .2, 0],
      charts: [{ id: 'd', title: '‖x_t − x̃‖', xlabel: 'iteration t', ylabel: 'distance', logy: true, ylim: [3e-3, 2], h: 220 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .5, labels: false });
        s.line(F, circlePts(0, 0, ctx.data.delta), C.flat, 1, .03);
        s.marker(F, -G0 / 2, 1, C.good); s.marker(F, -G0 / 2, -1, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, seed = Date.now() % 1e6, R0 = rng(seed);
        const th = 2 * Math.PI * R0(), rr = D.delta * Math.sqrt(R0()), x0 = [rr * Math.cos(th), rr * Math.sin(th)];
        s.clearBalls();
        ctx.b = [s.ball(F, { start: x0, step: svrg(D, rng(seed + 1), true), speed: .35, maxSteps: 120 }),
          s.ball(F, { start: x0, step: svrg(D, rng(seed + 1), false), color: C.opt2, speed: .35, maxSteps: 120 })];
        for (const q of ctx.b) { q.h = [[0, Math.max(Math.hypot(...q.p), 3e-3)]]; q.onStep = z => z.h.push([z.steps, Math.max(Math.hypot(...z.p), 3e-3)]); }
      },
      frame(ctx) {
        const [a, b] = ctx.b || []; if (!a) return;
        const D = ctx.data;
        ctx.charts.d.clear().set({ xlim: [0, 120] }).hline(D.delta, { color: HEX.flat, dash: [4, 4], text: 'δ' }).vline(10, { color: HEX.muted, dash: [3, 3], text: '1/(ηγ)' })
          .line(b.h, { color: HEX.opt2, width: 1.5, label: 'perturbed' }).line(a.h, { color: HEX.ball, width: 2, label: 'stabilized' }).draw();
        const at10 = z => z.h[10] ? z.h[10][1].toFixed(3) : '…';
        ctx.stat(`t = ${a.steps} · ‖x − x̃‖ at t = 10: stabilized ${at10(a)}, perturbed ${at10(b)} · x-drift now: stabilized ${a.p[0].toFixed(3)}, perturbed ${b.p[0].toFixed(3)}${a.endT ? ` · shift dropped at t = ${a.endT}` : ''}`);
      },
    },
    {
      id: 'rates', name: 'Complexity', mode: '2d', title: 'Second-order guarantee at almost the first-order SVRG cost',
      caption: 'Stochastic gradients to an ε-second-order stationary point for a finite sum of n functions (Table 1, orders only). Stabilized SVRG costs Õ(n^{2/3}/ε² + n/ε^{1.5}); its main term equals plain SVRG’s first-order cost, and unlike Neon2+SVRG it needs no negative-curvature subroutine.',
      note: 'Constants and logs dropped. SPIDER-SFO+ is asymptotically cheaper but is not a simple SVRG variant.',
      hint: '', legend: [[HEX.ball, 'Stabilized SVRG'], [HEX.opt3, 'SVRG (first-order only)'], [HEX.opt2, 'Neon2 + SVRG'], [HEX.muted, 'SPIDER-SFO+']],
      controls: [{ id: 'e', label: 'accuracy ε', min: -4, max: -1, step: .5, value: -2, fmt: v => `10^${(+v).toFixed(1)}` }],
      plot: { xlabel: 'number of components n', ylabel: 'stochastic gradients', logx: true, logy: true, xlim: [10, 1e9], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const e = 10 ** ctx.vals.e;
        const rows = [[n => n ** (2 / 3) / e ** 2 + n, HEX.opt3, 1.8], [n => n ** (2 / 3) / e ** 2 + n / e ** 1.5 + n ** .75 / e ** 1.75, HEX.opt2, 1.8],
          [n => Math.sqrt(n) / e ** 2 + e ** -2.5, HEX.muted, 1.4], [n => n ** (2 / 3) / e ** 2 + n / e ** 1.5, HEX.ball, 2.6]];
        const P = ctx.plot.clear(); rows.forEach(([fn, c, w]) => P.fn(fn, { color: c, width: w })); P.draw();
        const n = 1e6, v = rows.map(([fn]) => fn(n).toExponential(1));
        ctx.stat(`n = 10⁶, ε = 10^${ctx.vals.e}: SVRG ${v[0]} · Neon2+SVRG ${v[1]} · SPIDER-SFO+ ${v[2]} · Stabilized SVRG ${v[3]}`);
      },
    },
  ],
};
