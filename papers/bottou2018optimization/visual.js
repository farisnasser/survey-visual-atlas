// Bottou, Curtis, Nocedal 2018 — stochastic gradient on a nonconvex objective: a fixed step only drives the average squared
// gradient down to a noise floor ᾱLM/μ (Theorem 4.8); diminishing steps drive it to zero (Theorem 4.10). scripts/check-math/bottou2018optimization.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => (x * x + y * y) / 4 + Math.sin(x) * Math.sin(y), g: (x, y) => [x / 2 + Math.cos(x) * Math.sin(y), y / 2 + Math.sin(x) * Math.cos(y)], dom: [-4, 4, -4, 4] });
const W1 = [2.5, -2], SIG = 1;
const sg = (a, R, dim) => (p, b, d) => { const g = d.g(p[0], p[1]), al = dim ? .5 / (1 + b.steps / 50) : a; return [p[0] - al * (g[0] + SIG * R.normal()), p[1] - al * (g[1] + SIG * R.normal())]; };

export default {
  tabs: [
    {
      id: 'sg', name: 'Fixed vs diminishing step', title: 'With a fixed step, SG keeps jittering; with a shrinking step it settles',
      caption: 'A smooth nonconvex objective with several minima, and stochastic gradients with Gaussian noise. Coral balls use a fixed step ᾱ: they reach a basin but keep jittering, and the average squared gradient levels off at a floor proportional to ᾱ (Theorem 4.8). Violet balls use the diminishing step 0.5/(1 + k/50): their jitter dies out and the gradient goes to zero (Theorem 4.10). Right: the averages over 2000 runs against the theorem’s bound.',
      note: 'F = (x² + y²)/4 + sin x sin y (L = 1.5, F ≥ −1), g = ∇F + N(0, I), so μ = μ_G = 1 and M = 2. Start w₁ = (2.5, −2). Right: dashed = bound (4.28b) ᾱLM/μ + 2(F(w₁) − F_inf)/(Kμᾱ).',
      legend: [[HEX.ball, 'SG, fixed step ᾱ'], [HEX.opt2, 'SG, step 0.5/(1 + k/50)']],
      action: 'Release balls', cam: [0, 4.6, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'a', type: 'seg', label: 'fixed step ᾱ', value: .2, options: [[.05, '0.05'], [.2, '0.2'], [.5, '0.5']] }],
      charts: [{ id: 'avg', title: 'E[(1/K) Σ‖∇F(w_k)‖²] over 2000 runs', xlabel: 'K', ylabel: 'avg ‖∇F‖²', logx: true, logy: true, xlim: [1, 3000], ylim: [.005, 20], h: 240, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.marker(F, ...W1, C.white, .08);
        const D = ctx.data.curves, a = String(+ctx.vals.a), P = ctx.charts.avg.clear();
        P.line(D[a].map(q => [q[0], q[2]]), { color: HEX.ball, dash: [5, 4], width: 1.4 }).line(D[a].map(q => [q[0], q[1]]), { color: HEX.ball, width: 2.2 })
          .line(D.dim.map(q => [q[0], q[1]]), { color: HEX.opt2, width: 2.2 }).line(D.dim.map(q => [q[0], q[2]]), { color: HEX.opt2, width: 1.2, dash: [2, 3] }).draw();
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, a = +ctx.vals.a, R = rng(2); s.clearBalls();
        ctx.fx = Array.from({ length: 6 }, () => s.ball(F, { start: W1, step: sg(a, R, false), speed: 2, maxSteps: 3000, r: .05 }));
        ctx.dm = Array.from({ length: 6 }, () => s.ball(F, { start: W1, step: sg(a, R, true), speed: 2, maxSteps: 3000, r: .05, color: C.opt2 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.fx) return;
        const gn = bs => bs.reduce((t, b) => t + F.g(...b.p).reduce((u, v) => u + v * v, 0), 0) / bs.length, D = ctx.data.curves, a = String(+ctx.vals.a), last = D[a][D[a].length - 1], dl = D.dim[D.dim.length - 1];
        ctx.stat(`step ${ctx.fx[0].steps} · current mean ‖∇F‖²: fixed ᾱ = ${a} ${gn(ctx.fx).toFixed(3)}, diminishing ${gn(ctx.dm).toFixed(4)} · offline at K = 3000: fixed running average ${last[1]} (bound ${last[2]}, floor ᾱLM/μ = ${(+a * 3).toFixed(2)}), diminishing last iterate ${dl[2]}`);
      },
    },
  ],
};
