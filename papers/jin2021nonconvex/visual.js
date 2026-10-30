// Jin, Netrapalli, Ge, Kakade, Jordan 2021 (JACM) — perturbed GD/SGD reach second-order stationary points with only
// polylogarithmic dependence on dimension. Saddle toy + escape time vs d: scripts/check-math/jin2021nonconvex.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => -x * x / 2 + x ** 4 / 4 + y * y / 2, g: (x, y) => [-x + x ** 3, y], dom: [-1.6, 1.6, -1.6, 1.6] });
const ETA = .1, RAD = .01;
/** Algorithm 1 of the JACM paper: x ← x − η(∇f(x) + ξ), ξ ~ N(0, (r²/d)I) at every step (here d = 2) */
const pgd = R => (p, b, d) => { const g = d.g(...p), s = RAD / Math.SQRT2; return [p[0] - ETA * (g[0] + s * R.normal()), p[1] - ETA * (g[1] + s * R.normal())]; };
const gd = (p, b, d) => { const g = d.g(...p); return [p[0] - ETA * g[0], p[1] - ETA * g[1]]; };

export default {
  tabs: [
    {
      id: 'saddle', name: 'Perturbed GD', title: 'A little noise at every step and the saddle cannot hold the iterate',
      caption: 'A strict saddle at the origin with minima at (±1, 0). Plain GD started exactly on the saddle’s stable manifold (white, on the y-axis) slides into the saddle and stays there forever. Perturbed GD (coral) adds a tiny Gaussian vector to every gradient step, as in the JACM version of the algorithm; the unstable direction amplifies it and the ball leaves after a few dozen steps, reaching a second-order stationary point (a minimum).',
      note: 'f = −x²/2 + x⁴/4 + y²/2, step 0.1, noise N(0, (r²/d)I) with r = 0.01, d = 2. All balls start on the y-axis at y = ±1.4, ±0.8.',
      legend: [[HEX.ball, 'perturbed GD'], [HEX.opt3, 'plain GD'], [HEX.good, 'minima (±1, 0)'], [HEX.neg, 'λ < 0 at the saddle']],
      action: 'Release balls', cam: [0, 4.2, 4.6], tgt: [0, .2, 0],
      build(ctx) { const s = ctx.stage; s.surface(F); s.marker(F, 1, 0, C.good); s.marker(F, -1, 0, C.good); s.eigenLines(F, 0, 0, { len: .5 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(12); s.clearBalls();
        ctx.p = [1.4, -1.4, .8, -.8].map(y => s.ball(F, { start: [0, y], step: pgd(R), speed: 1, maxSteps: 600, r: .055 }));
        ctx.g = s.ball(F, { start: [0, 1.2], step: gd, speed: 1, maxSteps: 600, color: C.opt3, r: .055 });
      },
      frame(ctx) {
        if (!ctx.p) return;
        const at = ctx.p.filter(b => Math.abs(Math.abs(b.p[0]) - 1) < .05).length;
        ctx.stat(`step ${ctx.g.steps} · perturbed GD at a minimum: ${at}/4 · plain GD at (${ctx.g.p[0].toFixed(3)}, ${ctx.g.p[1].toExponential(1)}), f = ${ctx.g.loss().toExponential(1)} (a saddle, gradient → 0)`);
      },
    },
    {
      id: 'dim', name: 'Dimension', mode: '2d', title: 'Escape time grows like log d',
      caption: 'Steps perturbed GD needs to leave the saddle of f = −x₁²/2 + x₁⁴/4 + Σᵢ≥₂ xᵢ²/2 in d dimensions, started exactly at the saddle, with the paper’s noise N(0, (r²/d)I): the noise along the single escape direction shrinks like 1/√d, so leaving takes about log(√d)/log(1 + ηγ) more steps. Going from d = 1 to d = 16 384 only doubles the time: dependence on dimension is logarithmic, the paper’s main point.',
      note: 'Offline, 400 runs per d, median with 10–90% band. Because the coordinates decouple, the escape coordinate is simulated exactly; step 0.1, r = 0.01, escape = |x₁| > 0.5.',
      hint: '', legend: [[HEX.ball, 'median escape step'], [HEX.muted, '10–90% of runs'], [HEX.good, 'slope 1/(2 log(1 + ηγ))']],
      plot: { xlabel: 'dimension d', ylabel: 'steps to escape', logx: true, xlim: [.8, 2e4], ylim: [40, 140], legend: false },
      build(ctx) {
        const D = ctx.data, M = D.med, P = ctx.plot.clear(), th = 1 / (2 * Math.log(1 + ETA));
        P.line(M.map(q => [q[0], q[2]]), { color: HEX.muted, dash: [3, 3] }).line(M.map(q => [q[0], q[3]]), { color: HEX.muted, dash: [3, 3] })
          .fn(d => M[0][1] + th * Math.log(d), { color: HEX.good, dash: [5, 4] }).line(M.map(q => [q[0], q[1]]), { color: HEX.ball, width: 2.4, dots: 4 }).draw();
        ctx.stat(`median escape steps: ${M.map(q => `d = ${q[0]}: ${q[1]}`).join(', ')} · fitted slope ${D.slope.toFixed(2)} per unit log d (theory ${th.toFixed(2)})`);
      },
    },
  ],
};
