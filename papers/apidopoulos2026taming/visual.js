// Apidopoulos, Lytras, Mertikopoulos 2026 — tamed SGD converges a.s. and avoids strict saddles under (L0, L1)-smoothness.
// Toy f = x⁴/4 − x²/2 + y²/2 + y⁴/4 (cubic gradient growth, strict saddle at 0). Offline sweep: scripts/check-math/apidopoulos2026taming.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => x ** 4 / 4 - x * x / 2 + y * y / 2 + y ** 4 / 4, g: (x, y) => [x ** 3 - x, y + y ** 3], dom: [-3, 3, -3, 3] });
const P = .8, Q = .5, T0 = 10, SIG = .3;
/** One stochastic step at iteration t: ĝ = ∇f + N(0, σ²I); 'sgd' plain, 'tsgd' x − γₜ/(1 + λₜ r(x)) ĝ with r(x) = 1 + ‖x‖³. */
const step = (kind, g0, R) => (p, b, d) => {
  const t = b.steps, gam = g0 / ((t + T0) / T0) ** P, lam = 1 / ((t + T0) / T0) ** Q, gr = d.g(p[0], p[1]);
  const gh = [gr[0] + SIG * R.normal(), gr[1] + SIG * R.normal()], s = kind === 'tsgd' ? gam / (1 + lam * (1 + Math.hypot(p[0], p[1]) ** 3)) : gam;
  return [p[0] - s * gh[0], p[1] - s * gh[1]];
};
const START = Array.from({ length: 6 }, (_, k) => [2.8 * Math.cos(.3 + k * Math.PI / 3), 2.8 * Math.sin(.3 + k * Math.PI / 3)]);

export default {
  tabs: [
    {
      id: 'taming', name: 'Tamed vs plain SGD', title: 'Gradients that grow like ‖x‖³ blow plain SGD away; taming keeps it on the surface',
      caption: 'A quartic loss: strict saddle at the origin, minima at (±1, 0), and gradients that grow like ‖x‖³, so there is no global Lipschitz constant (the loss is (L₀, L₁)-smooth instead). From the same six far starts and the same step schedule, plain SGD (violet) overshoots and leaves the picture once γ₀ is large; tamed SGD (coral) divides the step by 1 + λₜr(x) and settles at a minimum. The white ball starts exactly on the saddle’s stable manifold (x = 0): noise pushes it off and it too ends at a minimum.',
      note: 'γₜ = γ₀(1 + t/10)^−0.8, λₜ = (1 + t/10)^−0.5 (inside the paper’s region 25a), r(x) = 1 + ‖x‖³, gradient noise N(0, 0.3²I). Balls that leave the box stop at its edge.',
      legend: [[HEX.ball, 'tamed SGD (TSGD)'], [HEX.opt2, 'plain SGD'], [HEX.opt3, 'TSGD from the stable manifold'], [HEX.good, 'minima (±1, 0)'], [HEX.neg, 'λ < 0 at the saddle']],
      action: 'Release balls', cam: [0, 5, 4.4], tgt: [0, .2, 0],
      controls: [{ id: 'g0', type: 'seg', label: 'initial step γ₀', value: 1, options: [[.1, '0.1'], [.3, '0.3'], [1, '1'], [3, '3']] }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.marker(F, 1, 0, C.good); s.marker(F, -1, 0, C.good); s.eigenLines(F, 0, 0, { len: .6 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, g0 = +ctx.vals.g0, R1 = rng(1), R2 = rng(2), R3 = rng(3); s.clearBalls();
        const out = b => !F.in(b.p[0], b.p[1]) || Math.abs(b.p[0]) >= 2.999 || Math.abs(b.p[1]) >= 2.999;
        ctx.tb = START.map(p => s.ball(F, { start: p, step: step('tsgd', g0, R1), speed: 4, maxSteps: 5000, r: .05 }));
        ctx.sb = START.map(p => s.ball(F, { start: p, step: step('sgd', g0, R2), speed: 4, maxSteps: 5000, r: .05, color: C.opt2, stop: out }));
        ctx.wb = s.ball(F, { start: [0, 2.5], step: step('tsgd', g0, R3), speed: 4, maxSteps: 5000, color: C.opt3, r: .05 });
      },
      onControl(ctx) { this.drop(ctx); },
      frame(ctx) {
        if (!ctx.tb) return;
        const atMin = b => Math.min(Math.hypot(b.p[0] - 1, b.p[1]), Math.hypot(b.p[0] + 1, b.p[1])) < .15, gone = b => b.done && b.steps < 5000;
        ctx.stat(`γ₀ = ${ctx.vals.g0} · step ${ctx.tb[0].steps} · TSGD at a minimum: ${ctx.tb.filter(atMin).length}/6 · plain SGD at a minimum: ${ctx.sb.filter(atMin).length}/6, left the picture: ${ctx.sb.filter(gone).length}/6 · stable-manifold start: ${atMin(ctx.wb) ? 'at a minimum' : Math.hypot(...ctx.wb.p) < .15 ? 'near the saddle' : 'moving'}`);
      },
    },
    {
      id: 'stability', name: 'Step-size stability', mode: '2d', title: 'Taming makes SGD robust to the initial step size',
      caption: 'Fraction of 400 runs (random starts with ‖x₀‖ ≤ 4) that end within 0.15 of a minimum after 5000 steps, against the initial step γ₀. Plain SGD diverges more and more often from γ₀ = 0.3 on; clipped SGD and tamed SGD stay stable up to γ₀ = 10, and no run of any method ends at the saddle. The paper reports the same kind of robustness for a deep diagonal network (its Fig. 2).',
      note: 'Offline. Clipped SGD rescales ĝ to norm ≤ 10. Same schedules and noise as the first tab. Right: the paper’s condition (25a) for TSGD to converge and avoid strict saddles, with the (p, q) used here.',
      hint: '', legend: [[HEX.opt2, 'plain SGD'], [HEX.flat, 'clipped SGD'], [HEX.ball, 'tamed SGD'], [HEX.neg, 'plain SGD diverged']],
      plot: { xlabel: 'initial step γ₀', ylabel: 'fraction of runs', logx: true, xlim: [.08, 12], ylim: [0, 1.05], legend: false },
      charts: [{ id: 'pq', title: 'TSGD region (25a): γₜ ∝ t^−p, λₜ ∝ t^−q', xlabel: 'p', ylabel: 'q', xlim: [.5, 1], ylim: [0, 1], h: 220, legend: false }],
      build(ctx) {
        const R = ctx.data.rows, Pl = ctx.plot.clear(), col = (i, c, o = {}) => Pl.line(R.map(r => [r[0], r[i]]), { color: c, width: 2.2, dots: 4, ...o });
        col(1, HEX.neg, { dash: [5, 4] }); col(2, HEX.opt2); col(5, HEX.flat); col(8, HEX.ball); Pl.draw();
        ctx.charts.pq.clear().fn(p => p / 2, { color: HEX.muted }).fn(p => 2 * p - 1, { color: HEX.muted }).vline(2 / 3, { color: HEX.muted, dash: [4, 4], text: 'p = 2/3' })
          .text(.86, .5, 'region', { color: HEX.good, align: 'center' }).scatter([[ctx.data.p, ctx.data.q]], { color: HEX.ball, r: 5 }).draw();
        ctx.stat(R.map(r => `γ₀ = ${r[0]}: SGD ${Math.round(100 * r[2])}% (diverged ${Math.round(100 * r[1])}%), clipped ${Math.round(100 * r[5])}%, TSGD ${Math.round(100 * r[8])}%`).join(' · '));
      },
    },
  ],
};
