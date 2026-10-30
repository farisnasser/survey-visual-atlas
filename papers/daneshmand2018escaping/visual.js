// Daneshmand, Kohler, Lucchi, Hofmann 2018 — Correlated Negative Curvature: SGD's own sampling noise escapes saddles
// (CNC-PGD, Alg. 1; CNC-SGD, Alg. 2). Components and the dimension sweep: scripts/check-math/daneshmand2018escaping.py (data.json).
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => .5 * x * x - .5 * y * y + .25 * y ** 4, g: (x, y) => [x, -y + y ** 3], hess: (x, y) => [1, 0, -1 + 3 * y * y], h: v => v, dom: [-1.7, 1.7, -1.6, 1.6] });
const ETA = .05, RBIG = .4, GT = 1e-6, TT = 50;

export default {
  tabs: [
    {
      id: 'cnc', name: 'CNC', title: 'Sampling noise that leans on the escape direction is enough',
      caption: 'f is the mean of 64 components f_i = f + a_i y + c_i x. All balls start on the saddle’s stable set (y = 0). GD (white) converges to the saddle. CNC-PGD (coral) runs GD and, when the gradient is small, takes one stochastic-gradient step with a larger step r. CNC-SGD (violet) runs SGD with a larger step every t_thres iterations. No artificial noise is added; with CNC the component gradients have E⟨e_y, ∇f_i⟩² > 0, and both escape. Switch the noise to “orthogonal” (a_i = 0) and they stay stuck.',
      note: 'f = x²/2 − y²/2 + y⁴/4, η = 0.05, r = 0.4, g_thres = 10⁻⁶, t_thres = 50. The guaranteed method needs these periodic large steps; it is not a theorem about constant-step SGD.',
      legend: [[HEX.opt3, 'GD'], [HEX.ball, 'CNC-PGD (Alg. 1)'], [HEX.opt2, 'CNC-SGD (Alg. 2)'], [HEX.good, 'minima (0, ±1)'], [HEX.neg, 'escape direction v = e_y']],
      action: 'Run again', cam: [3.4, 3.2, 4.2], tgt: [0, .3, 0],
      controls: [{ id: 'noise', type: 'seg', label: 'component noise', value: 'cnc', options: [['cnc', 'has a y-part (CNC holds)'], ['orth', 'x only (CNC fails)']] }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .6 }); s.marker(F, 0, 1, C.good); s.marker(F, 0, -1, C.good); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, R = rng(Date.now() % 1e6), cnc = ctx.vals.noise === 'cnc', n = D.a.length; s.clearBalls();
        const gi = (p, i) => { const [gx, gy] = F.g(...p); return [gx + D.c[i], gy + (cnc ? D.a[i] : 0)]; };
        ctx.gd = s.ball(F, { start: [1.5, 0], step: opt.gd(ETA), color: C.opt3, speed: 3, maxSteps: 3000 });
        ctx.pgd = s.ball(F, { start: [1.5, 0], speed: 3, maxSteps: 3000, step: (p, b) => {
          b.t = (b.t ?? -1) + 1; b.tn ??= -TT - 1; b.kicks ??= 0;
          const [gx, gy] = F.g(...p);
          if (gx * gx + gy * gy <= GT && b.t - b.tn >= TT) { b.tn = b.t; b.kicks++; const q = gi(p, Math.floor(R() * n)); return [p[0] - RBIG * q[0], p[1] - RBIG * q[1]]; }
          return [p[0] - ETA * gx, p[1] - ETA * gy];
        } });
        ctx.sgd = s.ball(F, { start: [1.5, 0], color: C.opt2, speed: 3, maxSteps: 3000, step: (p, b) => {
          b.t = (b.t ?? -1) + 1; const st = b.t % TT === 0 ? RBIG : ETA, q = gi(p, Math.floor(R() * n)); return [p[0] - st * q[0], p[1] - st * q[1]];
        } });
        const gam = cnc ? D.a.reduce((s2, v) => s2 + v * v, 0) / n : 0;
        ctx.gam = gam;
      },
      onControl(ctx) { this.drop(ctx); },
      frame(ctx) {
        const { gd, pgd, sgd } = ctx; if (!gd) return;
        const where = b => Math.abs(b.p[1]) > .5 ? 'escaped' : 'at the saddle';
        ctx.stat(`CNC constant E⟨e_y, ∇f_i(0)⟩² = ${ctx.gam.toFixed(3)} · GD: ${where(gd)} · CNC-PGD: ${where(pgd)} (${pgd.kicks || 0} stochastic steps) · CNC-SGD: ${where(sgd)}`);
      },
    },
    {
      id: 'dim', name: 'Dimension', mode: '2d', title: 'Isotropic kicks spread thin in high dimension; CNC noise does not',
      caption: 'One kick of the same energy r² at the saddle of −w₁²/2 + w₁⁴/4 + ‖w_rest‖²/2 in R^d, then gradient descent. An isotropic kick puts only r²/d of its energy on the escape direction, so escape slows as d grows (logarithmically). Noise that satisfies CNC keeps a fixed share on that direction, so the escape time does not depend on d.',
      note: 'Offline, median of 60 runs per d, r = 0.1, η = 0.1, escaped when |w₁| > 0.5. The CNC kick gives half its energy to e₁ (E⟨e₁, ξ⟩² = r²/2).',
      hint: '', legend: [[HEX.pos, 'isotropic kick (uniform on the sphere)'], [HEX.ball, 'CNC-aligned kick']],
      plot: { xlabel: 'dimension d', ylabel: 'steps to escape', logx: true, xlim: [1.5, 15000], ylim: [0, 80] },
      build(ctx) {
        const D = ctx.data.dim, P = ctx.plot.clear();
        P.line(D.map(r => [r[0], r[1]]), { color: HEX.pos, width: 2.4, dots: 4, label: 'isotropic' }).line(D.map(r => [r[0], r[2]]), { color: HEX.ball, width: 2.4, dots: 4, label: 'CNC' }).draw();
        ctx.stat(D.map(r => `d = ${r[0]}: ${r[1]} vs ${r[2]}`).join(' · '));
      },
    },
  ],
};
