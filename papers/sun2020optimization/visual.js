// Sun 2020 — overview of optimization for deep learning; its first obstacle is gradient explosion/vanishing in deep chains.
// Depth-L scalar linear chain F = (w₁⋯w_L − 1)², sliced at w₁ = a, w₂ = … = w_L = b. scripts/check-math/sun2020optimization.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const slice = L => def({
  L: (a, b) => (a * b ** (L - 1) - 1) ** 2,
  g: (a, b) => { const e = a * b ** (L - 1) - 1; return [2 * e * b ** (L - 1), 2 * e * (L - 1) * a * b ** (L - 2)]; },
  dom: [-1.4, 1.4, -1.4, 1.4], h: v => Math.log1p(Math.min(v, 1e6)) * .7,
});
const ETA = .02;

export default {
  tabs: [
    {
      id: 'chain', name: 'Exploding & vanishing', title: 'A deep chain is flat near zero and a cliff beyond one',
      caption: 'The loss of a depth-L scalar linear network, F = (w₁w₂⋯w_L − 1)², on the slice where the last L − 1 weights are equal. The gradient with respect to w₁ is 2e·w₂⋯w_L, so with weights below 1 it shrinks like b^{L−1} (the flat plateau around zero, where GD crawls) and above 1 it grows like b^{L−1} (the cliffs, where a step that is safe near the solution sends GD flying). Deeper chains make both effects stronger. This is why initialization scale and normalization matter, the first topic of the overview.',
      note: 'GD step 0.02 on the slice from (0.3, 0.3) (inside the plateau), (0.9, 0.9) and (1.25, 1.25) (on the cliff); balls that leave the box stop at its edge.',
      legend: [[HEX.ball, 'start (0.3, 0.3)'], [HEX.opt2, 'start (0.9, 0.9)'], [HEX.opt3, 'start (1.25, 1.25)'], [HEX.good, 'zero-loss curve a·b^{L−1} = 1']],
      action: 'Release balls', cam: [0, 4.4, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'L', type: 'seg', label: 'depth L', value: 7, options: [[2, '2'], [4, '4'], [7, '7']] }],
      build(ctx) {
        const L = +ctx.vals.L, d = ctx.d = slice(L), s = ctx.stage; s.surface(d);
        const pts = []; for (let b = .72; b <= 1.4; b += .01) pts.push([1 / b ** (L - 1), b]);
        s.tube(d, pts.filter(p => Math.abs(p[0]) <= 1.4), C.good, .014);
        if (L % 2 === 0) s.tube(d, pts.map(p => [-p[0], -p[1]]).filter(p => Math.abs(p[0]) <= 1.4), C.good, .014);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, d = ctx.d; s.clearBalls();
        const st = (p, b, dd) => { const g = dd.g(p[0], p[1]); return [p[0] - ETA * g[0], p[1] - ETA * g[1]]; };
        const out = b => Math.abs(b.p[0]) >= 1.399 || Math.abs(b.p[1]) >= 1.399;
        ctx.conv = [null, null, null];
        ctx.bs = [[.3, .3, C.ball], [.9, .9, C.opt2], [1.25, 1.25, C.opt3]].map(([a, b, col], i) => { const q = s.ball(d, { start: [a, b], step: st, color: col, speed: 2, maxSteps: 3000, stop: out }); q.onStep = z => { if (ctx.conv[i] == null && z.loss() < 1e-6) ctx.conv[i] = z.steps; }; return q; });
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const L = +ctx.vals.L, gn = b => Math.hypot(...ctx.d.g(...b.p)), st = (b, i) => b.done && b.steps < 3000 && ctx.conv[i] == null ? 'left the box (exploded)' : ctx.conv[i] != null ? `loss < 10⁻⁶ after ${ctx.conv[i]} steps` : `loss ${b.loss().toExponential(1)}`;
        ctx.stat(`L = ${L} · step ${ctx.bs[0].steps} · gradient at start: ${ctx.bs.map(b => gn({ p: b === ctx.bs[0] ? [.3, .3] : b === ctx.bs[1] ? [.9, .9] : [1.25, 1.25] }).toExponential(1)).join(', ')} · now: ${ctx.bs.map(st).join(' · ')}`);
      },
    },
  ],
};
