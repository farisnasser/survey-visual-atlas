// Islamov, Ajroldi, Orvieto, Lucchi 2024 — the α-β condition admits saddle points (unlike PL). The landscape is the
// paper's Example 2; the α needed on the box and the PL ratio come from scripts/check-math/islamov2024loss.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rampCss } from '../../src/lib/charts.js';
import { rng } from '../../src/lib/math.js';

const E = (x, y, a, b) => Math.exp(-((x - a) ** 2 + (y - b) ** 2));
/** f = ½(f₁ + f₂), f₁ = 1 − e^{−‖x‖²}, f₂ = 1 − e^{−‖x − (2,2)‖²} (Example 2, Eq. 4). */
const D = def({
  L: (x, y) => .5 * (2 - E(x, y, 0, 0) - E(x, y, 2, 2)),
  g: (x, y) => { const a = E(x, y, 0, 0), b = E(x, y, 2, 2); return [x * a + (x - 2) * b, y * a + (y - 2) * b]; },
  h: v => v, dom: [-1.5, 3.5, -1.5, 3.5],
});
/** SGD: one component f_i per step, i uniform. ∇f_i(x) = 2(x − cᵢ)e^{−‖x − cᵢ‖²}. */
const sgd = (gamma, R) => p => { const c = R() < .5 ? 0 : 2, e = E(p[0], p[1], c, c); return [p[0] - gamma * 2 * (p[0] - c) * e, p[1] - gamma * 2 * (p[1] - c) * e]; };

export default {
  tabs: [
    {
      id: 'saddle', name: 'Saddle allowed', title: 'A landscape with a saddle that still satisfies the α-β condition',
      caption: 'The paper’s Example 2: the average of two Gaussian wells, one at the origin and one at (2, 2), with a strict saddle at (1, 1) between them. PL cannot hold here (the gradient vanishes at the saddle while the loss is above its minimum), yet the α-β condition does. SGD that samples one well per step settles into one of the two global minima; starts far out on the flat rim drift slowly.',
      note: 'f = ½(f₁ + f₂), f₁ = 1 − e^{−‖x‖²}, f₂ = 1 − e^{−‖x − (2,2)‖²}; SGD picks f₁ or f₂ at random, step 0.5. Height is f itself.',
      legend: [[HEX.ball, 'SGD from random starts'], [HEX.good, 'global minimizers'], [HEX.neg, 'λ < 0 at the saddle']],
      action: 'Release balls', cam: [-3.2, 3.6, 3.2], tgt: [0, .3, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(D); ctx.m = ctx.data.m1;
        s.marker(D, ctx.m[0], ctx.m[1], C.good); s.marker(D, 2 - ctx.m[0], 2 - ctx.m[1], C.good); s.eigenLines(D, 1, 1, { len: .7 });
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(2), N = rng(11); s.clearBalls(); ctx.bs = [];
        for (let k = 0; k < 16; k++) ctx.bs.push(s.ball(D, { start: [R.uniform(-1.4, 3.4), R.uniform(-1.4, 3.4)], step: sgd(.5, N), speed: 3, maxSteps: 3000, r: .05 }));
      },
      frame(ctx) {
        if (!ctx.bs) return;
        const m = ctx.m, near = ctx.bs.filter(b => Math.min(Math.hypot(b.p[0] - m[0], b.p[1] - m[1]), Math.hypot(b.p[0] - 2 + m[0], b.p[1] - 2 + m[1])) < .05).length, rim = ctx.bs.filter(b => b.loss() > .95).length;
        ctx.stat(`step ${Math.max(...ctx.bs.map(b => b.steps))} · within 0.05 of a global minimizer: ${near}/16 · on the flat rim (f > 0.95): ${rim}/16 · saddle (1, 1): ‖∇f‖ = 0, f − f* = 0.365, so PL fails there`);
      },
    },
    {
      id: 'condition', name: 'α needed', mode: '2d', title: 'Definition 1 holds, but only with an enormous α',
      caption: 'Colour: the smallest α for which Eq. (2) holds at each point with β = α − 8 (log scale). It is finite everywhere, saddle included, so Example 2 is in the class. But f₁ is only 9 × 10⁻⁷ at its nearest global minimizer, so α must reach about 9 × 10⁶, and Theorem 1’s error floor 2βσ²ᵢₙₜ/(α − β) is about 10⁶, far above the whole range of f − f* (0.5). Right: the PL ratio along the diagonal drops to 0 at the saddle.',
      note: 'Proj(x, S) = nearest global minimizer (either one when tied). Any other α − β gives the same floor, about 2σ²ᵢₙₜ/f₁(x_p). The paper says it does not aim for tight α, β.',
      hint: '', legend: [[HEX.good, 'global minimizers'], [HEX.neg, 'saddle']],
      plot: { xlabel: 'x', ylabel: 'y', xlim: [-1.5, 3.5], ylim: [-1.5, 3.5], legend: false },
      charts: [{ id: 'pl', title: 'PL ratio ‖∇f‖² / 2(f − f*) on x = y', xlabel: 't (point (t, t))', ylabel: 'ratio', xlim: [-1.5, 3.5], h: 220, legend: false }],
      build(ctx) {
        const Dd = ctx.data, z = Dd.req.z, zs = z.flat(), lo = Math.min(...zs), hi = Math.max(...zs), m = Dd.m1;
        ctx.plot.clear().heat(z, [-1.5, 3.5, -1.5, 3.5], { cell: v => rampCss((v - lo) / (hi - lo)) })
          .scatter([[m[0], m[1]], [2 - m[0], 2 - m[1]]], { color: HEX.good, r: 5 }).scatter([[1, 1]], { color: HEX.neg, r: 5 })
          .text(1.15, .8, 'saddle', { color: HEX.neg }).text(3.4, 3.3, `log₁₀ α from ${lo.toFixed(1)} to ${hi.toFixed(1)}`, { align: 'right', color: '#e8ecf8' }).draw();
        ctx.charts.pl.clear().line(Dd.pl.filter(q => q[1] != null), { color: HEX.ball, width: 2 }).vline(1, { color: HEX.neg, dash: [4, 4], text: 'saddle' }).draw();
        ctx.stat(`α ≥ ${(Dd.alpha / 1e6).toFixed(1)} × 10⁶ on [−1.5, 3.5]² (β = α − 8) · L = ${Dd.L} · σ²ᵢₙₜ = f* = ${Dd.fstar.toFixed(4)} · Theorem 1 floor 2βσ²ᵢₙₜ/(α − β) ≈ ${(Dd.floor / 1e6).toFixed(1)} × 10⁶ vs max (f − f*) = 0.50`);
      },
    },
  ],
};
