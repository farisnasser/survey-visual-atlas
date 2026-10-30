// Cheridito, Jentzen, Rossmannek 2024 — GD on shallow scalar-input ReLU nets with an affine target avoids most saddles; if
// its limit loss is below f′(α)²(β−α)³/(12(N−1)⁴) it is a global minimum. Checks + 40 runs: scripts/check-math/cheridito2024gradient.py
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

/** Slice of the real loss L = ∫₀¹ (N(x) − x)² dx through c = ½ with one neuron v·max(b − x, 0) (w = −1) and the rest dead.
 *  Closed form: ∫₀^m (A − Bx)² + ∫_m¹ (½ − x)², m = min(b, 1), A = ½ + vb, B = 1 + v; L = 1/12 for b ≤ 0. */
const SL = (b, v) => {
  if (b <= 0) return 1 / 12;
  const m = Math.min(b, 1), A = .5 + v * b, B = 1 + v;
  return A * A * m - A * B * m * m + B * B * m ** 3 / 3 + (.125 - (m - .5) ** 3) / 3;
};
const F = def({ L: SL, dom: [-.4, 1.2, -2.5, 2.5], square: true, h: v => Math.log(v + .002), fit: 1.6 });
const LV = n => 1 / (12 * (n + 1) ** 4);

export default {
  tabs: [
    {
      id: 'slice', name: 'Loss slice', title: 'Dead neurons make flat ground; a live one can be a saddle or a trap',
      caption: 'A two-parameter slice of the true training loss of a shallow ReLU network fitting the line f(x) = x on [0, 1]: the output bias sits at f(½) = ½ and one neuron v·max(b − x, 0) has its kink at x = b. For b ≤ 0 the neuron is dead and the loss is the constant 1/12 whatever v is, a flat set of critical points. Just past b = 0 the loss drops if v < 0 (those points are saddles) but rises if v > 0 (spurious local minima, Prop. 4.5 (ii)). GD on the slice either finds the downhill side or freezes on the flat ground.',
      note: 'GD restricted to the slice (all other weights frozen), step 0.05, 24 random starts. Height log(L + 0.002). Theorem 1.2 excludes these traps through its condition on the limit loss.',
      legend: [[HEX.ball, 'GD on the slice'], [HEX.flat, 'b = 0: edge of the dead region (L = 1/12)'], [HEX.good, 'best point of the slice']],
      action: 'Release balls', cam: [0, 4.4, 4.4], tgt: [0, .3, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.tube(F, segPts(0, -2.5, 0, 2.5), C.flat, .014);
        let best = [0, 0, 1]; for (let i = 0; i <= 400; i++) for (let j = 0; j <= 400; j++) { const b = -.4 + 1.6 * i / 400, v = -2.5 + 5 * j / 400, L = SL(b, v); if (L < best[2]) best = [b, v, L]; }
        ctx.best = best; s.marker(F, best[0], best[1], C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(6); s.clearBalls();
        ctx.bs = Array.from({ length: 24 }, () => s.ball(F, { start: [R.uniform(-.3, 1.1), R.uniform(-2.3, 2.3)], speed: 6, maxSteps: 6000, r: .045, step: (p, b, d) => { const g = d.g(p[0], p[1]); return [p[0] - .05 * g[0], p[1] - .05 * g[1]]; } }));
      },
      frame(ctx) {
        if (!ctx.bs) return;
        const dead = ctx.bs.filter(b => b.p[0] <= 0), good = ctx.bs.filter(b => Math.abs(b.loss() - ctx.best[2]) < 1e-4).length;
        ctx.stat(`frozen on the dead region (L = 1/12): ${dead.length}/24, ${dead.filter(b => b.p[1] > 0).length} of them with v > 0 (spurious minima) · at the slice minimum L = ${ctx.best[2].toFixed(4)} (b = ${ctx.best[0].toFixed(2)}, v = ${ctx.best[1].toFixed(2)}): ${good}/24`);
      },
    },
    {
      id: 'levels', name: 'Critical levels', mode: '2d', title: 'Non-global critical points sit at quantized losses; runs below the threshold end at zero',
      caption: 'For an affine target every critical point that is not a global minimum has loss exactly f′(α)²(β − α)³/(12(n + 1)⁴) for an even n, and the network is then a sawtooth around the line (right: n = 2). Dashed: these levels for n = 0, 2, 4; red: Theorem 1.2’s threshold for N = 4 neurons. All 40 GD runs from random starts with small output weights pass near the n = 0 level and end at loss 0, as the theorem says they must once they are below the threshold.',
      note: 'Offline. N = 4, target f(x) = x on [0, 1], exact square integral loss (fine midpoint rule), GD step 0.2, 20 000 steps. The theorem holds for almost every step size and start.',
      hint: '', legend: [[HEX.ball, 'training loss of one run'], [HEX.muted, 'critical levels n = 0, 2, 4'], [HEX.neg, 'threshold 1/(12(N − 1)⁴)']],
      plot: { xlabel: 'GD step', ylabel: 'loss', logx: true, logy: true, xlim: [1, 2e4], ylim: [1e-9, 1], legend: false },
      charts: [{ id: 'saw', title: 'the n = 2 critical network (sawtooth) vs f(x) = x', xlabel: 'x', ylabel: 'output', xlim: [0, 1], ylim: [-.05, 1.05], h: 220, legend: false }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        D.curves.forEach(cv => P.line(cv.map(q => [q[0], Math.max(q[1], 1e-12)]), { color: HEX.ball, width: 1, alpha: .5 }));
        [0, 2, 4].forEach(n => P.hline(LV(n), { color: HEX.muted, dash: [5, 4], text: `n = ${n}: ${LV(n).toExponential(2)}` }));
        P.hline(D.thr, { color: HEX.neg, dash: [2, 3] }).draw();
        const saw = x => { const i = Math.min(Math.floor(x * 3), 2); return x - (-1) ** i / 3 * (x - (i + .5) / 3); };
        ctx.charts.saw.clear().fn(x => x, { color: HEX.muted, dash: [4, 4] }).fn(saw, { color: HEX.good, width: 2.2 }).draw();
        const zero = D.final.filter(q => q < 1e-6).length;
        ctx.stat(`threshold for N = 4: ${D.thr.toExponential(3)} (= the n = 2 level) · runs ending below it: ${D.final.filter(q => q < D.thr).length}/40, all at loss < 10⁻⁷ (${zero}/40 below 10⁻⁶)`);
      },
    },
  ],
};
