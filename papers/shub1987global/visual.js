// Shub 1987 — the (center-)stable manifold theorem behind every "GD avoids strict saddles" proof: the starts that converge to a
// hyperbolic fixed point form a lower-dimensional manifold tangent to the contracting directions. scripts/check-math/shub1987global.py
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';

const K = .8, ETA = .1;
/** f = x²/2 − u²/2 + u⁴/4, u = y − kx²: saddle at 0, minima (0, ±1); the GD map's stable manifold is a curve, not the x-axis */
const F = def({
  L: (x, y) => { const u = y - K * x * x; return x * x / 2 - u * u / 2 + u ** 4 / 4; },
  g: (x, y) => { const u = y - K * x * x, w = -u + u ** 3; return [x - 2 * K * x * w, w]; },
  dom: [-1.2, 1.2, -1.2, 1.6],
});
const gd = (p, b, d) => { const g = d.g(...p); return [p[0] - ETA * g[0], p[1] - ETA * g[1]]; };
const DL = [1e-2, 1e-6, 1e-10];

export default {
  tabs: [
    {
      id: 'manifold', name: 'Stable manifold', title: 'The starts that reach a saddle form a thin curve',
      caption: 'Gradient descent on a function with a strict saddle at the origin. The white curve is the saddle’s stable manifold for the GD map, computed by bisection: from any point on it GD runs into the saddle (white ball). It passes through the saddle tangent to the contracting direction (blue) but bends away from that line, as the theorem says a manifold, not a subspace. Starts just above it (coral, 10⁻², 10⁻⁶, 10⁻¹⁰ off) all leave along the expanding direction (red), later the closer they start; a curve has zero area, so a random start avoids it with probability one.',
      note: 'f = x²/2 − u²/2 + u⁴/4 with u = y − 0.8x², GD step 0.1, so DG(0) = diag(0.9, 1.1): a hyperbolic fixed point. All balls start at x = 0.8. The white ball also leaves eventually: rounding errors are amplified by 1.1 per step.',
      legend: [[HEX.opt3, 'GD from the stable manifold'], [HEX.ball, 'GD from 10⁻², 10⁻⁶, 10⁻¹⁰ above it'], ['#ffffff', 'stable manifold (computed)'], [HEX.good, 'minima (0, ±1)']],
      action: 'Release balls', cam: [2.6, 3.8, 3.6], tgt: [0, .2, 0],
      build(ctx) {
        const s = ctx.stage, D = ctx.data; s.surface(F); s.marker(F, 0, 1, C.good); s.marker(F, 0, -1, C.good); s.eigenLines(F, 0, 0, { len: .5 });
        s.tube(F, D.xs.map((x, i) => [x, D.ys[i]]), C.white, .014); this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, i = D.xs.findIndex(x => Math.abs(x - .8) < 1e-9), y0 = D.ys[i]; s.clearBalls(); ctx.esc = [null, null, null]; ctx.dmin = Infinity;
        ctx.on = s.ball(F, { start: [.8, y0], step: gd, color: C.opt3, speed: 1, maxSteps: 600 });
        ctx.on.onStep = b => { ctx.dmin = Math.min(ctx.dmin, Math.hypot(...b.p)); };
        ctx.off = DL.map((dl, k) => { const b = s.ball(F, { start: [.8, y0 + dl], step: gd, speed: 1, maxSteps: 600, r: .05 }); b.onStep = q => { if (ctx.esc[k] == null && Math.abs(q.p[1] - K * q.p[0] ** 2) >= .5) ctx.esc[k] = q.steps; }; return b; });
      },
      frame(ctx) {
        if (!ctx.on) return;
        ctx.stat(`step ${ctx.on.steps} · ball on the manifold: closest approach to the saddle ${ctx.dmin.toExponential(1)} · escape steps from δ = 10⁻², 10⁻⁶, 10⁻¹⁰: ${ctx.esc.map(v => v ?? '…').join(', ')} (check script: ${ctx.data.esc.filter(q => DL.includes(q[0])).map(q => q[1]).join(', ')})`);
      },
    },
    {
      id: 'curve', name: 'Tangent, not equal', mode: '2d', title: 'Tangent to the contracting eigenvector, curved away from it',
      caption: 'The computed stable manifold y*(x) (white) against the contracting eigen-direction of DG(0), the x-axis (blue). Near the saddle y*(x) ≈ 0.28x², so the two touch at the origin and separate quadratically. Right: steps needed to leave from a distance δ above the manifold grow like log(1/δ)/log 1.1, about 24 extra steps per factor 100 in δ: avoidance is guaranteed, speed is not.',
      note: 'Bisection on y at 49 values of x in [−1.2, 1.2], 55 halvings each, classifying starts by the minimum they reach after 400 GD steps.',
      hint: '', legend: [['#ffffff', 'stable manifold y*(x)'], [HEX.pos, 'contracting eigen-direction (y = 0)'], [HEX.muted, 'fit 0.28x²']],
      plot: { xlabel: 'x', ylabel: 'y', xlim: [-1.2, 1.2], ylim: [-.1, .7], legend: false },
      charts: [{ id: 'esc', title: 'steps to leave from δ above the manifold', xlabel: 'δ', ylabel: 'steps', logx: true, xlim: [5e-11, .02], ylim: [0, 260], h: 220, legend: false }],
      build(ctx) {
        const D = ctx.data;
        ctx.plot.clear().hline(0, { color: HEX.pos, width: 2 }).fn(x => D.fit * x * x, { color: HEX.muted, dash: [5, 4] }).line(D.xs.map((x, i) => [x, D.ys[i]]), { color: '#ffffff', width: 2.4, dots: 2.5 }).draw();
        ctx.charts.esc.clear().fn(d => D.esc[0][1] + Math.log(D.esc[0][0] / d) / Math.log(1.1), { color: HEX.muted, dash: [5, 4] }).line(D.esc, { color: HEX.ball, width: 2.2, dots: 4 }).draw();
        ctx.stat(`fit near 0: y* ≈ ${D.fit.toFixed(3)}x² · y*(±1.2) = ${D.ys[0].toFixed(3)} · escape steps ${D.esc.map(q => `δ = ${q[0].toExponential(0)}: ${q[1]}`).join(', ')}`);
      },
    },
  ],
};
