// Mohtashami, Jaggi, Stich 2023 — with a large step GD cannot stay in a sharp spurious minimum and, on suitable
// landscapes, converges to the global one (Lemma 1, Theorem 1); small steps can get trapped. Toy and the outcome-vs-η
// curve are checked in scripts/check-math/mohtashami2023special.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const CG = [.8, -.4], CS = [-.7, .5], B = .15, A = .35, S = .15, START = [-.55, .62];
const well = (x, y) => A * Math.exp(-((x - CS[0]) ** 2 + (y - CS[1]) ** 2) / (2 * S * S));
const L = (x, y) => B * ((x - CG[0]) ** 2 + (y - CG[1]) ** 2) - well(x, y);
const G = (x, y) => { const w = well(x, y); return [2 * B * (x - CG[0]) + w * (x - CS[0]) / (S * S), 2 * B * (y - CG[1]) + w * (y - CS[1]) / (S * S)]; };
const surf = def({ L, g: G, dom: [-1.6, 1.6, -1.6, 1.6], h: v => Math.log1p(4 * (v + .3)) });
const f4 = v => (Math.abs(v) < 5e-5 ? 0 : v).toFixed(4);
const gd = eta => p => { const g = G(p[0], p[1]); return [p[0] - eta * g[0], p[1] - eta * g[1]]; };

export default {
  tabs: [
    {
      id: 'escape', name: 'Large step escapes', title: 'A step too big for the sharp minimum finds the global one',
      caption: 'A wide bowl around the global minimum (green) with a narrow, higher spurious well (red). From the same start, small-step GD (violet) slides into the narrow well and stays. With a step above 2/λ of that well (≈ 0.13) the well is linearly unstable, so GD (coral) is thrown out of it, the gradient norm briefly grows, and it converges to the global minimum; the wide bowl stays stable for any step below 2/0.3 ≈ 6.7.',
      note: 'Deterministic full-batch GD on a toy built like the paper’s function class: one-point convex towards the global minimum outside a small sharp region. Not the paper’s networks.',
      legend: [[HEX.opt2, 'GD, η = 0.05'], [HEX.ball, 'GD, chosen η'], [HEX.good, 'global min (λ = 0.3)'], [HEX.neg, 'spurious min (λ = 15.5)']],
      action: 'Restart', cam: [0, 4.4, 4.4], tgt: [0, .1, 0],
      controls: [{ id: 'eta', label: 'large step η', min: .05, max: 1.5, step: .01, value: .3 }],
      charts: [{ id: 'g', title: '‖∇f‖ along the chosen-η run (log)', xlabel: 'step', ylabel: '‖∇f‖', logy: true, ylim: [1e-8, 10], xlim: [0, 300], h: 220, legend: false }],
      build(ctx) {
        const s = ctx.stage; s.surface(surf); s.marker(surf, ...ctx.data.zg, C.good, .12); s.marker(surf, ...ctx.data.zl, C.neg, .07);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, eta = ctx.vals.eta; s.clearBalls(); ctx.hist = [[0, Math.hypot(...G(...START))]];
        ctx.small = s.ball(surf, { start: START, speed: 1, maxSteps: 3000, step: gd(.05), color: C.opt2 });
        ctx.big = s.ball(surf, { start: START, speed: 1, maxSteps: 3000, step: gd(eta) });
        ctx.big.onStep = b => ctx.hist.push([b.steps, Math.hypot(...G(...b.p))]);
      },
      frame(ctx) {
        const zg = ctx.data.zg, zl = ctx.data.zl, where = b => Math.hypot(b.p[0] - zg[0], b.p[1] - zg[1]) < 1e-2 ? 'global minimum' : Math.hypot(b.p[0] - zl[0], b.p[1] - zl[1]) < 1e-2 ? 'spurious minimum' : 'moving';
        ctx.charts.g.clear().line(ctx.hist.map(h => [h[0], Math.max(h[1], 1e-8)]), { color: HEX.ball, width: 1.6 }).draw();
        const eta = ctx.vals.eta;
        ctx.stat(`η = ${eta.toFixed(2)} (${eta > 2 / ctx.data.lamL ? '>' : '<'} 2/λ_spurious = ${(2 / ctx.data.lamL).toFixed(3)}): ${where(ctx.big)}, L = ${f4(L(...ctx.big.p))} · η = 0.05: ${where(ctx.small)}, L = ${f4(L(...ctx.small.p))}`);
      },
    },
    {
      id: 'odds', name: 'Outcome vs η', mode: '2d', title: 'Bigger steps, fewer starts trapped in the sharp minimum',
      caption: 'Fraction of 600 random starts that end at the global minimum, at the spurious minimum, or neither (still bouncing inside the narrow well after 8000 steps), for each step size. Trapping at the spurious minimum stops exactly at η = 2/λ_spurious; just above it some starts settle into an oscillation inside the well, and from η ≈ 0.3 every start reaches the global minimum, until steps beyond 2/0.3 would make the bowl itself unstable.',
      note: 'Same toy; starts uniform on [−1.6, 1.6]². Theorem 2 of the paper lower-bounds the probability of never visiting a small region by a quantity increasing in η (reported; the toy only shows the same trend).',
      hint: '', legend: [[HEX.good, 'global'], [HEX.neg, 'spurious'], [HEX.flat, 'neither (oscillating)']],
      plot: { xlabel: 'step size η (log)', ylabel: 'fraction of starts', logx: true, xlim: [.008, 8], ylim: [0, 1.05], legend: false },
      build(ctx) {
        const F = ctx.data.frac, P = ctx.plot.clear();
        P.vline(2 / ctx.data.lamL, { color: HEX.neg, dash: [4, 4], text: '2/λ_spurious' }).vline(2 / ctx.data.lamG, { color: HEX.good, dash: [4, 4], text: '2/λ_global' })
          .line(F.map(f => [f[0], f[1]]), { color: HEX.good, width: 2.2, dots: 3 }).line(F.map(f => [f[0], f[2]]), { color: HEX.neg, width: 2.2, dots: 3 })
          .line(F.map(f => [f[0], Math.max(0, 1 - f[1] - f[2] - f[3])]), { color: HEX.flat, width: 2.2, dots: 3 }).draw();
        const at = e => F.find(f => f[0] === e);
        ctx.stat(`η = 0.1: global ${(100 * at(.1)[1]).toFixed(1)}%, spurious ${(100 * at(.1)[2]).toFixed(1)}% · η = 0.15: spurious 0%, neither ${(100 * (1 - at(.15)[1])).toFixed(1)}% · η ≥ 0.3: global 100%`);
      },
    },
  ],
};
