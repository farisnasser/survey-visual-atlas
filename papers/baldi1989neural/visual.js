// Baldi & Hornik 1989 — linear autoencoder 2 -> 1 -> 2 on data with covariance diag(9, 1).
// Formulas checked in scripts/check-math/baldi1989neural.py
import { def, opt, C, HEX, logH, hyperbola, segPts, statLabels } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const S1 = 9, S2 = 1; // eigenvalues of the covariance
// tied slice B = A^T = w^T: E(w) = tr S - 2 w'Sw + |w|^2 w'Sw
const tied = def({
  L: (x, y) => { const q = S1 * x * x + S2 * y * y, n = x * x + y * y; return S1 + S2 - 2 * q + n * q; },
  g: (x, y) => { const q = S1 * x * x + S2 * y * y, n = x * x + y * y; return [-4 * S1 * x + 2 * x * q + 2 * n * S1 * x, -4 * S2 * y + 2 * y * q + 2 * n * S2 * y]; },
  hess: (x, y) => { const q = S1 * x * x + S2 * y * y, n = x * x + y * y; return [-4 * S1 + 2 * q + 4 * S1 * x * x + 2 * n * S1 + 4 * S1 * x * x, 4 * S1 * x * y + 4 * S2 * x * y, -4 * S2 + 2 * q + 4 * S2 * y * y + 2 * n * S2 + 4 * S2 * y * y]; },
  h: L => Math.log1p(Math.max(L - 1, 0)) * .62, dom: [-1.35, 1.35, -1.35, 1.35],
});
// untied slice A = (a, 0)^T, B = (b, 0): E = 9 (1 - ab)^2 + 1
const untied = def({ L: (a, b) => S1 * (1 - a * b) ** 2 + S2, g: (a, b) => { const r = 1 - a * b; return [-2 * S1 * b * r, -2 * S1 * a * r]; }, h: L => logH(L - 1) * 1.1 });

export default {
  tabs: [
    {
      id: 'pca', name: 'PCA valley', title: 'Every valley is the top principal direction',
      caption: 'Loss of a 2 → 1 → 2 linear autoencoder on data with variances 9 and 1. Balls from random starts all end at ±u₁.',
      note: 'Tied-weight slice B = Aᵀ (2 of 4 weights). In the full space the origin is a saddle, not a maximum.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'global minima ±u₁ (E = 1)'], [HEX.neg, 'λ < 0'], [HEX.pos, 'λ > 0']],
      action: 'Drop 12 balls', cam: [3.3, 3.6, 4.3], tgt: [0, .6, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(tied);
        s.marker(tied, 1, 0, C.good); s.marker(tied, -1, 0, C.good);
        s.marker(tied, 0, 1, C.white); s.marker(tied, 0, -1, C.white);
        s.eigenLines(tied, 0, 1, { len: .55 });
        s.label('+u₁ · E = 1', tied.W(1, 0, .3), '', HEX.good); s.label('−u₁ · E = 1', tied.W(-1, 0, .3), '', HEX.good);
        s.label('saddle u₂ · E = 9', tied.W(0, 1, .75), '', HEX.white);
        s.label('max on this slice · E = 10', tied.W(0, 0, .3), '', HEX.muted);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, r = rng(Date.now() % 1e6); s.clearBalls();
        for (let i = 0; i < 12; i++) {
          let p; do { p = [r.uniform(-1.25, 1.25), r.uniform(-1.25, 1.25)]; } while (Math.hypot(...p) < .1);
          s.ball(tied, { start: p, step: opt.gd(.01), speed: 3, stop: b => Math.hypot(...tied.g(...b.p)) < 1e-5 });
        }
      },
      frame(ctx) {
        const bs = ctx.stage.balls, at = bs.filter(b => b.done && Math.abs(b.loss() - 1) < 1e-3).length;
        ctx.stat(`${at} / ${bs.length} balls at E = 1 (±u₁)`);
      },
    },
    {
      id: 'angle', name: 'Projection angle', mode: '2d', title: 'Which direction keeps the most variance?',
      caption: 'Keep one number per point: its coordinate along a unit direction u(θ). The error is the variance you throw away.',
      note: '200 samples drawn from N(0, diag(9, 1)); the curve uses the exact covariance.',
      legend: [[HEX.ball, 'projection line u(θ)'], [HEX.muted, 'reconstruction error']],
      hint: 'drag the slider, or let gradient descent turn the line',
      controls: [{ id: 'th', label: 'angle θ', min: 0, max: 180, step: 1, value: 70, fmt: v => `${(+v).toFixed(0)}°` }],
      actions: [{ label: 'Descend on θ', fn: ctx => { ctx.anim = true; } }],
      plot: { xlim: [-9, 9], ylim: [-5.2, 5.2], xlabel: 'x₁', ylabel: 'x₂', legend: false },
      charts: [{ id: 'e', title: 'E(θ) = 1 + 8 sin²θ', xlabel: 'θ (degrees)', ylabel: 'E', xlim: [0, 180], ylim: [0, 10], legend: false }],
      build(ctx) {
        const r = rng(4); ctx.X = Array.from({ length: 200 }, () => [3 * r.normal(), r.normal()]); ctx.anim = false; this.draw(ctx);
      },
      onControl(ctx) { ctx.anim = false; this.draw(ctx); },
      frame(ctx) {
        if (!ctx.anim) return;
        const t = ctx.vals.th * Math.PI / 180, grad = 16 * Math.sin(t) * Math.cos(t);  // dE/dθ
        let nt = t - .02 * grad; if (Math.abs(nt - t) < 1e-5) { ctx.anim = false; }
        if (Math.abs(Math.sin(t)) > .9999 && Math.abs(grad) < 1e-3) nt = t + .01; // exactly at 90° the gradient vanishes
        ctx.set('th', ((nt * 180 / Math.PI) % 180 + 180) % 180); this.draw(ctx);
      },
      draw(ctx) {
        const P = ctx.plot, t = ctx.vals.th * Math.PI / 180, u = [Math.cos(t), Math.sin(t)];
        P.clear();
        let err = 0;
        P.custom((g, p) => {
          g.strokeStyle = 'rgba(142,153,189,.35)'; g.lineWidth = 1;
          for (const [x, y] of ctx.X) { const z = x * u[0] + y * u[1]; g.beginPath(); g.moveTo(p.sx(x), p.sy(y)); g.lineTo(p.sx(z * u[0]), p.sy(z * u[1])); g.stroke(); }
        });
        for (const [x, y] of ctx.X) { const z = x * u[0] + y * u[1]; err += (x - z * u[0]) ** 2 + (y - z * u[1]) ** 2; }
        P.scatter(ctx.X, { color: '#cfd6ef', r: 2 });
        P.line([[-12 * u[0], -12 * u[1]], [12 * u[0], 12 * u[1]]], { color: HEX.ball, width: 2.4 });
        P.draw();
        const E = 1 + 8 * Math.sin(t) ** 2, ch = ctx.charts.e;
        ch.clear().fn(d => 1 + 8 * Math.sin(d * Math.PI / 180) ** 2, { color: HEX.pos, width: 2 })
          .hline(1, { color: HEX.good, dash: [4, 4], text: 'PC1: E = λ₂ = 1' }).hline(9, { color: HEX.neg, dash: [4, 4], text: 'PC2: E = λ₁ = 9' });
        ch.scatter([[ctx.vals.th, E]], { color: HEX.ball, r: 5 }); ch.draw();
        ctx.stat(`θ = ${ctx.vals.th.toFixed(0)}° · E(θ) = ${E.toFixed(2)} · sample error ${(err / ctx.X.length).toFixed(2)}`);
      },
    },
    {
      id: 'equivalent', name: 'Many weights, one map', title: 'A valley of equivalent solutions',
      caption: 'AB = (AC)(C⁻¹B): every point on the green curve gives the same projection. The floor is E = 1, the variance PCA must drop.',
      note: 'Slice A = (a, 0)ᵀ, B = (b, 0) of the untied network. Height shows E − 1.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'ab = 1: same map W = AB'], [HEX.neg, 'λ < 0'], [HEX.pos, 'λ > 0']],
      action: 'Drop balls', cam: [0, 4.4, 5.6], tgt: [0, .4, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(untied);
        s.tube(untied, hyperbola(1, 1.6, 1), C.good); s.tube(untied, hyperbola(1, 1.6, -1), C.good);
        s.marker(untied, 0, 0, C.white); s.eigenLines(untied, 0, 0, { len: .8 });
        ctx.L = statLabels(s, [['', untied.W(0, 0, 0).setY(1.9)]]);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, r = rng(Date.now() % 1e6); s.clearBalls();
        for (let i = 0; i < 6; i++) s.ball(untied, { start: [r.uniform(-1.4, 1.4), r.uniform(-1.4, 1.4)], step: opt.gd(.005), speed: 3, stop: b => Math.hypot(...untied.g(...b.p)) < 1e-5 });
      },
      frame(ctx) {
        const done = ctx.stage.balls.filter(b => b.done);
        ctx.L[0].stat(done.length ? done.map(b => `a=${b.p[0].toFixed(2)}, b=${b.p[1].toFixed(2)}`).slice(0, 3).join('   ') : '');
        ctx.stat(done.length ? `${done.length} landed · every one has ab = ${done.map(b => (b.p[0] * b.p[1]).toFixed(3)).join(', ')} and E = 1` : 'rolling…');
      },
    },
  ],
};
