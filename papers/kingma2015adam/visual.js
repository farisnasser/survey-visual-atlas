// Kingma & Ba 2015 — Adam: bias-corrected moving averages of g and g², step α·m̂/(√v̂ + ε).
// Step-size facts and the toy comparison are checked in scripts/check-math/kingma2015adam.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const A = [20, .05], SIG = .5, START = [1.5, 1.8], B1 = .9, B2 = .999;
const surf = def({ L: (x, y) => .5 * (A[0] * x * x + A[1] * y * y), g: (x, y) => [A[0] * x, A[1] * y], dom: [-2, 2, -2, 2], h: v => Math.log1p(v) });
const noisy = (R, p) => { const g = surf.g(...p); return [g[0] + SIG * R.normal(), g[1] + SIG * R.normal()]; };

export default {
  tabs: [
    {
      id: 'scales', name: 'Badly scaled', title: 'Per-coordinate scaling crosses the flat direction',
      caption: 'A noisy quadratic whose curvatures differ by a factor 400. SGD (violet) and SGD with momentum (white) must use a step small enough for the steep direction, so they creep along the flat one. Adam (coral) divides each coordinate’s momentum by the root of its own second-moment estimate: its steps are about α in every coordinate whatever the gradient scale, so it crosses the flat direction quickly.',
      note: 'f = ½(20x² + 0.05y²) with Gaussian gradient noise σ = 0.5. Steps are each optimizer’s best on a small grid in the check script: SGD 0.06, momentum (β = 0.9) 0.01, Adam α = 0.03 (β1 = 0.9, β2 = 0.999).',
      legend: [[HEX.opt2, 'SGD'], [HEX.opt3, 'SGD + momentum'], [HEX.ball, 'Adam'], [HEX.good, 'minimum']],
      action: 'Restart', cam: [0, 4.4, 4.2], tgt: [0, .1, 0],
      charts: [{ id: 'f', title: 'loss (log)', xlabel: 'step', ylabel: 'f', logy: true, ylim: [1e-4, 30], xlim: [0, 600], h: 230, legend: false }],
      build(ctx) { ctx.stage.surface(surf); ctx.stage.marker(surf, 0, 0, C.good, .1); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, Rs = rng(1), Rm = rng(2), Ra = rng(3), lr = ctx.data.best; s.clearBalls(); ctx.H = { s: [], m: [], a: [] };
        const rec = k => b => ctx.H[k].push([b.steps, surf.L(...b.p)]);
        ctx.bs = s.ball(surf, { start: START, speed: 1, maxSteps: 600, color: C.opt2, step: p => { const g = noisy(Rs, p); return [p[0] - lr.sgd * g[0], p[1] - lr.sgd * g[1]]; } }); ctx.bs.onStep = rec('s');
        ctx.bm = s.ball(surf, { start: START, speed: 1, maxSteps: 600, color: C.opt3, step: (p, b) => { const g = noisy(Rm, p); b.v = b.v || [0, 0]; b.v = [.9 * b.v[0] - lr.mom * g[0], .9 * b.v[1] - lr.mom * g[1]]; return [p[0] + b.v[0], p[1] + b.v[1]]; } }); ctx.bm.onStep = rec('m');
        ctx.ba = s.ball(surf, { start: START, speed: 1, maxSteps: 600, step: (p, b) => {
          const g = noisy(Ra, p), t = (b.t = (b.t || 0) + 1); b.m = b.m || [0, 0]; b.v = b.v || [0, 0]; const q = p.slice();
          for (let i = 0; i < 2; i++) { b.m[i] = B1 * b.m[i] + (1 - B1) * g[i]; b.v[i] = B2 * b.v[i] + (1 - B2) * g[i] * g[i]; q[i] -= lr.adam * (b.m[i] / (1 - B1 ** t)) / (Math.sqrt(b.v[i] / (1 - B2 ** t)) + 1e-8); }
          return q;
        } }); ctx.ba.onStep = rec('a');
      },
      frame(ctx) {
        const P = ctx.charts.f.clear(); [['s', HEX.opt2], ['m', HEX.opt3], ['a', HEX.ball]].forEach(([k, c]) => P.line(ctx.H[k].map(h => [h[0], Math.max(h[1], 1e-4)]), { color: c, width: 1.4 })); P.draw();
        const late = k => { const h = ctx.H[k].slice(-100); return h.length ? (h.reduce((a, x) => a + x[1], 0) / h.length).toFixed(4) : '–'; };
        ctx.stat(`step ${ctx.ba.steps} · mean loss over the last 100 steps: SGD ${late('s')} · momentum ${late('m')} · Adam ${late('a')}`);
      },
    },
    {
      id: 'bias', name: 'Bias correction', mode: '2d', title: 'Why divide by 1 − β1ᵗ and 1 − β2ᵗ',
      caption: 'Adam’s step size in units of α, for β1 = 0.9 and a chosen β2. With bias correction and a constant gradient the step is exactly α (coral). Without it the moving averages start at zero and the second one lags far more, so the first steps are up to 6.5α for β2 = 0.999 (violet), the instability the paper sees in its VAE experiment without correction. In the most sparse case, one gradient after a long run of zeros, even corrected Adam can step up to α(1 − β1)/√(1 − β2) (white).',
      note: 'Exact formulas: uncorrected (1 − β1ᵗ)/√(1 − β2ᵗ); single spike at step t: (1 − β1)/√(1 − β2) · √(1 − β2ᵗ)/(1 − β1ᵗ). Checked by simulation.',
      hint: '', legend: [[HEX.ball, 'corrected, constant g'], [HEX.opt2, 'uncorrected, constant g'], [HEX.opt3, 'corrected, single spike at t']],
      controls: [{ id: 'b2', type: 'seg', label: 'β2', value: .999, options: [[.99, '0.99'], [.999, '0.999'], [.9999, '0.9999']] }],
      plot: { xlabel: 'step t (log)', ylabel: '|Δθ| / α', logx: true, xlim: [1, 1e5], ylim: [0, 22], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const b2 = +ctx.vals.b2, unc = t => (1 - B1 ** t) / Math.sqrt(1 - b2 ** t), spike = t => (1 - B1) / Math.sqrt(1 - b2) * Math.sqrt(1 - b2 ** t) / (1 - B1 ** t);
        const P = ctx.plot.clear().hline(1, { color: HEX.ball, width: 2.4 }).fn(unc, { color: HEX.opt2, width: 2.2 }).fn(spike, { color: HEX.opt3, width: 1.8, dash: [5, 3] })
          .hline((1 - B1) / Math.sqrt(1 - b2), { color: HEX.muted, dash: [2, 4], text: '(1 − β1)/√(1 − β2)' }).draw();
        let mx = 0, at = 1; for (let t = 1; t < 1e5; t++) { const u = unc(t); if (u > mx) { mx = u; at = t; } }
        ctx.stat(`β2 = ${b2}: uncorrected step peaks at ${mx.toFixed(2)}α (t = ${at}); sparse-case bound (1 − β1)/√(1 − β2) = ${((1 - B1) / Math.sqrt(1 - b2)).toFixed(2)}`);
      },
    },
  ],
};
