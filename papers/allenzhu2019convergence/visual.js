// Allen-Zhu, Li, Song 2019 — deep ReLU nets (fixed A, B; trained W1..W3). All curves and the slice are precomputed by
// scripts/check-math/allenzhu2019convergence.py (too large to train live) and loaded from data.json.
import { gridDef, C, HEX, segPts } from '../../src/lib/atlas3d.js';

const COLS = { 16: HEX.opt2, 64: HEX.flat, 256: HEX.pos, 1024: HEX.ball };

export default {
  tabs: [
    {
      id: 'slice', name: 'Gradient vs curvature', title: 'Negative curvature is there, the gradient wins',
      caption: 'The real training loss of a 3-layer ReLU net (width 256) around a point on its GD path, along the negative gradient (coral) and the most negative curvature direction (red). Moving downhill drops the loss fast; the negative curvature barely matters.',
      note: 'Analogue of the paper’s Fig. 1 (CIFAR-10 there; 10 random points in R¹⁰ here). 41×41 grid computed offline; directions normalized.',
      legend: [[HEX.ball, '−∇F direction'], [HEX.neg, 'most negative curvature direction'], ['#ffffff', 'current point W_t']],
      cam: [3.4, 3.3, 3.9], tgt: [0, .6, 0],
      build(ctx) {
        const S = ctx.data.slice, d = gridDef(S.F, [S.a[0], S.a[1], S.b[0], S.b[1]], { h: v => v, square: true });
        const s = ctx.stage; s.surface(d);
        const am = S.a[0] + (S.a[1] - S.a[0]) * S.F[20].indexOf(Math.min(...S.F[20])) / (S.n - 1);
        s.tube(d, segPts(0, 0, am, 0), C.ball, .022); s.tube(d, segPts(0, S.b[0] * .9, 0, S.b[1] * .9), C.neg, .016);
        s.marker(d, 0, 0, C.white, .09); s.dot(d.W(am, 0, .05), C.ball, .06);
        s.label(`F(W_t) = ${S.F0.toFixed(2)}`, d.W(0, 0, .35), '', HEX.white);
        s.label(`min along −∇F: ${Math.min(...S.F[20]).toFixed(2)}`, d.W(am, 0, .35), '', HEX.ball);
        s.label(`λ_min ≈ ${S.lam_min.toFixed(2)}`, d.W(0, S.b[1] * .8, .3), '', HEX.neg);
        ctx.stat(`Hessian at W_t: λ_max ≈ ${S.lam_max.toFixed(0)}, λ_min ≈ ${S.lam_min.toFixed(3)} — strictly negative but tiny relative to the gradient’s pull`);
      },
    },
    {
      id: 'width', name: 'Width', mode: '2d', title: 'Wider nets: geometric decrease, weights barely move',
      caption: 'Full-batch GD on a 3-layer ReLU net with 10 training points, for four widths m (step size ∝ 1/m as in Theorem 1). Wider nets converge geometrically and stay near initialization.',
      note: 'Right: the Theorem 3 quantity ‖∇F‖² / (m·F) along each run (bounded away from 0 ⇒ no bad critical point nearby), and relative weight movement.',
      hint: '', legend: [],
      plot: { logy: true, xlabel: 'GD iteration', ylabel: 'training loss F', xlim: [0, 600] },
      charts: [
        { id: 'ratio', title: '‖∇F‖² / (m·F) along the run', xlabel: 'iteration', ylabel: 'ratio', logy: true, xlim: [0, 600], h: 200 },
        { id: 'mv', title: 'weight movement ‖W_t − W_0‖ / ‖W_0‖', xlabel: 'iteration', ylabel: 'relative change', logy: true, xlim: [0, 600], h: 200 },
      ],
      build(ctx) {
        const W = ctx.data.widths, P = ctx.plot.clear(), R = ctx.charts.ratio.clear(), M = ctx.charts.mv.clear();
        for (const m of Object.keys(W)) {
          const c = COLS[m], s = x => x.map((v, k) => [k, v]);
          P.line(s(W[m].loss), { color: c, width: 2, label: `m = ${m}` });
          R.line(s(W[m].ratio), { color: c, width: 1.6 }); M.line(s(W[m].move).slice(1), { color: c, width: 1.6 });
        }
        P.draw(); R.draw(); M.draw();
        ctx.stat(Object.keys(W).map(m => `m=${m}: F₆₀₀ = ${W[m].loss[599].toExponential(1)}`).join(' · '));
      },
    },
    {
      id: 'separation', name: 'Data separation δ', mode: '2d', title: 'Closer inputs, slower training',
      caption: 'Two unit-norm inputs with opposite labels at distance δ, width 256. The theorem’s iteration count grows polynomially in 1/δ; identical inputs with different labels can never be fit.',
      note: 'Same network family as the Width tab; curves precomputed.',
      hint: '', legend: [],
      plot: { logy: true, xlabel: 'GD iteration', ylabel: 'training loss F', xlim: [0, 3000] },
      build(ctx) {
        const S = ctx.data.separation, P = ctx.plot.clear(), cols = [HEX.ball, HEX.flat, HEX.pos, HEX.opt2];
        Object.keys(S).forEach((k, i) => P.line(S[k].loss.map((v, j) => [j * 5, v]), { color: cols[i], width: 2, label: `δ = ${k}` }));
        P.hline(1e-3, { color: HEX.muted, dash: [4, 4], text: 'F = 10⁻³' }).draw();
        ctx.stat(Object.keys(S).map(k => `δ=${k}: ${S[k].hit ?? '> 3000'} iterations to F < 10⁻³`).join(' · '));
      },
    },
  ],
};
