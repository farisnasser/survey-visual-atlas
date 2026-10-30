// Zou & Gu 2019 — over-parameterized ReLU nets: linear convergence of GD and a short trajectory from initialization.
// Every number and grid comes from scripts/check-math/zou2019improved.py (data.json): a one-hidden-layer ReLU net,
// W rows ~ N(0, 2/m), v ~ N(0, 1/d) (Algorithm 1), n = 10 points with ‖x‖ = 1 and x_d = 0.5, step 1/λ_max(Gram).
import { gridDef, opt, C, HEX } from '../../src/lib/atlas3d.js';

const WCOL = { 32: HEX.opt3, 128: '#7cc4ff', 512: HEX.opt2, 2048: HEX.flat, 8192: HEX.ball };

export default {
  tabs: [
    {
      id: 'slice', name: 'Loss near init', title: 'Around a wide network’s initialization, the loss along the training path is a clean bowl',
      caption: 'Training loss on the plane through the initial weights W₀ spanned by the actual GD displacement D₁ = W_T − W₀ (left–right) and a random direction D₂ of the same norm. The white ring is W₀, the green ring is where GD ended (loss ≈ 0). At width 8 192 the slice is a smooth convex bowl; at width 32 it is visibly non-convex. The ball runs plain GD on this 2-D slice.',
      note: 'Real loss of a one-hidden-layer ReLU net on 10 points, evaluated on a 41 × 41 grid; heights log-scaled. The ball optimizes the slice only, not the full network.',
      legend: [[HEX.opt3, 'initialization W₀'], [HEX.good, 'end of training W_T'], [HEX.ball, 'GD on the slice']],
      cam: [3.6, 3.2, 3.8], tgt: [0, .2, 0],
      controls: [{ id: 'm', type: 'seg', label: 'width m', value: 8192, options: [[32, '32'], [8192, '8 192']] }],
      action: 'Run again',
      build(ctx) {
        const D = ctx.data, m = ctx.vals.m, s = ctx.stage; s.clear();
        ctx.F = gridDef(D[`slice${m}`], [-.5, 1.5, -1, 1], { h: v => Math.log10(v + 1e-3) });
        s.surface(ctx.F); s.marker(ctx.F, 0, 0, C.opt3, .1); s.marker(ctx.F, 1, 0, C.good, .1);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, F = ctx.F; s.clearBalls();
        ctx.b = s.ball(F, { start: [0, 0], step: opt.gd(ctx.vals.m === 32 ? .02 : .02), speed: .6, maxSteps: 400 });
      },
      onControl(ctx) { this.build(ctx); },
      frame(ctx) {
        const D = ctx.data, m = ctx.vals.m, b = ctx.b; if (!b) return;
        ctx.stat(`width ${m} · slice along the training direction is ${D[`convex${m}`] ? 'convex' : 'not convex'} · full-network GD: loss ${D[`loss${m}`][0].toFixed(2)} → < 10⁻⁶ of it after ${D[`k${m}`]} steps · ball on the slice at (${b.p.map(v => v.toFixed(2)).join(', ')}), loss ${b.loss().toFixed(3)}`);
      },
    },
    {
      id: 'stay', name: 'Wider = closer', mode: '2d', title: 'Wider networks converge linearly while barely moving their weights',
      caption: 'Left: training loss of full-batch GD for five widths (log scale), each a straight line = linear convergence, the content of the gradient lower bound. Bars: how far the weights travel, ‖W_T − W₀‖₂/‖W₀‖₂, and the fraction of ReLU on/off patterns that flip. Both fall roughly like 1/√m — the trajectory stays in the region where the gradient lower bound holds, which is what the proof needs.',
      note: 'Step η = 1/λ_max of the Gram matrix at init, which scales like 1/m (Theorem 3.3 takes η = O(k/(L²m))). 3 000 steps; widths 32 to 8 192.',
      hint: '', legend: Object.entries(WCOL).map(([m, c]) => [c, `m = ${m}`]),
      controls: [{ id: 'show', type: 'seg', label: 'right panel', value: 'mv', options: [['mv', 'weight movement'], ['fl', 'flipped patterns']] }],
      plot: { xlabel: 'GD step', ylabel: 'training loss', logy: true, xlim: [0, 600], ylim: [1e-14, 30], legend: false },
      charts: [{ id: 'bar', title: 'after training', xlabel: 'width m', ylabel: '', h: 240, legend: false }],
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        for (const m of D.widths) P.line(D[`loss${m}`].map((v, i) => [i * 10, v]), { color: WCOL[m], width: 2 });
        P.draw();
        const key = ctx.vals.show;
        ctx.charts.bar.clear().set({ ylabel: key === 'mv' ? '‖W_T − W₀‖₂ / ‖W₀‖₂' : 'flipped activations', xfmt: i => D.widths[Math.round(i)] ?? '' })
          .bars(D.widths.map(m => ({ y: D[`${key}${m}`], color: WCOL[m] }))).draw();
        ctx.stat(D.widths.map(m => `m = ${m}: move ${D[`mv${m}`].toFixed(3)}, flips ${(100 * D[`fl${m}`]).toFixed(1)}%`).join(' · '));
      },
    },
    {
      id: 'width', name: 'Width needed', mode: '2d', title: 'How wide must the network be for the proof to work?',
      caption: 'Sufficient width for GD on L-hidden-layer ReLU nets to reach any ε training loss (Table 1, orders). Allen-Zhu et al. needed Ω̃(k n²⁴ L¹² / φ⁸); this paper Ω̃(k n⁸ L¹² / φ⁴), a factor n¹⁶/φ⁴ less, with iteration count O(n²L² log(1/ε)/φ) instead of O(n⁶L² log(1/ε)/φ²).',
      note: 'Orders only, k = 1, L = 3, logs dropped. These are sufficient conditions from the proofs; real networks train at far smaller widths (here 32 already fits 10 points).',
      hint: '', legend: [[HEX.opt2, 'Allen-Zhu et al. 2018 n²⁴/φ⁸'], [HEX.ball, 'Zou & Gu 2019 n⁸/φ⁴']],
      controls: [{ id: 'phi', label: 'separation φ', min: -3, max: 0, step: .25, value: -1, fmt: v => `10^${(+v).toFixed(2)}` }],
      plot: { xlabel: 'training examples n', ylabel: 'sufficient width m', logx: true, logy: true, xlim: [10, 1e6], ylim: [1, 1e200], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const ph = 10 ** ctx.vals.phi, L12 = 3 ** 12;
        ctx.plot.clear().fn(n => n ** 24 * L12 / ph ** 8, { color: HEX.opt2, width: 2 }).fn(n => n ** 8 * L12 / ph ** 4, { color: HEX.ball, width: 2.6 }).draw();
        const n = 1e4;
        ctx.stat(`n = 10⁴, φ = 10^${ctx.vals.phi.toFixed(2)}, L = 3: Allen-Zhu et al. m ≳ ${(n ** 24 * L12 / ph ** 8).toExponential(1)} · Zou & Gu m ≳ ${(n ** 8 * L12 / ph ** 4).toExponential(1)}`);
      },
    },
  ],
};
