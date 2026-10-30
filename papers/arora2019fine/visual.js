// Arora, Du, Hu, Li, Wang 2019 — training speed and generalization of a wide two-layer ReLU net are read off the
// spectrum of H∞ and the label projections. All curves, spectra and the weight-space slice come from
// scripts/check-math/arora2019fine.py (data.json): n = 60 unit vectors in ℝ¹⁰, m = 8 000 neurons, η = 1/λ_max(H∞).
import { gridDef, opt, C, HEX } from '../../src/lib/atlas3d.js';

export default {
  tabs: [
    {
      id: 'valley', name: 'Fast and slow directions', title: 'In weight space, GD drops fast along top kernel directions and crawls along the bottom ones',
      caption: 'The real training loss of the 8 000-neuron network on a plane through its initialization W₀, spanned by the weight change that moves the outputs along the top eigenvector of H∞ (left–right) and the one for the bottom eigenvector (front–back), both of unit length. The valley is steep one way and almost flat the other (λ_max/λ_min ≈ 40). The ball runs GD on this slice: it reaches the valley floor at once, then creeps.',
      note: 'Structured labels y = sign(x₁). The bottom axis spans 3× the top axis (square display); the valley floor does not reach zero because the other 58 directions are frozen on this slice. The ball optimizes the slice only.',
      legend: [[HEX.ball, 'GD on the slice'], [HEX.opt3, 'initialization W₀']],
      cam: [3.4, 3.6, 3.6], tgt: [0, .2, 0], action: 'Run again',
      charts: [{ id: 'L', title: 'loss along the ball’s path', xlabel: 'GD step on the slice', ylabel: 'loss', logy: true, h: 200 }],
      build(ctx) {
        const D = ctx.data, s = ctx.stage;
        ctx.F = gridDef(D.slice, D.dom, { h: v => v, square: true });
        s.surface(ctx.F); s.marker(ctx.F, 0, 0, C.opt3, .09);
        const h = (D.dom[1] - D.dom[0]) / 30, Lf = ctx.F.L, c = (Lf(h, 0) - 2 * Lf(0, 0) + Lf(-h, 0)) / (h * h);
        ctx.lr = .9 / c;
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.b = s.ball(ctx.F, { start: [0, 0], step: opt.gd(ctx.lr), speed: .25, maxSteps: 300 });
        ctx.b.h = [[0, ctx.b.loss()]]; ctx.b.onStep = z => z.h.push([z.steps, z.loss()]);
      },
      frame(ctx) {
        const b = ctx.b; if (!b) return;
        ctx.charts.L.clear().set({ xlim: [0, Math.max(20, b.steps)] }).line(b.h, { color: HEX.ball, width: 2 }).draw();
        ctx.stat(`step ${b.steps} · position (top, bottom) = (${b.p[0].toFixed(3)}, ${b.p[1].toFixed(3)}) · loss ${b.loss().toFixed(4)} · λ_max(H∞) = ${ctx.data.lam[ctx.data.lam.length - 1].toFixed(2)}, λ_min = ${ctx.data.lam[0].toFixed(3)}`);
      },
    },
    {
      id: 'labels', name: 'True vs random labels', mode: '2d', title: 'Labels aligned with the top of the kernel spectrum train fast and generalize',
      caption: 'Solid: training loss of the real network under GD for structured labels y = sign(x₁) (coral) and random ±1 labels (violet). Dashed: Theorem 4.1’s prediction ½‖(I − ηH∞)ᵏ(y − u₀)‖², computed from the kernel alone. Right: where the labels sit in the spectrum, (vᵢᵀy)² against λᵢ: structured labels load the large eigenvalues, random labels spread into the small ones, which are learned slowly and cost a lot in yᵀ(H∞)⁻¹y.',
      note: 'n = 60, d = 10, m = 8 000, w_r ~ N(0, I), a_r = ±1 fixed, η = 1/λ_max(H∞). We keep the initial output u₀ in the prediction (the theorem takes a small init scale so that u₀ ≈ 0).',
      hint: '', legend: [[HEX.ball, 'structured labels'], [HEX.opt2, 'random labels'], [HEX.muted, 'Theorem 4.1 prediction (dashed)']],
      plot: { xlabel: 'GD iteration k', ylabel: 'training loss ½‖y − u(k)‖²', logy: true, xlim: [0, 300], ylim: [1e-6, 100], legend: false },
      charts: [{ id: 'sp', title: 'label energy per eigenvector', xlabel: 'eigenvalue λᵢ of H∞', ylabel: '(vᵢᵀy)²', logx: true, logy: true, ylim: [1e-4, 60], h: 250 }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear(), pts = a => a.map((v, i) => [i, v]);
        P.line(pts(D.Pr), { color: HEX.opt2, dash: [5, 4], width: 1.3 }).line(pts(D.Ps), { color: HEX.ball, dash: [5, 4], width: 1.3 })
          .line(pts(D.Lr), { color: HEX.opt2, width: 2 }).line(pts(D.Ls), { color: HEX.ball, width: 2.4 }).draw();
        ctx.charts.sp.clear().scatter(D.lam.map((l, i) => [l, Math.max(D.pr[i], 1e-4)]), { color: HEX.opt2, r: 2.6 }).scatter(D.lam.map((l, i) => [l, Math.max(D.ps[i], 1e-4)]), { color: HEX.ball, r: 2.6 }).draw();
        const bnd = c => Math.sqrt(2 * c / D.n).toFixed(2);
        ctx.stat(`yᵀ(H∞)⁻¹y: structured ${D.cs.toFixed(1)}, random ${D.cr.toFixed(1)} · Theorem 5.1 leading term √(2yᵀ(H∞)⁻¹y/n): ${bnd(D.cs)} vs ${bnd(D.cr)} · loss at k = 100: ${D.Ls[100].toExponential(1)} vs ${D.Lr[100].toExponential(1)}`);
      },
    },
  ],
};
