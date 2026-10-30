// Keskar et al. 2017 — large batches converge to sharp minimizers; small-batch gradient noise pushes iterates out of sharp basins.
// A finite-sum toy of that explanation (not a network): scripts/check-math/keskar2017large.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const SF = .55, SS = .12, ETA = .02;
const F = def({
  L: (x, y) => .02 * (x * x + y * y) - Math.exp(-((x + 1) ** 2 + y * y) / (2 * SF * SF)) - Math.exp(-((x - 1) ** 2 + y * y) / (2 * SS * SS)),
  g: (x, y) => { const a = Math.exp(-((x + 1) ** 2 + y * y) / (2 * SF * SF)) / (SF * SF), b = Math.exp(-((x - 1) ** 2 + y * y) / (2 * SS * SS)) / (SS * SS); return [.04 * x + a * (x + 1) + b * (x - 1), .04 * y + a * y + b * y]; },
  dom: [-2.2, 2.2, -1.5, 1.5], h: v => v, fit: 1.4,
});
/** minibatch SGD: f(w) = (1/n) Σᵢ [F(w) + vᵢ·w] with Σ vᵢ = 0; a batch of B indices (with replacement) adds the mean of B tilts */
const sgd = (B, V, R) => (p, b, d) => {
  const g = d.g(p[0], p[1]); let nx = 0, ny = 0;
  for (let k = 0; k < B; k++) { const v = V[Math.floor(R() * V.length)]; nx += v[0]; ny += v[1]; }
  return [p[0] - ETA * (g[0] + nx / B), p[1] - ETA * (g[1] + ny / B)];
};

export default {
  tabs: [
    {
      id: 'batch', name: 'Batch size', title: 'Small-batch noise shakes SGD out of the sharp basin',
      caption: 'Two equally deep minima, a wide one on the left and a narrow one on the right (the flat-minima page’s landscape). The training loss is a sum of 64 “samples”, each tilting the landscape slightly; the tilts cancel on average, so the full-batch loss is exactly the surface shown. SGD with a small batch gets noisy gradients and cannot stay in the narrow basin; with larger batches the noise shrinks and runs that land in the sharp basin stay there, as Keskar et al. argue for real networks.',
      note: 'Each sample i adds vᵢ·w with vᵢ ~ N(0, 5²I) recentred; step 0.02, 30 starts uniform in [−2, 2] × [−1.2, 1.2]. A model of the paper’s explanation, not its experiments (fully connected and convolutional nets on MNIST, TIMIT, CIFAR).',
      legend: [[HEX.ball, 'minibatch SGD'], [HEX.good, 'minima (flat left, sharp right)']],
      action: 'Release balls', cam: [0, 4.2, 4.4], tgt: [0, .2, 0],
      controls: [{ id: 'B', type: 'seg', label: 'batch size B', value: 2, options: [[2, '2'], [4, '4'], [64, '64 (full)']] }],
      build(ctx) { const s = ctx.stage, D = ctx.data; s.surface(F); s.marker(F, ...D.mF, C.good); s.marker(F, ...D.mS, C.good, .06); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, B = +ctx.vals.B, R = rng(5), N = rng(7 + B); s.clearBalls();
        ctx.bs = Array.from({ length: 30 }, () => s.ball(F, { start: [R.uniform(-2, 2), R.uniform(-1.2, 1.2)], step: sgd(B, ctx.data.V, N), speed: 8, maxSteps: 8000, r: .045 }));
      },
      onControl(ctx) { this.drop(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const D = ctx.data, nF = ctx.bs.filter(b => Math.hypot(b.p[0] - D.mF[0], b.p[1] - D.mF[1]) < .4).length, nS = ctx.bs.filter(b => Math.hypot(b.p[0] - D.mS[0], b.p[1] - D.mS[1]) < .2).length;
        const off = D.rows.find(q => q[0] === +ctx.vals.B);
        ctx.stat(`B = ${ctx.vals.B} · step ${ctx.bs[0].steps} · in the flat basin ${nF}/30, in the sharp basin ${nS}/30 · offline, 400 runs × 20 000 steps: flat ${Math.round(100 * off[1])}%, sharp ${Math.round(100 * off[2])}% · sharpness metric (4), ε = 10⁻³: flat ${D.phi[0].toFixed(3)}, sharp ${D.phi[1].toFixed(3)}`);
      },
    },
    {
      id: 'threshold', name: 'Batch threshold', mode: '2d', title: 'Above a threshold batch size, a fixed share of runs ends sharp',
      caption: 'Where 400 SGD runs end after 20 000 steps, against the batch size. Up to B = 2 almost none end in the sharp basin; from B = 3 on about a fifth do, and larger batches change little. Keskar et al. likewise report a batch size beyond which test accuracy drops and sharpness jumps (their Fig. 4). With B = 1 the noise is so strong that most runs have not settled anywhere.',
      note: 'Offline, same toy and step as the first tab, no step decay.',
      hint: '', legend: [[HEX.good, 'ends in the flat basin'], [HEX.neg, 'ends in the sharp basin']],
      plot: { xlabel: 'batch size B', ylabel: 'fraction of runs', logx: true, xlim: [.8, 80], ylim: [0, 1], legend: false },
      build(ctx) {
        const R = ctx.data.rows;
        ctx.plot.clear().line(R.map(q => [q[0], q[1]]), { color: HEX.good, width: 2.2, dots: 4 }).line(R.map(q => [q[0], q[2]]), { color: HEX.neg, width: 2.2, dots: 4 }).draw();
        ctx.stat(R.map(q => `B = ${q[0]}: ${Math.round(100 * q[1])}% flat, ${Math.round(100 * q[2])}% sharp`).join(' · '));
      },
    },
  ],
};
