// Chapter 8 (survey Sections III-D to III-H, Appendix A-B). New visual 'adamw' (Eqs. 27-28; numbers in
// scripts/check-math/guide3_numbers.py); reused: duchi2011adaptive 'sparse', crash-course 'bias' and 'adamfail' (course_adaptive.py).
import { HEX } from '../../src/lib/atlas3d.js';
import ad from '../../course/adaptive/viz.js';
import { paperTab } from '../lib.js';

const EPS = 1e-8;
const adamw = {
  title: 'L2 inside Adam is not weight decay: the penalty inside the adaptive step against AdamW, Eq. (28)',
  caption: 'Each coordinate has its own adaptive scale D = √v̂ + ε, set by the size of its gradients. Putting the penalty λθ inside the adaptive step (survey Section III-H) divides the decay by D too, so a coordinate with large gradients is barely decayed and one with tiny gradients is decayed hard. AdamW (Eq. 28) shrinks every weight by the same factor 1 − ηλ.',
  note: 'Schematic, as in the survey: D_t held fixed for one step, ε = 10⁻⁸. Vertical axis: the fraction of a weight removed by the decay term in one step, capped at 1 (a cap of 1 would mean the decay alone overshoots zero).',
  legend: [[HEX.ball, 'L2 penalty inside Adam, ηλ/D (survey Section III-H)'], [HEX.good, 'AdamW, ηλ (Eq. 28)']],
  mode: '2d',
  plot: { xlabel: 'gradient scale of the coordinate, √v̂ (log scale)', ylabel: 'fraction of the weight removed per step', logx: true, logy: true, xlim: [1e-6, 10], ylim: [1e-9, 2], legend: false },
  controls: [
    { id: 'le', label: 'learning rate η', min: -5, max: -1, step: .05, value: -3, fmt: v => (10 ** v).toExponential(1) },
    { id: 'll', label: 'decay coefficient λ', min: -4, max: 0, step: .05, value: -2, fmt: v => (10 ** v).toExponential(1) },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const eta = 10 ** ctx.vals.le, lam = 10 ** ctx.vals.ll, P = ctx.plot.clear();
    const xs = Array.from({ length: 300 }, (_, i) => 10 ** (-6 + 7 * i / 299));
    P.line(xs.map(s => [s, Math.min(eta * lam / (s + EPS), 1)]), { color: HEX.ball, width: 2.2 });
    P.line(xs.map(s => [s, eta * lam]), { color: HEX.good, width: 2.2 });
    P.hline(1, { color: 'rgba(255,255,255,.3)', dash: [3, 4], text: 'whole weight removed' });
    const marks = [1e-4, 1e-2, 1];
    P.scatter(marks.map(s => [s, Math.min(eta * lam / (s + EPS), 1)]), { color: HEX.ball, r: 4 });
    P.draw();
    const f = s => Math.min(eta * lam / (s + EPS), 1);
    ctx.stat(`η = ${eta.toExponential(1)}, λ = ${lam.toExponential(1)} · AdamW removes ηλ = ${(eta * lam).toExponential(2)} of every weight per step · L2-in-Adam removes ${f(1e-4).toExponential(2)} at √v̂ = 10⁻⁴, ${f(1e-2).toExponential(2)} at 10⁻², ${f(1).toExponential(2)} at 1: a factor ${(f(1e-4) / f(1)).toExponential(1)} between rarely and often updated coordinates`);
  },
};

export default { sparse: await paperTab('duchi2011adaptive', 'sparse'), bias: ad.bias, adamfail: ad.adamfail, adamw };
