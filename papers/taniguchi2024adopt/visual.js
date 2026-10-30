// Taniguchi et al. 2024 — ADOPT: normalize the fresh gradient by the previous second moment, then apply momentum.
// Converges for any β2 on the paper's toy (Eq. 17) where Adam fails. Checked in scripts/check-math/taniguchi2024adopt.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const B1 = .9, EPS = 1e-8;
// theta1: the paper's toy f(theta) = theta on [-1, 1] with gradient k^2 w.p. 1/k, -k otherwise; theta2: a plain bowl
const surf = def({ L: (a, b) => a + b * b / 2, g: (a, b) => [1, b], dom: [-1, 1, -1, 1], h: v => v, N: 60 });
const lr = t => .01 / Math.sqrt(1 + .01 * t);
const clip = p => [Math.max(-1, Math.min(1, p[0])), Math.max(-1, Math.min(1, p[1]))];
function sampler(k, seed) { const R = rng(seed); return p => [R() < 1 / k ? k * k : -k, p[1]]; }
function adam(k, b2, ams, seed) {
  const sg = sampler(k, seed);
  return (p, b) => {
    const g = sg(p), t = (b.t = (b.t || 0) + 1); b.m = b.m || [0, 0]; b.v = b.v || [0, 0]; b.vm = b.vm || [0, 0]; const q = p.slice();
    for (let i = 0; i < 2; i++) {
      b.m[i] = B1 * b.m[i] + (1 - B1) * g[i]; b.v[i] = b2 * b.v[i] + (1 - b2) * g[i] * g[i];
      let vh = b.v[i] / (1 - b2 ** t); if (ams) { b.vm[i] = Math.max(b.vm[i], vh); vh = b.vm[i]; }
      q[i] -= lr(t) * (b.m[i] / (1 - B1 ** t)) / (Math.sqrt(vh) + EPS);
    }
    return clip(q);
  };
}
function adopt(k, b2, seed) {
  const sg = sampler(k, seed);
  return (p, b) => {
    if (!b.init) { const g0 = sg(p), g1 = sg(p); b.v = g0.map(x => x * x); b.m = g1.map((x, i) => x / Math.max(Math.sqrt(b.v[i]), EPS)); b.gn = g1; b.t = 0; b.init = true; }
    const t = ++b.t, q = clip([p[0] - lr(t) * b.m[0], p[1] - lr(t) * b.m[1]]);
    for (let i = 0; i < 2; i++) b.v[i] = b2 * b.v[i] + (1 - b2) * b.gn[i] * b.gn[i];
    const g = sg(q); b.gn = g;
    for (let i = 0; i < 2; i++) b.m[i] = B1 * b.m[i] + (1 - B1) * g[i] / Math.max(Math.sqrt(b.v[i]), EPS);
    return q;
  };
}
const MCOL = { adam: HEX.opt2, amsgrad: HEX.opt3, adopt: HEX.ball };

export default {
  tabs: [
    {
      id: 'toy', name: 'Where Adam fails', title: 'Same unbiased noisy gradient, opposite answers',
      caption: 'Left–right is the paper’s toy: minimize θ on [−1, 1] (answer −1) when the gradient is usually −k and rarely a huge +k², unbiased with mean 1. Adam (violet) normalizes by a second moment that already contains the current gradient, so the rare huge gradients are shrunk while the frequent −k steps are not, and it drifts to the wrong end +1 for small β2. ADOPT (coral) divides each fresh gradient by the previous second moment before the momentum and heads to −1 for every β2. AMSGrad (white) gets there more slowly.',
      note: 'f(θ₁, θ₂) = θ₁ + θ₂²/2 on [−1, 1]²: θ₁ is the paper’s Eq. (17) toy, θ₂ a noiseless bowl so the surface has two axes. β1 = 0.9, η_t = 0.01/√(1 + 0.01t), projection onto the box, 400 steps per frame.',
      legend: [[HEX.ball, 'ADOPT'], [HEX.opt2, 'Adam'], [HEX.opt3, 'AMSGrad'], [HEX.good, 'solution θ₁ = −1']],
      action: 'Restart', cam: [0, 3.4, 3.8], tgt: [0, 0, 0],
      controls: [{ id: 'k', type: 'seg', label: 'noise k', value: 10, options: [[10, '10'], [50, '50']] }, { id: 'b2', type: 'seg', label: 'β2', value: .5, options: [[.1, '0.1'], [.5, '0.5'], [.9, '0.9'], [.99, '0.99'], [.999, '0.999']] }],
      build(ctx) {
        const s = ctx.stage; s.surface(surf);
        s.line(surf, [[-1, -1], [-1, 1]], C.good, .9); s.label('θ₁ = −1', surf.W(-1, 0, .35), '', HEX.good); s.label('θ₁ = +1', surf.W(1, 0, .35), '', HEX.neg);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, k = +ctx.vals.k, b2 = +ctx.vals.b2, st = [0, .8]; s.clearBalls();
        ctx.ba = s.ball(surf, { start: st, speed: 400, maxSteps: 4e5, step: adam(k, b2, false, 1), color: C.opt2, maxTrail: 3000 });
        ctx.bm = s.ball(surf, { start: st, speed: 400, maxSteps: 4e5, step: adam(k, b2, true, 1), color: C.opt3, maxTrail: 3000 });
        ctx.bo = s.ball(surf, { start: st, speed: 400, maxSteps: 4e5, step: adopt(k, b2, 1), maxTrail: 3000 });
      },
      frame(ctx) {
        ctx.stat(`step ${ctx.bo.steps.toLocaleString('en')} · θ₁: ADOPT ${ctx.bo.p[0].toFixed(3)} · Adam ${ctx.ba.p[0].toFixed(3)} · AMSGrad ${ctx.bm.p[0].toFixed(3)} (target −1)`);
      },
    },
    {
      id: 'beta2', name: 'Any β2', mode: '2d', title: 'Final θ after 400 000 steps, for every β2',
      caption: 'Average final θ over 8 runs of the toy (lower is better, −1 is the solution). With k = 10 Adam ends at +1 unless β2 ≥ 0.99, while ADOPT reaches −1 for every β2. With k = 50 Adam fails even at β2 = 0.999, AMSGrad stays above 0, and ADOPT is still moving towards −1 at 400 000 steps (single runs can already be close, the 8-run average is not).',
      note: 'Precomputed in the check script: 8 seeds, 4 × 10⁵ steps, β1 = 0.9, η_t = 0.01/√(1 + 0.01t). The paper’s Fig. 1 shows the same pattern (reported).',
      hint: '', legend: [[HEX.ball, 'ADOPT'], [HEX.opt2, 'Adam'], [HEX.opt3, 'AMSGrad']],
      controls: [{ id: 'k', type: 'seg', label: 'noise k', value: 10, options: [[10, '10'], [50, '50']] }],
      plot: { xlabel: 'β2', ylabel: 'final θ', xlim: [-.4, 4.4], ylim: [-1.1, 1.1], legend: false, xfmt: v => ['0.1', '0.5', '0.9', '0.99', '0.999'][Math.round(v)] ?? '' },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, k = ctx.vals.k, P = ctx.plot.clear().hline(-1, { color: HEX.good, dash: [4, 4], text: 'solution' }).hline(0, { color: HEX.muted, alpha: .4 });
        for (const m of ['adam', 'amsgrad', 'adopt']) P.line(D.b2s.map((b, i) => [i, D.res[`${k}-${b}`][m]]), { color: MCOL[m], width: 2.2, dots: 4 });
        P.draw();
        ctx.stat(`k = ${k}: ` + D.b2s.map(b => `β2 ${b}: Adam ${D.res[`${k}-${b}`].adam.toFixed(2)}, ADOPT ${D.res[`${k}-${b}`].adopt.toFixed(2)}`).join(' · '));
      },
    },
  ],
};
