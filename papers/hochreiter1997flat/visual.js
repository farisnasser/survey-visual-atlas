// Hochreiter & Schmidhuber 1997 — a flat minimum is a large region where the error stays nearly constant; its weights can be
// stated with low precision (few bits), which the paper links to low overfitting. Two-minimum toy: scripts/check-math/hochreiter1997flat.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const SF = .55, SS = .12;
const F = def({
  L: (x, y) => .02 * (x * x + y * y) - Math.exp(-((x + 1) ** 2 + y * y) / (2 * SF * SF)) - Math.exp(-((x - 1) ** 2 + y * y) / (2 * SS * SS)),
  g: (x, y) => { const a = Math.exp(-((x + 1) ** 2 + y * y) / (2 * SF * SF)) / (SF * SF), b = Math.exp(-((x - 1) ** 2 + y * y) / (2 * SS * SS)) / (SS * SS); return [.04 * x + a * (x + 1) + b * (x - 1), .04 * y + a * y + b * y]; },
  dom: [-2.4, 2.4, -1.8, 1.8], h: v => v, fit: 1.4,
});
/** boundary of {f < f(m) + eps} around a minimum m, by bisection along rays (the basins are close to round) */
const level = (m, fm, eps) => Array.from({ length: 121 }, (_, i) => {
  const t = 2 * Math.PI * i / 120, c = Math.cos(t), s = Math.sin(t); let lo = 0, hi = 1.2;
  for (let k = 0; k < 40; k++) { const mid = (lo + hi) / 2; if (F.L(m[0] + mid * c, m[1] + mid * s) < fm + eps) lo = mid; else hi = mid; }
  return [m[0] + lo * c, m[1] + lo * s];
});

export default {
  tabs: [
    {
      id: 'flat', name: 'Flat vs sharp', title: 'Two minima of equal depth: one wide, one narrow',
      caption: 'Both minima reach the same training error, but the left one is flat: the white ring, where the error rises by ε, encloses a region about 21 times larger than the ring around the sharp right minimum. Each “Shake” jiggles 40 copies of each solution with random weight noise of size s: the flat solution barely changes its error, the sharp one loses most of its fit. In the paper’s minimum-description-length argument, weights that tolerate this imprecision can be described with fewer bits, which is why flat minima should overfit less.',
      note: 'f = 0.02‖w‖² − exp(−‖w − (−1, 0)‖²/(2·0.55²)) − exp(−‖w − (1, 0)‖²/(2·0.12²)): a 2-weight stand-in, not one of the paper’s networks. Rings: ε = 0.1.',
      legend: [[HEX.ball, 'perturbed flat solution'], [HEX.opt2, 'perturbed sharp solution'], ['#ffffff', 'ε-flat region (ε = 0.1)'], [HEX.good, 'minima']],
      action: 'Shake the weights', cam: [0, 4.2, 4.4], tgt: [0, .2, 0],
      controls: [{ id: 's', label: 'weight noise s', min: .02, max: .3, step: .01, value: .1, fmt: v => (+v).toFixed(2), live: false }],
      build(ctx) {
        const s = ctx.stage, D = ctx.data; s.surface(F); s.marker(F, ...D.mF, C.good); s.marker(F, ...D.mS, C.good, .06);
        s.tube(F, level(D.mF, D.fF, .1), C.white, .012); s.tube(F, level(D.mS, D.fS, .1), C.white, .012);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, R = rng(Math.floor(1e6 * Math.random())), sd = ctx.vals.s; s.clearBalls();
        let dF = 0, dS = 0;
        for (let k = 0; k < 40; k++) {
          const pf = [D.mF[0] + sd * R.normal(), D.mF[1] + sd * R.normal()], ps = [D.mS[0] + sd * R.normal(), D.mS[1] + sd * R.normal()];
          s.ball(F, { start: pf, step: p => p, maxSteps: 1, trail: false, r: .035 }); s.ball(F, { start: ps, step: p => p, maxSteps: 1, trail: false, r: .035, color: C.opt2 });
          dF += F.L(...pf) - D.fF; dS += F.L(...ps) - D.fS;
        }
        const rob = D.rob.reduce((a, q) => Math.abs(q[0] - sd) < Math.abs(a[0] - sd) ? q : a);
        ctx.stat(`s = ${sd.toFixed(2)} · mean error increase of these 40 shakes: flat ${(dF / 40).toFixed(4)}, sharp ${(dS / 40).toFixed(4)} · check script (20 000 shakes, s = ${rob[0]}): flat ${rob[1].toFixed(4)}, sharp ${rob[2].toFixed(4)} · ε-region areas (ε = 0.1): ${D.areas[2][1].toFixed(3)} vs ${D.areas[2][2].toFixed(4)}, ${D.areas[2][3].toFixed(1)} fewer bits`);
      },
      onControl(ctx) { this.drop(ctx); },
    },
  ],
};
