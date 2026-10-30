// Shin, Lee, Chung, Lee 2025 — SASSHA: SAM gradient + square-rooted |diag H| preconditioner (lazy, EMA'd).
// Toy dynamics and the reported Table 1 are in scripts/check-math/shin2025sassha.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const CS = [-.55, .35], SS = .16, CF = [.9, -.45], SF = .55, B = .04, AF = 1.035, START = [-.3, .2], DELTA = 1e-3;
const ws = (x, y) => Math.exp(-((x - CS[0]) ** 2 + (y - CS[1]) ** 2) / (2 * SS * SS)), wf = (x, y) => AF * Math.exp(-((x - CF[0]) ** 2 + (y - CF[1]) ** 2) / (2 * SF * SF));
const L = (x, y) => 1 + B * (x * x + y * y) - ws(x, y) - wf(x, y);
const G = (x, y) => { const s = ws(x, y), f = wf(x, y); return [2 * B * x + s * (x - CS[0]) / SS ** 2 + f * (x - CF[0]) / SF ** 2, 2 * B * y + s * (y - CS[1]) / SS ** 2 + f * (y - CF[1]) / SF ** 2]; };
const Hd = (x, y) => { const s = ws(x, y), f = wf(x, y); return [2 * B + s * (1 / SS ** 2 - (x - CS[0]) ** 2 / SS ** 4) + f * (1 / SF ** 2 - (x - CF[0]) ** 2 / SF ** 4), 2 * B + s * (1 / SS ** 2 - (y - CS[1]) ** 2 / SS ** 4) + f * (1 / SF ** 2 - (y - CF[1]) ** 2 / SF ** 4)]; };
const surf = def({ L, g: G, dom: [-1.6, 1.6, -1.6, 1.6], h: v => Math.log1p(v * 3) });
const newton = eta => p => { const g = G(...p), h = Hd(...p); return [p[0] - eta * g[0] / (Math.abs(h[0]) + DELTA), p[1] - eta * g[1] / (Math.abs(h[1]) + DELTA)]; };
const sassha = (eta, rho) => p => {
  const g = G(...p), n = Math.hypot(...g) || 1, q = [p[0] + rho * g[0] / n, p[1] + rho * g[1] / n], gt = G(...q), h = Hd(...q);
  return [p[0] - eta * gt[0] / Math.sqrt(Math.abs(h[0]) + DELTA), p[1] - eta * gt[1] / Math.sqrt(Math.abs(h[1]) + DELTA)];
};

export default {
  tabs: [
    {
      id: 'wells', name: 'Sharp or flat', title: 'Dividing by |H| keeps you anywhere; dividing by √|H| does not',
      caption: 'The equal-depth sharp/flat toy from the SAM page. A diagonal-Newton step (violet, like AdaHessian or Sophia-H) rescales each coordinate by its curvature, so a sharp minimum is as stable as a flat one and it stays in the sharp well. SASSHA (coral) divides by √|diag H| at a SAM-perturbed point: near a minimum of curvature a its step multiplies the error by about 1 − η√a, so sharp minima become unstable and it settles in the flat well. Lower η or ρ to see when it stays.',
      note: 'Deterministic toy with exact diagonal Hessians, no moving averages (β1 = β2 = 0), Hessian every step (k = 1), δ = 10⁻³ under the root. The 1 − η√a rule is our 1-D linearization ignoring the SAM term; the paper’s Corollary 4.6 gives the stochastic stability conditions.',
      legend: [[HEX.opt2, 'diagonal Newton (g / |H|)'], [HEX.ball, 'SASSHA (g̃ / √|H̃|)'], [HEX.good, 'flat min'], [HEX.neg, 'sharp min']],
      action: 'Restart', cam: [0, 5, 4.4], tgt: [0, .2, 0],
      controls: [{ id: 'eta', label: 'step η', min: .05, max: .7, step: .01, value: .5 }, { id: 'rho', type: 'seg', label: 'SAM radius ρ', value: .05, options: [[0, '0'], [.02, '0.02'], [.05, '0.05']] }],
      build(ctx) { const s = ctx.stage; s.surface(surf); s.marker(surf, ...ctx.data.flat, C.good, .1); s.marker(surf, ...ctx.data.sharp, C.neg, .07); this.drop(ctx); },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, eta = ctx.vals.eta, rho = +ctx.vals.rho; s.clearBalls();
        ctx.bn = s.ball(surf, { start: START, speed: 2, maxSteps: 6000, step: newton(eta), color: C.opt2 });
        ctx.bs = s.ball(surf, { start: START, speed: 2, maxSteps: 6000, step: sassha(eta, rho) });
      },
      frame(ctx) {
        const D = ctx.data, eta = ctx.vals.eta, at = b => Math.hypot(b.p[0] - D.flat[0], b.p[1] - D.flat[1]) < .1 ? 'flat well' : Math.hypot(b.p[0] - D.sharp[0], b.p[1] - D.sharp[1]) < .1 ? 'sharp well' : 'between';
        ctx.stat(`step ${ctx.bs.steps} · diagonal Newton: ${at(ctx.bn)} · SASSHA: ${at(ctx.bs)} · η√a: sharp well ${(eta * Math.sqrt(D.aS)).toFixed(2)}, flat well ${(eta * Math.sqrt(D.aF)).toFixed(2)} (unstable above 2)`);
      },
    },
    {
      id: 'table', name: 'Reported sharpness', mode: '2d', title: 'Second-order methods find sharp minima; SASSHA does not',
      caption: 'Table 1 of the paper (ResNet-32 on CIFAR-100): the top Hessian eigenvalue of the solution each optimizer found (bars, log scale) and its validation accuracy. AdaHessian, Sophia-H and Shampoo end 45 to 1600 times sharper than SGD and generalize worse; SASSHA ends flatter than SGD with the best accuracy.',
      note: 'Reported in the paper (mean over runs), not reproduced here.',
      hint: '', legend: [[HEX.neg, 'λ_max(H) of the solution'], [HEX.good, 'SASSHA']],
      plot: { xlabel: '', ylabel: 'λ_max(H) (log)', logy: true, ylim: [50, 1e6], legend: false, xfmt: () => '' },
      build(ctx) {
        const T = ctx.data.table1, P = ctx.plot.clear();
        P.bars(T.opt.map((o, i) => ({ y: T.lmax[i], color: o === 'SASSHA' ? HEX.good : o === 'SGD' ? HEX.muted : HEX.neg })));
        T.opt.forEach((o, i) => P.text(i, T.lmax[i] * 1.6, `${o}`, { color: HEX.white, align: 'center' }).text(i, 70, `${T.acc[i]}%`, { color: '#0b0e1a', align: 'center' }));
        P.draw();
        ctx.stat(T.opt.map((o, i) => `${o}: λ_max ${T.lmax[i].toLocaleString('en')}, acc ${T.acc[i]}%`).join(' · '));
      },
    },
  ],
};
