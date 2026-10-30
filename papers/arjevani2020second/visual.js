// Arjevani, Carmon, Duchi, Foster, Sekhari, Sridharan 2020 — the "elbow": stochastic Hessian-vector products buy
// ε^-3 instead of ε^-4, and no p-th order oracle does better. HVP-RVR estimator (Alg. 1) on a real function with
// noisy oracles. Checked in scripts/check-math/arjevani2020second.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng, logspace } from '../../src/lib/math.js';

const f = (x, y) => .25 * (x * x - 1) ** 2 + .5 * y * y + .3 * Math.sin(2 * x) * y;
const g = (x, y) => [x ** 3 - x + .6 * Math.cos(2 * x) * y, y + .3 * Math.sin(2 * x)];
const H = (x, y) => [3 * x * x - 1 - 1.2 * Math.sin(2 * x) * y, .6 * Math.cos(2 * x), 1];
const F = def({ L: f, g, hess: H, h: v => Math.log1p(Math.max(v + .4, 0) * 2.5), dom: [-1.8, 1.8, -1.5, 1.5] });
const SG = 1, SH = 1, ETA = .05;
const noisyG = (p, R) => { const [a, b] = g(...p); return [a + SG * R.normal(), b + SG * R.normal()]; };
const noisyHv = (p, v, R) => { const [a, c, e] = H(...p), z1 = R.normal(), z2 = R.normal(), z3 = R.normal(), o = SH * (z2 + z3) / 2;
  return [(a + SH * z1) * v[0] + (c + o) * v[1], (c + o) * v[0] + (e + SH * R.normal()) * v[1]]; };

/** Alg. 2 (SGD) with Alg. 1's estimator: reset with prob. b (mean of NB gradients), else integrate K noisy HVPs along the step. */
const rvr = (R, b = .1, NB = 20, eps = .1) => (p, ball) => {
  if (!ball.gEst || R() < b) { let a = 0, c = 0; for (let i = 0; i < NB; i++) { const q = noisyG(p, R); a += q[0]; c += q[1]; } ball.gEst = [a / NB, c / NB]; ball.calls += NB; }
  else {
    const d = [p[0] - ball.xp[0], p[1] - ball.xp[1]], K = Math.min(50, Math.max(1, Math.ceil((d[0] ** 2 + d[1] ** 2) / (b * eps * eps))));
    for (let k = 1; k <= K; k++) { const a0 = [ball.xp[0] + (k - 1) / K * d[0], ball.xp[1] + (k - 1) / K * d[1]], hv = noisyHv(a0, [d[0] / K, d[1] / K], R); ball.gEst = [ball.gEst[0] + hv[0], ball.gEst[1] + hv[1]]; }
    ball.calls += K;
  }
  ball.xp = p.slice(); return [p[0] - ETA * ball.gEst[0], p[1] - ETA * ball.gEst[1]];
};
const sgd = R => (p, ball) => { const q = noisyG(p, R); ball.calls++; return [p[0] - ETA * q[0], p[1] - ETA * q[1]]; };

export default {
  tabs: [
    {
      id: 'race', name: 'Noisy race', title: 'Same oracle noise, a far calmer gradient estimate',
      caption: 'Both balls take SGD steps with η = 0.05 and query oracles with unit Gaussian noise. Violet uses one stochastic gradient per step. Coral uses Algorithm 1: occasionally re-anchor with a batch of 20 gradients, otherwise update the estimate with noisy Hessian-vector products integrated along the last step. Right: ‖∇f‖ against oracle calls.',
      note: 'f = (x² − 1)²/4 + y²/2 + 0.3 sin(2x) y; reset probability b = 0.1, K ∝ ‖Δx‖²/(bε²) with ε = 0.1, K ≤ 50. One small instance: it shows the lower-variance path, not the ε^{−3} exponent.',
      legend: [[HEX.ball, 'SGD + HVP-RVR'], [HEX.opt2, 'SGD'], [HEX.good, 'minima']],
      action: 'Run again', cam: [0, 4.2, 4.4], tgt: [0, .3, 0],
      charts: [{ id: 'gn', title: '‖∇f(x)‖ vs oracle calls', xlabel: 'stochastic gradient + HVP calls', ylabel: '‖∇f‖', logy: true, ylim: [1e-3, 5], h: 240 }],
      build(ctx) { const s = ctx.stage; s.surface(F); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6); s.clearBalls();
        ctx.a = s.ball(F, { start: [-1.5, 1.2], step: rvr(R), speed: .5, maxSteps: 3000, stop: b => b.calls >= 4000 }); ctx.a.calls = 0;
        ctx.b = s.ball(F, { start: [-1.5, 1.2], step: sgd(R), color: C.opt2, speed: 2, maxSteps: 4000, stop: b => b.calls >= 4000 }); ctx.b.calls = 0;
        for (const b of [ctx.a, ctx.b]) { b.h = []; b.onStep = q => q.h.push([q.calls, Math.max(Math.hypot(...g(...q.p)), 1e-3)]); }
      },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        ctx.charts.gn.clear().set({ xlim: [0, 4000] }).line(b.h, { color: HEX.opt2, width: 1.2, label: 'SGD' }).line(a.h, { color: HEX.ball, width: 1.8, label: 'HVP-RVR' }).draw();
        const tail = h => { const t = h.slice(-40); return t.length ? t.reduce((s, r) => s + r[1], 0) / t.length : NaN; };
        ctx.stat(`oracle calls: RVR ${a.calls}, SGD ${b.calls} · recent ‖∇f‖: RVR ${tail(a.h).toFixed(3)}, SGD ${tail(b.h).toFixed(3)}`);
      },
    },
    {
      id: 'estimator', name: 'HVP estimator', mode: '2d', title: 'Why Hessian-vector products help: their error shrinks with the step',
      caption: 'Estimate the gradient change ∇f(x) − ∇f(x_prev). Two independent single-point stochastic gradients always have error about 2dσ² = 4. Algorithm 1’s integral of K noisy Hessian-vector products along the segment has error ≈ 1.5‖Δx‖²/K, tiny for the short steps SGD actually takes.',
      note: 'Monte Carlo, 1500 draws per point, at x_prev = (0.4, −0.3) on the same f as Noisy race. Dashed: predicted variance 1.5‖Δx‖²/K for the symmetric noise (Z + Zᵀ)/2.',
      hint: '', legend: [[HEX.opt2, 'two single-point gradients'], [HEX.ball, 'HVP integral (Alg. 1)'], [HEX.muted, '1.5‖Δx‖²/K']],
      controls: [{ id: 'dx', label: 'step length ‖Δx‖', min: .01, max: 1, step: .01, value: .2, digits: 2 }],
      plot: { xlabel: 'HVP queries K', ylabel: 'mean squared error', logx: true, logy: true, xlim: [1, 64], ylim: [1e-6, 20], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const R = rng(11), dn = ctx.vals.dx, xp = [.4, -.3], x = [xp[0] + .8 * dn, xp[1] + .6 * dn], gt = g(...x), g0 = g(...xp), dT = [gt[0] - g0[0], gt[1] - g0[1]];
        const Ks = [1, 2, 4, 8, 16, 32, 64], N = 1500, pts = [];
        for (const K of Ks) {
          let m = 0;
          for (let t = 0; t < N; t++) { let e = [0, 0]; for (let k = 1; k <= K; k++) { const a0 = [xp[0] + (k - 1) / K * (x[0] - xp[0]), xp[1] + (k - 1) / K * (x[1] - xp[1])], hv = noisyHv(a0, [(x[0] - xp[0]) / K, (x[1] - xp[1]) / K], R); e = [e[0] + hv[0], e[1] + hv[1]]; } m += (e[0] - dT[0]) ** 2 + (e[1] - dT[1]) ** 2; }
          pts.push([K, m / N]);
        }
        let m2 = 0; for (let t = 0; t < 4000; t++) { const a = noisyG(x, R), b = noisyG(xp, R); m2 += (a[0] - b[0] - dT[0]) ** 2 + (a[1] - b[1] - dT[1]) ** 2; } m2 /= 4000;
        ctx.plot.clear().hline(m2, { color: HEX.opt2, width: 2, text: 'two gradients' }).fn(K => 1.5 * dn * dn / K, { color: HEX.muted, dash: [5, 4] })
          .line(pts, { color: HEX.ball, width: 2.2, dots: 4 }).draw();
        ctx.stat(`‖Δx‖ = ${dn.toFixed(2)}: two-gradient MSE ${m2.toFixed(2)} · HVP integral K = 1: ${pts[0][1].toExponential(2)}, K = 64: ${pts[6][1].toExponential(2)}`);
      },
    },
    {
      id: 'elbow', name: 'Elbow', mode: '2d', title: 'Second-order information helps once, then never again',
      caption: 'Exponent of 1/ε in the number of oracle calls needed for ‖∇f‖ ≤ ε with a p-th order oracle. Stochastic: 4 with gradients only, 3 for every p ≥ 2, matched by the paper’s Ω(ε^{−3}) lower bound. Noiseless: (p + 1)/p keeps improving.',
      note: 'Exact exponents (Fig. 1 of the paper). Stochastic upper bound O(ε^{−3}) uses gradients + HVPs from a single-point oracle.',
      hint: '', legend: [[HEX.ball, 'stochastic oracles'], [HEX.pos, 'noiseless oracles']],
      plot: { xlabel: 'oracle order p', ylabel: 'exponent of 1/ε', xlim: [.6, 4.4], ylim: [0, 4.6], legend: false },
      build(ctx) {
        const P = ctx.plot.clear().set({ xfmt: v => Number.isInteger(v) ? String(v) : '' }), ps = [1, 2, 3, 4];
        P.line(ps.map(p => [p, p === 1 ? 4 : 3]), { color: HEX.ball, width: 2.6, dots: 5 }).line(ps.map(p => [p, (p + 1) / p]), { color: HEX.pos, width: 2.6, dots: 5 });
        ps.forEach(p => { P.text(p, (p === 1 ? 4 : 3) + .25, p === 1 ? 'ε⁻⁴' : 'ε⁻³', { color: HEX.ball, align: 'center' }); P.text(p, (p + 1) / p - .32, ['ε⁻²', 'ε⁻³ᐟ²', 'ε⁻⁴ᐟ³', 'ε⁻⁵ᐟ⁴'][p - 1], { color: HEX.pos, align: 'center' }); });
        P.draw();
        ctx.stat('(ε, γ)-SOSP: upper O(ε⁻³ + ε⁻²γ⁻² + γ⁻⁵), lower Ω(ε⁻³ + γ⁻⁵)');
      },
    },
  ],
};
