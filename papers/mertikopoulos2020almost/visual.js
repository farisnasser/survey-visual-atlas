// Mertikopoulos, Hallak, Kavis, Cevher 2020 — SGD with γₙ ∝ 1/nᵖ converges a.s. to the critical set and avoids strict
// saddle MANIFOLDS (a ridge, their Fig. 1). Radial toy and the local-rate runs: scripts/check-math/mertikopoulos2020almost.py
import { def, C, HEX, circlePts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

/** f = φ(r), φ(r) = r⁴/4 − r³ + r², φ′(r) = r(r − 1)(r − 2): minimum at 0, a ridge of strict saddles at r = 1, minima on r = 2. */
const F = def({ L: (x, y) => { const r = Math.hypot(x, y); return r ** 4 / 4 - r ** 3 + r * r; }, g: (x, y) => { const r = Math.hypot(x, y), k = r * r - 3 * r + 2; return [k * x, k * y]; }, dom: [-2.6, 2.6, -2.6, 2.6], h: v => v, fit: 1.5 });
const GAM = 1, M = 10, SIG = .5;
const sgd = (p, R, noise = true) => (q, b, d) => {
  const gam = GAM * (M / (b.steps + M)) ** p, g = d.g(q[0], q[1]), a = 2 * Math.PI * R();
  return [q[0] - gam * (g[0] + (noise ? SIG * Math.cos(a) : 0)), q[1] - gam * (g[1] + (noise ? SIG * Math.sin(a) : 0))];
};

export default {
  tabs: [
    {
      id: 'ridge', name: 'Saddle ridge', title: 'A whole ring of saddles, and SGD still never settles on it',
      caption: 'f depends only on the radius: a minimum in the middle, a ring of minima at r = 2, and between them a ridge at r = 1 made entirely of strict saddles (curvature −1 across the ridge, 0 along it), the kind of non-isolated saddle the paper calls a strict saddle manifold. Plain GD started exactly on the ridge stays there forever. SGD with decreasing steps started at the same places always falls off, into the middle or out to the ring, as Theorem 3 says.',
      note: 'f = r⁴/4 − r³ + r², γₙ = (10/(n + 10))^p with p = 0.75, noise = 0.5 × a uniformly random unit vector (bounded and exciting every direction, Assumption 5).',
      legend: [[HEX.ball, 'SGD started on the ridge'], [HEX.opt3, 'GD started on the ridge'], [HEX.neg, 'strict saddle ridge r = 1'], [HEX.good, 'minima (0 and r = 2)']],
      action: 'Release balls', cam: [0, 4.6, 4.2], tgt: [0, .2, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.tube(F, circlePts(0, 0, 1, 120), C.neg, .016); s.tube(F, circlePts(0, 0, 2, 160), C.good, .016); s.marker(F, 0, 0, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(8); s.clearBalls();
        ctx.bs = Array.from({ length: 8 }, (_, k) => s.ball(F, { start: [Math.cos(k * Math.PI / 4), Math.sin(k * Math.PI / 4)], step: sgd(.75, R), speed: 3, maxSteps: 4000, r: .05 }));
        ctx.gd = s.ball(F, { start: [1, 0], step: sgd(.75, R, false), speed: 3, maxSteps: 4000, color: C.opt3 });
      },
      frame(ctx) {
        if (!ctx.bs) return;
        const r = b => Math.hypot(...b.p), inn = ctx.bs.filter(b => r(b) < .2).length, out = ctx.bs.filter(b => Math.abs(r(b) - 2) < .2).length, on = ctx.bs.filter(b => Math.abs(r(b) - 1) < .2).length;
        ctx.stat(`step ${ctx.gd.steps} · SGD: in the middle ${inn}/8, on the outer ring ${out}/8, still near the ridge ${on}/8 · GD from (1, 0): r = ${r(ctx.gd).toFixed(6)} (on the ridge) · offline, 200 SGD runs from the ridge, 20 000 steps: ${Math.round(100 * ctx.data.res['0.75'][0])}% middle, ${Math.round(100 * ctx.data.res['0.75'][1])}% outer ring, 0% ridge`);
      },
    },
    {
      id: 'rate', name: 'Local rate', mode: '2d', title: 'Near a regular minimum the error falls like 1/nᵖ',
      caption: 'Theorem 4: once SGD stays near a regular minimizer x* (here the central minimum, Hessian 2I), E‖Xₙ − x*‖² = O(1/nᵖ) for γₙ ∝ 1/(n + m)ᵖ. Mean squared distance over 2000 runs on log–log axes, with dashed lines of slope −p: the fitted slopes match.',
      note: 'Offline. Smaller noise (0.1) and step (γ = 0.5) than the first tab so that every run stays in U = {‖x‖ < 0.5}, the event the theorem conditions on.',
      hint: '', legend: [[HEX.pos, 'p = 0.5'], [HEX.flat, 'p = 0.75'], [HEX.ball, 'p = 1'], [HEX.muted, 'slope −p']],
      plot: { xlabel: 'iteration n', ylabel: 'E‖Xₙ − x*‖²', logx: true, logy: true, xlim: [10, 1e5], ylim: [1e-8, 1e-2], legend: false },
      build(ctx) {
        const D = ctx.data.rates, P = ctx.plot.clear(), cols = { '0.5': HEX.pos, '0.75': HEX.flat, '1.0': HEX.ball }, fit = {};
        for (const k of Object.keys(D)) {
          const pts = D[k], p = +k, sel = pts.filter(q => q[0] >= 1000), xs = sel.map(q => Math.log(q[0])), ys = sel.map(q => Math.log(q[1]));
          const mx = xs.reduce((a, b) => a + b) / xs.length, my = ys.reduce((a, b) => a + b) / ys.length;
          fit[k] = xs.reduce((a, x, i) => a + (x - mx) * (ys[i] - my), 0) / xs.reduce((a, x) => a + (x - mx) ** 2, 0);
          const [n1, v1] = pts[pts.length - 1];
          P.fn(n => v1 * (n / n1) ** -p, { color: HEX.muted, dash: [5, 4], width: 1.2 }).line(pts, { color: cols[k], width: 2.2 });
        }
        P.draw();
        ctx.stat(`fitted slopes over n ∈ [10³, 10⁵]: ${Object.keys(fit).map(k => `p = ${k}: ${fit[k].toFixed(3)}`).join(' · ')}`);
      },
    },
  ],
};
