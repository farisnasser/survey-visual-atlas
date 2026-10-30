// Anandkumar & Ge 2016 — third-order local minima can be reached efficiently, fourth-order ones cannot (NP-hard).
// Toys: f3 = x² + y³ + y⁴ (fails the third-order test at 0), f4 = x² − y⁴ + y⁶ (passes it). scripts/check-math/anandkumar2016efficient.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { eig2, hess2 } from '../../src/lib/math.js';

const F3 = def({ L: (x, y) => x * x + y ** 3 + y ** 4, g: (x, y) => [2 * x, 3 * y * y + 4 * y ** 3], dom: [-1, 1, -1.1, .9], h: v => v, fit: 1.6 });
const F4 = def({ L: (x, y) => x * x - y ** 4 + y ** 6, g: (x, y) => [2 * x, -4 * y ** 3 + 6 * y ** 5], dom: [-1, 1, -1.05, 1.05], h: v => v, fit: 1.6 });
const LR = .05;
/** GD plus a third-order step (simplified from Sec. 5): when the gradient is small and the Hessian shows no negative
 *  curvature, take the eigenvector v of the smallest |λ| and move against the sign of ∇³f[v, v, v] if it is large. */
const gd3 = (counter) => (p, b, d) => {
  const gr = d.g(p[0], p[1]);
  if (Math.hypot(...gr) < 1e-2) {
    const [a, c, e] = hess2(d.L, p[0], p[1]), ev = eig2(a, c, e);
    if (ev[0][0] > -1e-3) {
      const v = Math.abs(ev[0][0]) < Math.abs(ev[1][0]) ? ev[0][1] : ev[1][1], h = 1e-3, F = t => d.L(p[0] + t * v[0], p[1] + t * v[1]);
      const c3 = (F(2 * h) - 2 * F(h) + 2 * F(-h) - F(-2 * h)) / (2 * h ** 3);
      if (Math.abs(c3) > 1) { counter.n++; counter.at = b.steps; return [p[0] - Math.sign(c3) * .1 * v[0], p[1] - Math.sign(c3) * .1 * v[1]]; }
    }
  }
  return [p[0] - LR * gr[0], p[1] - LR * gr[1]];
};
const gd = (p, b, d) => { const gr = d.g(p[0], p[1]); return [p[0] - LR * gr[0], p[1] - LR * gr[1]]; };

export default {
  tabs: [
    {
      id: 'third', name: 'Third-order saddle', title: 'The Hessian says “minimum”; the third derivative says “go down”',
      caption: 'f = x² + y³ + y⁴. At the origin the gradient is zero and the Hessian diag(2, 0) is positive semidefinite, so any second-order test accepts it; but the cubic term y³ falls on one side, and the real minimum is at (0, −3/4). Plain GD from y > 0 creeps into the origin and stalls there. GD with the paper’s idea (when curvature is flat, test the third derivative in that direction and step against it) leaves and reaches the minimum.',
      note: 'GD step 0.05 from (0.8, 0.6). The third-order step here is simplified: the paper alternates cubic-regularized Newton with a step in a “competitive subspace” where third derivatives dominate.',
      legend: [[HEX.ball, 'GD'], [HEX.opt2, 'GD + third-order step'], [HEX.good, 'local minimum (0, −3/4)'], [HEX.flat, 'flat curvature at 0']],
      action: 'Release balls', cam: [3.4, 3, 2.6], tgt: [0, .2, 0],
      charts: [{ id: 'f', title: 'f along the run', xlabel: 'step', ylabel: 'f', xlim: [0, 1500], ylim: [-.12, .9], h: 200, legend: false }],
      build(ctx) { const s = ctx.stage; s.surface(F3); s.marker(F3, 0, -.75, C.good); s.eigenLines(F3, 0, 0, { len: .45 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); ctx.cnt = { n: 0 }; ctx.hist = [[], []];
        ctx.bs = [s.ball(F3, { start: [.8, .6], step: gd, speed: 2, maxSteps: 1500 }), s.ball(F3, { start: [.8, .6], step: gd3(ctx.cnt), speed: 2, maxSteps: 1500, color: C.opt2 })];
      },
      frame(ctx) {
        if (!ctx.bs) return;
        ctx.bs.forEach((b, i) => { const h = ctx.hist[i]; if (!h.length || h[h.length - 1][0] < b.steps) h.push([b.steps, b.loss()]); });
        ctx.charts.f.clear().line(ctx.hist[0], { color: HEX.ball, width: 2 }).line(ctx.hist[1], { color: HEX.opt2, width: 2 }).draw();
        const [a, b] = ctx.bs;
        ctx.stat(`GD: step ${a.steps}, y = ${a.p[1].toFixed(4)}, f = ${a.loss().toExponential(1)} (stalls at the saddle, y ≈ 1/(3ηk)) · GD + third-order step: ${ctx.cnt.n ? `third-order step at iteration ${ctx.cnt.at}, ` : ''}y = ${b.p[1].toFixed(4)}, f = ${b.loss().toFixed(4)}`);
      },
    },
    {
      id: 'fourth', name: 'Fourth-order saddle', title: 'Passes every third-order test, still not a minimum',
      caption: 'f = x² − y⁴ + y⁶. At the origin the gradient is zero, the Hessian is diag(2, 0) and the third derivative vanishes in the flat direction, so the origin satisfies the paper’s third-order condition; only the fourth-order term −y⁴ reveals the way down to the minima at y = ±√(2/3). Here it is easy to see, but deciding whether a quartic form is non-negative in many dimensions is NP-hard, which is why finding fourth-order local minima is NP-hard (Theorem 10). GD escapes only from off-axis starts, and slowly: about 1/(8ηy₀²) steps.',
      note: 'GD step 0.05 from (0.5, y₀); the violet ball (GD + third-order step) starts on the axis at y = 10⁻⁹ and finds no third-order direction.',
      legend: [[HEX.ball, 'GD, y₀ = 0.05, 0.1, 0.2'], [HEX.opt2, 'GD + third-order step'], [HEX.good, 'global minima'], [HEX.flat, 'flat curvature at 0']],
      action: 'Release balls', cam: [3.4, 3, 2.6], tgt: [0, .2, 0],
      build(ctx) { const s = ctx.stage; s.surface(F4); const m = Math.sqrt(2 / 3); s.marker(F4, 0, m, C.good); s.marker(F4, 0, -m, C.good); s.eigenLines(F4, 0, 0, { len: .45 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); ctx.cnt = { n: 0 };
        ctx.bs = [.05, .1, .2].map(y0 => s.ball(F4, { start: [.5, y0], step: gd, speed: 2, maxSteps: 2000, r: .055 }));
        ctx.bs.push(s.ball(F4, { start: [.5, 1e-9], step: gd3(ctx.cnt), speed: 2, maxSteps: 2000, color: C.opt2 }));
        ctx.esc = [null, null, null];
        ctx.bs.slice(0, 3).forEach((b, i) => { b.onStep = q => { if (ctx.esc[i] == null && Math.abs(q.p[1]) > .5) ctx.esc[i] = q.steps; }; });
      },
      frame(ctx) {
        if (!ctx.bs) return;
        ctx.stat(`step ${ctx.bs[0].steps} · GD reaches |y| > 0.5 at step: y₀ = 0.05 → ${ctx.esc[0] ?? '…'}, 0.1 → ${ctx.esc[1] ?? '…'}, 0.2 → ${ctx.esc[2] ?? '…'} (1/(8ηy₀²) = 1000, 250, 62) · on-axis ball with third-order test: ${ctx.cnt.n} third-order steps, y = ${ctx.bs[3].p[1].toExponential(1)}`);
      },
    },
  ],
};
