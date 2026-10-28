// Ghadimi & Lan 2013 — randomized stochastic gradient (RSG) on a smooth non-convex landscape.
// Landscape, L, f*, critical points and the Corollary 2.2 check: scripts/check-math/ghadimi2013stochastic.py
import { def, C, HEX, statLabels } from '../../src/lib/atlas3d.js';
import { rng, logspace } from '../../src/lib/math.js';
import { rampCss } from '../../src/lib/charts.js';

const WELLS = [[-1.3, 1.0, .45, .5], [-.9, -1.1, -.7, .35], [.7, -.1, 1.2, .3]];
const f = (x, y) => { let v = .12 * (x * x + y * y); for (const [a, cx, cy, s2] of WELLS) v += a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s2); return v; };
const g = (x, y) => { let gx = .24 * x, gy = .24 * y; for (const [a, cx, cy, s2] of WELLS) { const e = a * Math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / s2); gx += e * (-2 * (x - cx) / s2); gy += e * (-2 * (y - cy) / s2); } return [gx, gy]; };
const L_SMOOTH = 5.529, F_STAR = -1.1596, X1 = [1.6, 1.5];
const CRIT = [[.959, .428, 'global min'], [-1.050, -.668, 'local min'], [-.194, -.182, 'saddle'], [-.173, 1.975, 'saddle'], [-.132, 1.286, 'max']];
const land = () => def({ L: f, g, h: v => (v + 1.25) * .62, dom: [-2.2, 2.2, -2.2, 2.2], N: 130 });
const g2 = p => { const [a, b] = g(p[0], p[1]); return a * a + b * b; };
function marks(s, S) {
  for (const [x, y, k] of CRIT) {
    const col = k === 'global min' ? C.good : k === 'max' ? C.flat : C.white;
    s.marker(S, x, y, col, .08); s.label(k, S.W(x, y, .3), '', k === 'global min' ? HEX.good : k === 'max' ? HEX.flat : HEX.white);
  }
}

export default {
  tabs: [
    {
      id: 'rsg', name: 'RSG run', title: 'Noisy gradients, random output',
      caption: 'SGD with unbiased noisy gradients. RSG runs N steps and returns a uniformly random iterate x_R (white) instead of the last one.',
      note: 'Gradient noise: N(0, s²I) per step, so σ² = 2s². Constant step ⇒ P(R = k) uniform (Eq. 2.3).',
      legend: [[HEX.ball, 'SGD path'], ['#ffffff', 'output x_R'], [HEX.good, 'global minimum'], [HEX.flat, 'maximum']],
      action: 'Run RSG', cam: [0, 4.6, 5.0], tgt: [0, .4, 0],
      controls: [
        { id: 's', label: 'noise s', min: 0, max: 1.5, step: .05, value: .6 },
        { id: 'N', label: 'steps N', min: 20, max: 600, step: 10, value: 300, fmt: v => (+v).toFixed(0) },
      ],
      build(ctx) { const s = ctx.stage, S = ctx.S = s.surface(land()); marks(s, S); this.drop(ctx); },
      onControl() { },
      drop(ctx) {
        const s = ctx.stage, S = ctx.S, r = rng(Date.now() % 1e6), N = ctx.vals.N, sig = Math.SQRT2 * ctx.vals.s;
        s.clearBalls(); if (ctx.out) s.remove(ctx.out); ctx.out = null;
        const Df = Math.sqrt(2 * (f(...X1) - F_STAR) / L_SMOOTH), gam = sig > 0 ? Math.min(1 / L_SMOOTH, Df / (sig * Math.sqrt(N))) : 1 / L_SMOOTH;
        ctx.R = 1 + Math.floor(r() * N); ctx.gam = gam; ctx.path = [];
        const sn = ctx.vals.s;
        s.ball(S, { start: X1.slice(), speed: 2, maxSteps: N, step: (p, b) => { const [gx, gy] = g(p[0], p[1]); ctx.path.push(p.slice()); return [p[0] - gam * (gx + sn * r.normal()), p[1] - gam * (gy + sn * r.normal())]; } });
      },
      frame(ctx) {
        const b = ctx.stage.balls[0]; if (!b) return;
        if (b.done && !ctx.out) { const xr = ctx.path[ctx.R - 1] || b.p; ctx.xr = xr; ctx.out = ctx.stage.dot(ctx.S.W(xr[0], xr[1], .07), C.white, .07); }
        ctx.stat(`step ${b.steps}/${ctx.vals.N} · γ = ${ctx.gam.toFixed(3)} · R = ${ctx.R}` + (ctx.xr ? ` · ‖∇f(x_R)‖² = ${g2(ctx.xr).toExponential(2)}` : ''));
      },
    },
    {
      id: 'rate', name: 'Rate vs bound', mode: '2d', title: 'E‖∇f(x_R)‖² shrinks like 1/√N',
      caption: 'Average over 150 RSG runs per N on this landscape (dots), against the Corollary 2.2 bound L·B_N (line). The bound holds at every N.',
      note: 'L = 5.53 (max |eig ∇²f| on a grid), f* = −1.1596, x₁ = (1.6, 1.5), D̃ = D_f. Run live in your browser.',
      legend: [], hint: '',
      controls: [{ id: 's', label: 'noise s', min: .1, max: 1.5, step: .05, value: .6, live: false }],
      actions: [{ label: 'Re-run', fn: ctx => ctx.shell.tab.compute(ctx) }],
      plot: { logx: true, logy: true, xlabel: 'iteration limit N', ylabel: 'E ‖∇f(x_R)‖²', xlim: [8, 4000] },
      build(ctx) { this.compute(ctx); },
      onControl(ctx) { this.compute(ctx); },
      compute(ctx) {
        const s = ctx.vals.s, sig = Math.SQRT2 * s, Df = Math.sqrt(2 * (f(...X1) - F_STAR) / L_SMOOTH), r = rng(Date.now() % 1e6), pts = [];
        const Ns = [10, 20, 50, 100, 200, 500, 1000, 3000];
        for (const N of Ns) {
          const gam = Math.min(1 / L_SMOOTH, Df / (sig * Math.sqrt(N)));
          let acc = 0; const runs = 150;
          for (let k = 0; k < runs; k++) { let x = X1[0], y = X1[1]; const R = 1 + Math.floor(r() * N); for (let t = 1; t < R; t++) { const [gx, gy] = g(x, y); x -= gam * (gx + s * r.normal()); y -= gam * (gy + s * r.normal()); } acc += g2([x, y]); }
          pts.push([N, acc / runs]);
        }
        const bound = N => L_SMOOTH * (L_SMOOTH * Df * Df / N + (Df + Df * Df / Df) * sig / Math.sqrt(N));
        const P = ctx.plot.clear();
        P.fn(bound, { color: HEX.neg, width: 2, label: 'Corollary 2.2: L·B_N' });
        P.fn(N => bound(10) * Math.sqrt(10 / N), { color: HEX.muted, dash: [4, 5], label: '∝ 1/√N' });
        P.line(pts, { color: HEX.ball, width: 2, dots: 3.5, label: 'empirical (150 runs each)' });
        P.draw();
        ctx.stat(pts.every(([N, v]) => v <= bound(N)) ? '✓ empirical value below the bound at every N' : '✗ bound violated (should not happen)');
      },
    },
    {
      id: 'smallgrad', name: 'Small ≠ minimum', title: 'A small gradient is not a minimum',
      caption: 'Amber marks every point with ‖∇f‖² ≤ ε. It surrounds the minima, the saddles and the maximum alike: first-order stationarity cannot tell them apart.',
      legend: [[HEX.flat, '‖∇f‖² ≤ ε']], cam: [0, 5.0, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'eps', label: 'ε', min: -4, max: -.5, step: .1, value: -2, fmt: v => `1e${(+v).toFixed(1)}` }],
      build(ctx) { const s = ctx.stage; ctx.S = s.surface(land()); marks(s, ctx.S); this.paint(ctx); },
      onControl(ctx) { this.paint(ctx); },
      paint(ctx) {
        const eps = 10 ** ctx.vals.eps, S = ctx.S, rgb = css => css.match(/\d+/g).map(v => v / 255);
        let n = 0, tot = 0;
        ctx.stage.recolor(S, (x, y) => { tot++; const v = g2([x, y]); if (v <= eps) { n++; return [1, .78, .34]; } return rgb(rampCss(Math.pow((S.H(x, y) - S.hmin) / (S.hmax - S.hmin), .6))); });
        ctx.stat(`${(100 * n / tot).toFixed(1)}% of the plotted area has ‖∇f‖² ≤ ${eps.toExponential(0)}`);
      },
    },
    {
      id: 'zeroth', name: 'Zeroth order', title: 'Descent from function values only',
      caption: 'RSGF never sees a gradient: it evaluates f at x and x + μu with u ~ N(0, I) and steps along u. Same landscape, same start.',
      note: 'Estimator G_μ = (f(x + μu) − f(x))/μ · u is unbiased for ∇f_μ, the gradient of a Gaussian-smoothed f (Eq. 3.12).',
      legend: [[HEX.ball, 'RSG (noisy gradient)'], [HEX.opt2, 'RSGF (function values)']],
      action: 'Race again', cam: [0, 4.6, 5.0], tgt: [0, .4, 0],
      controls: [{ id: 'mu', label: 'smoothing μ', min: .01, max: .8, step: .01, value: .05 }],
      build(ctx) { const s = ctx.stage; ctx.S = s.surface(land()); marks(s, ctx.S); ctx.L = statLabels(s, [['', ctx.S.W(0, -2.2, 0).setY(0)]]); this.drop(ctx); },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, S = ctx.S, r = rng(Date.now() % 1e6), mu = ctx.vals.mu; s.clearBalls();
        s.ball(S, { start: [1.7, -1.6], speed: 2, maxSteps: 600, step: p => { const [gx, gy] = g(p[0], p[1]); return [p[0] - .05 * (gx + .4 * r.normal()), p[1] - .05 * (gy + .4 * r.normal())]; } });
        s.ball(S, { start: [1.7, -1.6], speed: 2, maxSteps: 600, color: C.opt2, step: p => { const u = [r.normal(), r.normal()], d = (f(p[0] + mu * u[0], p[1] + mu * u[1]) - f(p[0], p[1])) / mu; return [p[0] - .05 * d * u[0], p[1] - .05 * d * u[1]]; } });
      },
      frame(ctx) { const [a, b] = ctx.stage.balls; if (a && b) ctx.stat(`step ${a.steps} · RSG f = ${a.loss().toFixed(3)} · RSGF f = ${b.loss().toFixed(3)} · f* = −1.160`); },
    },
  ],
};
