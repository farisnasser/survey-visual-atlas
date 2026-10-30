// Murty & Kabadi 1987 — deciding local optimality is NP-hard: copositivity, a quartic whose Hessian is silent,
// and subset sum written as a box-constrained QP (Example 2). Checked in scripts/check-math/murty1987some.py.
import { def, opt, C, HEX, segPts, ramp } from '../../src/lib/atlas3d.js';

const RED = [1, .36, .45];
const tint = (L, t) => L < 0 ? ramp(Math.pow(t, .6)).map((v, i) => v * .35 + RED[i] * .65) : ramp(Math.pow(t, .6));

export default {
  tabs: [
    {
      id: 'copositive', name: 'Copositivity', title: 'Positive on the orthant is weaker than positive everywhere',
      caption: 'q(x) = xᵀQx with Q = [[1, q], [q, 1]]. Red: q(x) < 0. Q is copositive when no red reaches the quadrant x ≥ 0 (white edges), which holds for every q ≥ −1, while positive semidefiniteness needs |q| ≤ 1.',
      note: 'For 2×2 matrices both tests are easy; Murty–Kabadi show that deciding “Q is not copositive” is NP-complete for general n (their corollary).',
      legend: [[HEX.neg, 'q(x) < 0'], [HEX.white, 'edges of the orthant x ≥ 0'], [HEX.flat, 'minimizer on the simplex x₁ + x₂ = 1']],
      cam: [3.4, 3.6, 4.2], tgt: [0, .3, 0],
      controls: [{ id: 'q', label: 'off-diagonal q', min: -2, max: 2, step: .05, value: -.6, digits: 2 }],
      build(ctx) {
        const q = ctx.vals.q, L = (x, y) => x * x + 2 * q * x * y + y * y;
        const S = def({ L, g: (x, y) => [2 * x + 2 * q * y, 2 * y + 2 * q * x], hess: () => [2, 2 * q, 2], h: v => v, dom: [-1, 1, -1, 1] });
        const s = ctx.stage; s.surface(S, { colorFn: (x, y, t) => tint(L(x, y), t) });
        s.tube(S, segPts(0, 0, 1, 0), C.white, .014); s.tube(S, segPts(0, 0, 0, 1), C.white, .014);
        s.tube(S, segPts(1, 0, 0, 1), C.flat, .008);
        const mid = q < 1, m = mid ? [.5, .5] : [1, 0];
        s.marker(S, m[0], m[1], C.flat, .08);
        s.label('x ≥ 0', S.W(.7, .7, .35), '', HEX.white);
        const psd = Math.abs(q) <= 1, cop = q >= -1, mn = Math.min(1, (1 + q) / 2);
        ctx.stat(`PSD: ${psd ? 'yes' : 'no'} (eigenvalues ${(1 - q).toFixed(2)}, ${(1 + q).toFixed(2)}) · copositive: ${cop ? 'yes' : 'no'} · min of q on the simplex = ${mn.toFixed(3)}${mn < 0 ? ' < 0' : ''}`);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
    },
    {
      id: 'quartic', name: 'Silent Hessian', title: 'A local minimum test the Hessian cannot answer',
      caption: 'p(u) = u₁⁴ + 2q u₁²u₂² + u₂⁴. Gradient and Hessian are zero at the origin for every q, so second-order tests say nothing. The origin is a local minimum exactly when Q is copositive (q ≥ −1); for q < −1 gradient descent slides away along the diagonal.',
      note: 'Substituting xᵢ = uᵢ² turns copositivity of Q into local minimality of a quartic, which is how the hardness carries over to smooth unconstrained problems.',
      legend: [[HEX.ball, 'GD from (0.6, 0.62)'], [HEX.opt2, 'GD from (−0.55, 0.5)'], [HEX.flat, 'all Hessian eigenvalues 0 at the origin']],
      action: 'Release balls', cam: [3.2, 3.4, 4.4], tgt: [0, .3, 0],
      controls: [{ id: 'q', label: 'q', min: -1.6, max: 1, step: .05, value: -.5, digits: 2 }],
      build(ctx) {
        const q = ctx.vals.q;
        ctx.P = def({ L: (a, b) => a ** 4 + 2 * q * a * a * b * b + b ** 4, g: (a, b) => [4 * a ** 3 + 4 * q * a * b * b, 4 * b ** 3 + 4 * q * a * a * b],
          hess: (a, b) => [12 * a * a + 4 * q * b * b, 8 * q * a * b, 12 * b * b + 4 * q * a * a], h: v => v, dom: [-1.2, 1.2, -1.2, 1.2] });
        const s = ctx.stage; s.surface(ctx.P, { colorFn: (x, y, t) => tint(ctx.P.L(x, y), t) });
        s.eigenLines(ctx.P, 0, 0, { len: .5 }); s.marker(ctx.P, 0, 0, C.flat, .08);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.a = s.ball(ctx.P, { start: [.6, .62], step: opt.gd(.05), maxSteps: 4000, speed: 2 });
        ctx.b = s.ball(ctx.P, { start: [-.55, .5], step: opt.gd(.05), color: C.opt2, maxSteps: 4000, speed: 2 });
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        const q = ctx.vals.q, a = ctx.a; if (!a) return;
        const verdict = q > -1 ? 'origin is a strict local minimum (no second-order certificate exists)' : q === -1 ? 'p = (u₁² − u₂²)²: a valley of non-strict minima' : 'origin is NOT a local minimum: p < 0 along u₁ = u₂';
        ctx.stat(`coral: |u| = ${Math.hypot(...a.p).toFixed(3)}, p = ${a.loss().toExponential(2)} · ${verdict}`);
      },
    },
    {
      id: 'subset', name: 'Subset sum', title: 'A smooth box QP whose global minimum answers subset sum',
      caption: 'Example 2: F(y) = (3y₁ + 5y₂ − d₀)² + y₁(1 − y₁) + y₂(1 − y₂) on 0 ≤ y ≤ 1. The minimum is 0 exactly when some subset of {3, 5} sums to d₀. Balls run projected gradient descent and can stop in non-global corners.',
      note: 'With n numbers the same construction has 2ⁿ candidate corners, so computing a global minimum of a smooth non-convex QP is NP-hard.',
      legend: [[HEX.ball, 'projected GD (9 starts)'], [HEX.good, 'global minimum, F = 0'], [HEX.white, 'best point when no subset exists']],
      action: 'Release balls', cam: [2.9, 3.4, 3.9], tgt: [0, .35, 0],
      controls: [{ id: 'd0', type: 'seg', label: 'target d₀', value: 3, options: [[3, '3'], [4, '4'], [5, '5'], [8, '8']] }],
      build(ctx) {
        const d0 = +ctx.vals.d0, L = (a, b) => (3 * a + 5 * b - d0) ** 2 + a * (1 - a) + b * (1 - b);
        ctx.F = def({ L, g: (a, b) => [6 * (3 * a + 5 * b - d0) + 1 - 2 * a, 10 * (3 * a + 5 * b - d0) + 1 - 2 * b], h: v => Math.sqrt(v), dom: [0, 1, 0, 1], size: 2.8 });
        const s = ctx.stage; s.surface(ctx.F);
        const sol = { 3: [[1, 0]], 5: [[0, 1]], 8: [[1, 1]], 4: [] }[d0];
        sol.forEach(p => { s.marker(ctx.F, p[0], p[1], C.good, .08); s.label('F = 0', ctx.F.W(p[0], p[1], .3), '', HEX.good); });
        if (!sol.length) { s.marker(ctx.F, 1, .1875, C.white, .08); s.label('min F = 0.156', ctx.F.W(1, .1875, .3), '', HEX.white); }
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls(); ctx.bs = [];
        for (const a of [.15, .5, .85]) for (const b of [.15, .5, .85]) ctx.bs.push(s.ball(ctx.F, { start: [a, b], step: opt.gd(.01), maxSteps: 3000, speed: 4, r: .05 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const v = ctx.bs.map(b => b.loss()), best = Math.min(...v), d0 = +ctx.vals.d0;
        const ends = [...new Set(v.map(x => x.toFixed(3)))].sort().join(', ');
        ctx.stat(`d₀ = ${d0}: ${d0 === 4 ? 'no subset of {3, 5} sums to 4, so min F > 0' : 'a subset sums to d₀, so min F = 0'} · balls now at F ∈ {${ends}} · best ${best.toFixed(4)}`);
      },
    },
  ],
};
