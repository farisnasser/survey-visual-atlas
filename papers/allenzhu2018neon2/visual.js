// Allen-Zhu & Li 2018 (NEON2) — finding a negative-curvature direction with stochastic gradients only (Neon2online, Alg. 2).
// Components and the d = 20 outcome sweep come from scripts/check-math/allenzhu2018neon2.py (data.json).
import { def, C, HEX, circlePts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const K = .4;
const F = def({ L: (x, y) => .5 * x * x - .5 * K * y * y + .25 * y ** 4, g: (x, y) => [x, -K * y + y ** 3], hess: (x, y) => [1, 0, -K + 3 * y * y], h: v => v, dom: [-1.1, 1.1, -1.1, 1.1] });
const gi = (D, p, i) => { const [a, b, c, e] = D.comp; return [p[0] + a[i] + c[i] * p[0] + e[i] * p[1], -K * p[1] + p[1] ** 3 + b[i] - c[i] * p[1] + e[i] * p[0]]; };
const ETA = .1, R0 = .3, T = 400, SIG = 1e-3;
const POINTS = { saddle: [0, 0], weak: [0, .3], min: [0, Math.sqrt(K)] };

export default {
  tabs: [
    {
      id: 'neon2', name: 'Neon2', title: 'Gradient differences at one sample act like a power method on the Hessian',
      caption: 'Start a hair away from x₀ and repeat x ← x − η(∇f_i(x) − ∇f_i(x₀)) with one random component i, the same i at both points. Near x₀ this multiplies the offset by (I − η∇²f_i(x₀)), so it grows along negative curvature. If it reaches the ring of radius r, the direction is returned; if not within T steps, Neon2 reports “no negative curvature” (⊥).',
      note: 'f = x²/2 − 0.2y² + y⁴/4 is the mean of 100 components with random linear and curvature terms; η = 0.1, r = 0.3, T = 400. δ = 0.4 at the saddle.',
      legend: [[HEX.ball, 'Neon2 iterate x_t'], [HEX.flat, 'ring ‖x − x₀‖ = r'], [HEX.neg, 'negative curvature'], [HEX.pos, 'positive curvature']],
      action: 'Run again', cam: [2.8, 3.4, 3.8], tgt: [0, .3, 0],
      controls: [{ id: 'x0', type: 'seg', label: 'test point x₀', value: 'saddle', options: [['saddle', 'saddle (λ_min = −0.4)'], ['weak', 'y = 0.3 (λ_min = −0.13)'], ['min', 'minimum (λ_min = +0.8)']] }],
      build(ctx) {
        const s = ctx.stage, x0 = POINTS[ctx.vals.x0]; s.surface(F);
        s.eigenLines(F, ...x0, { len: .45 }); s.tube(F, circlePts(x0[0], x0[1], R0), C.flat, .01, .03);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, x0 = POINTS[ctx.vals.x0], R = rng(Date.now() % 1e6), n = D.comp[0].length; s.clearBalls();
        const g0 = i => gi(D, x0, i);
        ctx.b = s.ball(F, { start: [x0[0] + SIG * R.normal(), x0[1] + SIG * R.normal()], speed: .5, maxSteps: T,
          step: p => { const i = Math.floor(R() * n), a = gi(D, p, i), c = g0(i); return [p[0] - ETA * (a[0] - c[0]), p[1] - ETA * (a[1] - c[1])]; },
          stop: b => Math.hypot(b.p[0] - x0[0], b.p[1] - x0[1]) >= R0 });
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        const b = ctx.b; if (!b) return; const x0 = POINTS[ctx.vals.x0], d = [b.p[0] - x0[0], b.p[1] - x0[1]], nd = Math.hypot(...d);
        const [h11, h12, h22] = F.hess(...x0), lmin = Math.min(h11, h22);
        if (nd >= R0) { const v = [d[0] / nd, d[1] / nd], q = h11 * v[0] ** 2 + 2 * h12 * v[0] * v[1] + h22 * v[1] ** 2; ctx.stat(`returned v after ${b.steps} steps · vᵀ∇²f(x₀)v = ${q.toFixed(3)} (λ_min = ${lmin.toFixed(3)})`); }
        else ctx.stat(b.done ? `⊥ after T = ${T} steps: certifies ∇²f(x₀) ⪰ −δI for δ = 0.4 (true λ_min = ${lmin.toFixed(3)})` : `step ${b.steps} · ‖x_t − x₀‖ = ${nd.toExponential(2)}`);
      },
    },
    {
      id: 'outcomes', name: 'Two outcomes', mode: '2d', title: 'Theorem 1: either a good direction, or a certificate',
      caption: 'Neon2online in d = 20 on a finite sum with Hessian diag(λ_min, 0.1 … 1). Theorem 1 forces a direction when λ_min ≤ −δ and forbids one when λ_min > −δ/2 (any returned v has vᵀHv ≤ −δ/2). In between, either answer is allowed.',
      note: 'Offline: 150 runs per point, δ = 0.2, η = 0.05, T = 1.4·log(r/σ)/(ηδ) = 871, r = 0.05, σ = 10⁻⁴. Constants are ours; the paper’s are for the worst case.',
      hint: '', legend: [[HEX.ball, 'P(return v)'], [HEX.pos, 'mean vᵀHv/δ of returned v'], [HEX.muted, 'band where both outcomes are allowed']],
      plot: { xlabel: 'λ_min / δ', ylabel: 'probability · curvature/δ', xlim: [-1.7, .7], ylim: [-1.75, 1.1] },
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        P.band(-1, -.5, { color: 'rgba(142,153,189,.14)' }).hline(0, { color: HEX.muted }).hline(-.5, { color: HEX.neg, dash: [4, 4], text: 'vᵀHv = −δ/2' }).hline(1, { color: HEX.muted, dash: [2, 5] });
        P.line(D.rows.map(r => [r[0], r[1]]), { color: HEX.ball, width: 2.4, dots: 3, label: 'P(return v)' });
        P.line(D.rows.filter(r => r[2] != null).map(r => [r[0], r[2]]), { color: HEX.pos, width: 2, dots: 3, label: 'mean vᵀHv/δ' }).draw();
        const bad = D.rows.reduce((s, r) => s + r[4], 0), tot = D.rows.reduce((s, r) => s + r[5], 0);
        ctx.stat(`λ_min ≤ −δ: v returned in ≥ ${Math.round(100 * Math.min(...D.rows.filter(r => r[0] <= -1).map(r => r[1])))}% · λ_min ≥ −δ/2: ⊥ in ≥ ${Math.round(100 * (1 - Math.max(...D.rows.filter(r => r[0] >= -.5).map(r => r[1]))))}% · ${bad} of ${tot} returned v violate vᵀHv ≤ −δ/2`);
      },
    },
  ],
};
