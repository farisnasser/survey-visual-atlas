// Bhojanapalli, Kyrillidis, Sanghavi 2016 — factored gradient descent U⁺ = U − η∇f(UUᵀ)U needs a step that scales
// with ‖X‖. The paper's own toy (Sec. 3): f(uuᵀ) = ‖uuᵀ − Y‖²_F, Y = a²e₁e₁ᵀ − b²e₂e₂ᵀ. Checked in scripts/check-math/bhojanapalli2016dropping.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const B = 1, M = 2;
const g = (u1, u2, a) => (u1 * u1 - a * a) ** 2 + 2 * u1 * u1 * u2 * u2 + (u2 * u2 + B * B) ** 2;
/** paper's update (the factor 2 of the true gradient is absorbed in η): u⁺ = u − η·2(uuᵀ − Y)u */
const fgd = (u, a, eta) => { const r = [u[0] * u[0] - a * a, u[0] * u[1], u[1] * u[1] + B * B]; return [u[0] - 2 * eta * (r[0] * u[0] + r[1] * u[1]), u[1] - 2 * eta * (r[1] * u[0] + r[2] * u[1])]; };
const norm2 = (s11, s12, s22) => { const t = (s11 + s22) / 2, d = Math.hypot((s11 - s22) / 2, s12); return Math.max(Math.abs(t + d), Math.abs(t - d)); };
/** step size (8): η = 1 / (16 (M‖X₀‖₂ + ‖∇f(X₀)‖₂)), ∇f(X) = 2(X − Y) */
const step8 = (u, a) => 1 / (16 * (M * (u[0] * u[0] + u[1] * u[1]) + 2 * norm2(u[0] * u[0] - a * a, u[0] * u[1], u[1] * u[1] + B * B)));
const dist = (u, a) => Math.min(Math.hypot(u[0] - a, u[1]), Math.hypot(u[0] + a, u[1]));
const opts = [[.8, '0.8'], [1.5, '1.5'], [2, '2'], [3, '3']];

export default {
  tabs: [
    {
      id: 'step', name: 'Step size', title: 'The convex step 1/M overshoots once the factor is large; the paper’s step does not',
      caption: 'f(uuᵀ) = ‖uuᵀ − Y‖²_F with u ∈ R²: convex in X = uuᵀ, non-convex in u, with a strict saddle at u = 0 created by the factorization and minima at u = ±(a, 0). Both balls start at the paper’s example point (a/2, b/10). Violet uses η = 1/M, the textbook step for f; coral uses the paper’s step (8), which shrinks with ‖X₀‖ and ‖∇f(X₀)‖. Raise a and 1/M blows up while the paper’s step still lands at the minimum.',
      note: 'Y = diag(a², −b²), b = 1, M = 2. A constant step is locally stable only for η < 1/(2a²) = 1/(2‖X*‖₂), so any fixed 1/M fails for large a. Balls stop when they leave the picture.',
      legend: [[HEX.ball, 'FGD, step (8)'], [HEX.opt2, 'FGD, η = 1/M'], [HEX.good, 'U* = ±(a, 0)'], [HEX.neg, 'λ < 0 at u = 0']],
      action: 'Release balls', cam: [0, 4.2, 4.6], tgt: [0, .3, 0],
      controls: [{ id: 'a', type: 'seg', label: 'signal a (‖X*‖ = a²)', value: 2, options: opts }],
      build(ctx) {
        const a = +ctx.vals.a, R = 1.6 * a, d = ctx.d = def({ L: (x, y) => g(x, y, a), dom: [-R, R, -R, R] }), s = ctx.stage;
        s.surface(d); s.marker(d, a, 0, C.good); s.marker(d, -a, 0, C.good); s.eigenLines(d, 0, 0, { len: .5 * a });
        this.drop(ctx);
      },
      drop(ctx) {
        const a = +ctx.vals.a, s = ctx.stage, u0 = [a / 2, B / 10], e8 = step8(u0, a); s.clearBalls(); ctx.e8 = e8;
        const mk = (eta, color) => s.ball(ctx.d, { start: u0, color, speed: 1, maxSteps: 3000, step: p => fgd(p, a, eta), stop: b => !ctx.d.in(...fgd(b.p, a, eta)) || dist(b.p, a) < 1e-8 });
        ctx.bs = [mk(e8, C.ball), mk(.5, C.opt2)];
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const a = +ctx.vals.a, st = b => dist(b.p, a) < 1e-6 ? `at U* after ${b.steps} steps` : b.done ? `left the picture after ${b.steps} steps (diverges)` : `step ${b.steps}, Dist ${dist(b.p, a).toExponential(1)}`;
        ctx.stat(`a = ${a} · step (8) η = ${ctx.e8.toFixed(4)}: ${st(ctx.bs[0])} · η = 1/M = 0.5: ${st(ctx.bs[1])} · stable steps need η < 1/(2a²) = ${(1 / (2 * a * a)).toFixed(4)}`);
      },
    },
    {
      id: 'rate', name: 'Linear rate', mode: '2d', title: 'From a good start, FGD converges linearly, as gradient descent on a strongly convex f would',
      caption: 'Dist(Uₖ, U*)² on a log scale, from a start inside the paper’s basin (A2). The paper’s step converges linearly and stays under the Corollary 4.3 bound ρᵏ·Dist₀² (dashed). A step tuned to 90% of the stability limit is faster; η = 1/M diverges for a > 1.',
      note: 'U₀ = U* + 0.99·(a/100)(0.6, 0.8), so Dist(U₀, U*) ≤ σ_r(U*)/(100κ)·σ_r(X*)/σ₁(X*) with κ = 1, rank 1. ρ = 1 − mσ_r(X*)/(64(M‖X*‖₂ + ‖∇f(X*)‖₂)), m = M = 2. Computed live.',
      hint: '', legend: [[HEX.ball, 'step (8)'], [HEX.opt3, 'η = 0.9 / (2a²)'], [HEX.opt2, 'η = 1/M'], [HEX.muted, 'bound ρᵏ Dist₀²']],
      controls: [{ id: 'a', type: 'seg', label: 'signal a', value: 2, options: opts }],
      plot: { xlabel: 'iteration k', ylabel: 'Dist(Uₖ, U*)²', logy: true, xlim: [0, 400], ylim: [1e-30, 10], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const a = +ctx.vals.a, u0 = [a + .6 * a / 100 * .99, .8 * a / 100 * .99], e8 = step8(u0, a), d0 = dist(u0, a) ** 2;
        const run = eta => { let u = u0.slice(); const out = [[0, d0]]; for (let k = 1; k <= 400; k++) { u = fgd(u, a, eta); const v = dist(u, a) ** 2; out.push([k, isFinite(v) && v < 1e6 ? Math.max(v, 1e-32) : NaN]); if (!(v < 1e6)) break; } return out; };
        const rho = 1 - M * a * a / (64 * (M * a * a + 2 * B * B)), A = run(e8), T = run(.9 / (2 * a * a)), Mv = run(.5);
        ctx.plot.clear().fn(k => d0 * rho ** k, { color: HEX.muted, dash: [5, 4] }).line(A, { color: HEX.ball, width: 2.2 }).line(T, { color: HEX.opt3, width: 1.8 }).line(Mv, { color: HEX.opt2, width: 1.8 }).draw();
        const ratio = A[200][1] / A[199][1], blew = Mv.findIndex(q => isNaN(q[1]));
        ctx.stat(`a = ${a} · η(8) = ${e8.toFixed(5)} · bound ρ = ${rho.toFixed(4)} per step, measured ${ratio.toFixed(4)} · η = 1/M: ${blew > 0 ? `diverged at k = ${blew}` : 'converges (a < 1)'}`);
      },
    },
  ],
};
