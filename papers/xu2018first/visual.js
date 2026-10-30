// Xu, Jin, Yang 2018 — NEON: negative curvature from noise. Gradient descent on the centered function
// f̂_x(u) = f(x+u) − f(x) − ∇f(x)ᵀu (Alg. 1) and its Nesterov version NEON+ (Alg. 2). Checked in scripts/check-math/xu2018first.py.
import { def, C, HEX, circlePts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const f = (x, y) => .5 * x * x - .3 * y * y + .25 * y ** 4 + .4 * x * y * y;
const g = (x, y) => [x + .4 * y * y, -.6 * y + y ** 3 + .8 * x * y];
const X0 = [.15, .05], G0 = g(...X0), F0 = f(...X0);
const fh = (u, v) => f(X0[0] + u, X0[1] + v) - F0 - G0[0] * u - G0[1] * v;
const gh = (u, v) => { const q = g(X0[0] + u, X0[1] + v); return [q[0] - G0[0], q[1] - G0[1]]; };
const hx = [1 + 0, .8 * X0[1], -.6 + 3 * X0[1] ** 2 + .8 * X0[0]];   // ∇²f(x0): [1, .8y; .8y, -.6 + 3y² + .8x]
const lam = (hx[0] + hx[2]) / 2 - Math.sqrt(((hx[0] - hx[2]) / 2) ** 2 + hx[1] ** 2);
const FH = def({ L: fh, g: gh, h: v => v, dom: [-1, 1, -1.2, 1.2] });
const ETA = .1, R0 = .05, THR = -2.5 * .02;
const neon = (p) => { const [a, b] = gh(...p); return [p[0] - ETA * a, p[1] - ETA * b]; };
const neonPlus = zeta => (p, ball) => { ball.yp ??= p.slice(); const [a, b] = gh(...p), y = [p[0] - ETA * a, p[1] - ETA * b], u = [y[0] + zeta * (y[0] - ball.yp[0]), y[1] + zeta * (y[1] - ball.yp[1])]; ball.yp = y; return u; };

export default {
  tabs: [
    {
      id: 'fhat', name: 'Centered function', title: 'Subtract the gradient, and noise finds the negative curvature',
      caption: 'At a point x where the gradient is not zero, NEON studies f̂_x(u) = f(x + u) − f(x) − ∇f(x)ᵀu, which has a stationary point at u = 0 with the same Hessian as f at x. Gradient descent on f̂_x from a random point of a tiny sphere (coral) is a power method on I − η∇²f(x); NEON+ (violet) adds Nesterov momentum and gets there faster. Both stop once f̂_x ≤ −2.5F.',
      note: 'f = x²/2 − 0.3y² + y⁴/4 + 0.4xy², x = (0.15, 0.05) with |∇f(x)| = 0.15; η = 0.1, r = 0.05, F = 0.02, ζ = 1 − √(ηγ) with γ = |λ_min| (Theorem 3).',
      legend: [[HEX.ball, 'NEON'], [HEX.opt2, 'NEON+'], [HEX.flat, 'start sphere ‖u‖ = r'], [HEX.neg, 'negative curvature of ∇²f(x)']],
      action: 'Run again', cam: [3, 3.2, 3.8], tgt: [0, .4, 0],
      build(ctx) {
        const s = ctx.stage; s.surface(FH); s.eigenLines(FH, 0, 0, { len: .6 }); s.tube(FH, circlePts(0, 0, R0), C.flat, .008, .03);
        s.label('u = 0 (the point x)', FH.W(0, 0, .3), '', HEX.white);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6), a = R() * 2 * Math.PI, u0 = [R0 * Math.cos(a), R0 * Math.sin(a)]; s.clearBalls();
        const stop = b => fh(...b.p) <= THR;
        ctx.a = s.ball(FH, { start: u0, step: neon, speed: .5, maxSteps: 2000, stop });
        ctx.b = s.ball(FH, { start: u0, step: neonPlus(1 - Math.sqrt(ETA * Math.abs(lam))), color: C.opt2, speed: .5, maxSteps: 2000, stop });
      },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        const cosv = p => { const v = [hx[1], lam - hx[0]], n = Math.hypot(...v) * Math.hypot(...p); return Math.abs(p[0] * v[0] + p[1] * v[1]) / n; };
        const st = q => q.done && fh(...q.p) <= THR ? `detected at τ = ${q.steps}, cos(u, e_min) = ${cosv(q.p).toFixed(3)}` : `τ = ${q.steps}, f̂ = ${fh(...q.p).toExponential(2)}`;
        ctx.stat(`λ_min(∇²f(x)) = ${lam.toFixed(4)} · NEON: ${st(a)} · NEON+: ${st(b)}`);
      },
    },
    {
      id: 'speed', name: 'Power vs Lanczos', mode: '2d', title: 'Iterations to detect negative curvature −γ',
      caption: 'In d = 50 with ∇²f(x) = diag(−γ, 0 … 1), median iterations until f̂_x drops below the threshold. NEON behaves like the power method (∝ 1/γ); NEON+ like Lanczos (∝ 1/√γ), which is what lets the full algorithm reach Õ(d/ε^{3.5}) time.',
      note: 'Offline, 15 random unit starts per γ, η = 0.5, threshold 10⁻³ from radius 10⁻³. Slopes are fitted on the log–log points.',
      hint: '', legend: [[HEX.ball, 'NEON'], [HEX.opt2, 'NEON+']],
      plot: { xlabel: 'negative curvature γ', ylabel: 'iterations to detect', logx: true, logy: true, xlim: [7e-4, .15], ylim: [30, 3e4] },
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        P.line(D.gams.map((x, i) => [x, D.neon[i]]), { color: HEX.ball, width: 2, dots: 4, label: `NEON (slope ${D.slopes[0]})` })
          .line(D.gams.map((x, i) => [x, D.neonp[i]]), { color: HEX.opt2, width: 2, dots: 4, label: `NEON+ (slope ${D.slopes[1]})` }).draw();
        ctx.stat(`γ = 0.001: NEON ${D.neon[4]} iterations, NEON+ ${D.neonp[4]} · ${(D.neon[4] / D.neonp[4]).toFixed(0)}× fewer with momentum`);
      },
    },
  ],
};
