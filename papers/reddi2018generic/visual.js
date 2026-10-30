// Reddi, Zaheer, Sra, Póczos, Bach, Salakhutdinov, Smola 2018 — a generic framework: alternate a gradient-focused
// optimizer (SVRG) with a Hessian-focused one (HessianDescent). Checked in scripts/check-math/reddi2018generic.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng, logspace } from '../../src/lib/math.js';

const F = def({ L: (x, y) => .5 * x * x - .5 * y * y + .25 * y ** 4, g: (x, y) => [x, -y + y ** 3], hess: (x, y) => [1, 0, -1 + 3 * y * y], h: v => v, dom: [-1.6, 1.6, -1.6, 1.6] });
const N = 50, R0 = rng(3), A = Array.from({ length: N }, () => R0.normal()), Cc = Array.from({ length: N }, () => .4 * R0.normal());
const ma = A.reduce((s, v) => s + v, 0) / N, mc = Cc.reduce((s, v) => s + v, 0) / N; A.forEach((v, i) => A[i] = v - ma); Cc.forEach((v, i) => Cc[i] = v - mc);
const gi = (p, i) => { const [gx, gy] = F.g(...p); return [gx + A[i] + Cc[i] * p[0], gy - Cc[i] * p[1]]; };
const M = 9, L = 1 + 3 * 1.5 ** 2, ETA = 1 / (4 * L * N ** (2 / 3)), GAMMA = .5;   // M = Hessian-Lipschitz bound on |y| <= 1.5

/** One SVRG step (Alg. 2); the ball keeps its snapshot and epoch counter. */
function svrg(p, b, R) {
  if (b.k % N === 0) { b.snap = p.slice(); b.mu = F.g(...p); b.its = []; }
  const i = Math.floor(R() * N), a = gi(p, i), c = gi(b.snap, i);
  const q = [p[0] - ETA * (a[0] - c[0] + b.mu[0]), p[1] - ETA * (a[1] - c[1] + b.mu[1])]; b.its.push(q); b.k++; return q;
}
/** HessianDescent (Alg. 3) with the exact minimum eigenvector; returns [point, moved?]. */
function hessDescent(x) {
  const h22 = -1 + 3 * x[1] ** 2;
  if (Math.min(1, h22) > -GAMMA) return [x, false];
  const v = h22 < 1 ? [0, 1] : [1, 0], q = Math.min(1, h22), gv = F.g(...x)[0] * v[0] + F.g(...x)[1] * v[1], s = Math.sign(gv) || 1;
  const u = [x[0] - Math.abs(q) / M * s * v[0], x[1] - Math.abs(q) / M * s * v[1]];
  return [F.L(...u) < F.L(...x) ? u : x, true];
}

export default {
  tabs: [
    {
      id: 'framework', name: 'Framework', title: 'A first-order method plus an occasional curvature step',
      caption: 'Both balls start on the saddle’s stable set (y = 0) of a 50-component finite sum. White runs SVRG only and converges to the saddle: no component gradient ever has a y-part there. Coral runs Algorithm 1: an SVRG epoch, then pick its last iterate with probability p or a random one, then one HessianDescent step, which moves along negative curvature whenever λ_min ≤ −γ.',
      note: 'f_i = x²/2 − y²/2 + y⁴/4 + a_i x + c_i(x² − y²)/2 (zero-mean a, c). SVRG step 1/(4Ln^{2/3}) = 0.0024, epoch m = n = 50, p = ½, γ = 0.5, M = 9. HessianDescent uses the exact eigenvector here.',
      legend: [[HEX.opt3, 'SVRG alone'], [HEX.ball, 'SVRG + HessianDescent'], [HEX.good, 'minima (0, ±1)'], [HEX.flat, 'HessianDescent move']],
      action: 'Run again', cam: [3.2, 3.4, 4.2], tgt: [0, .3, 0],
      build(ctx) { const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .6 }); s.marker(F, 0, 1, C.good); s.marker(F, 0, -1, C.good); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6); s.clearBalls(); (ctx.marks || []).forEach(m => s.remove(m)); ctx.marks = [];
        ctx.a = s.ball(F, { start: [1.5, 0], color: C.opt3, speed: 12, maxSteps: 12000, step: (p, b) => { b.k ??= 0; return svrg(p, b, R); } });
        ctx.b = s.ball(F, { start: [1.5, 0], speed: 12, maxSteps: 12000, stop: b => b.cert && Math.hypot(...F.g(...b.p)) < 1e-6,
          step: (p, b) => {
            b.k ??= 0; b.hfo ??= 0; b.rounds ??= 0;
            let q = svrg(p, b, R);
            if (b.k % N === 0) {   // end of an epoch: u = last iterate w.p. p, else a random one; then HFO
              b.rounds++; const u = R() < .5 ? q : b.its[Math.floor(R() * b.its.length)];
              const [y, moved] = hessDescent(u); b.cert = !moved;
              if (moved && (y[0] !== u[0] || y[1] !== u[1])) { b.hfo++; ctx.marks.push(s.dot(F.W(y[0], y[1], .05), C.flat, .04)); }
              q = y;
            }
            return q;
          } });
      },
      frame(ctx) {
        const { a, b } = ctx; if (!a) return;
        ctx.stat(`SVRG alone: f = ${a.loss().toFixed(4)}, y = ${a.p[1].toExponential(1)} · framework: round ${b.rounds || 0}, HessianDescent moves ${b.hfo || 0}, f = ${b.loss().toFixed(4)}${b.done ? ' (min −0.25)' : ''}`);
      },
    },
    {
      id: 'lemma2', name: 'Lemma 2', mode: '2d', title: 'One HessianDescent step guarantees a decrease of order γ³/M²',
      caption: 'On f = x²/2 − (γ/2)y² + (M/6)|y|³ at the saddle (Hessian M-Lipschitz). With the exact eigenvector the step α = γ/M lowers f by γ³/(3M²). Any unit v with vᵀ∇²f v ≤ λ_min + γ/2 still gives at least Lemma 2’s h = γ³/(24M²).',
      note: 'Worst case over all admissible v, scanned on 2001 angles. M = 1. All three curves have slope 3 on log–log axes.',
      hint: '', legend: [[HEX.good, 'exact eigenvector: γ³/(3M²)'], [HEX.ball, 'worst admissible v'], [HEX.muted, 'Lemma 2: γ³/(24M²)']],
      plot: { xlabel: 'negative curvature γ', ylabel: 'decrease f(x) − f(y)', logx: true, logy: true, xlim: [.05, 2], ylim: [5e-6, 5], legend: false },
      build(ctx) {
        const Mx = 1, worstAt = gam => {
          const fg = (x, y) => .5 * x * x - .5 * gam * y * y + Mx / 6 * Math.abs(y) ** 3; let w = Infinity;
          for (let k = 0; k <= 2000; k++) { const phi = k / 2000 * Math.PI / 2, v = [Math.sin(phi), Math.cos(phi)], q = v[0] ** 2 - gam * v[1] ** 2; if (q > -gam / 2) continue; const al = Math.abs(q) / Mx; w = Math.min(w, -Math.min(fg(-al * v[0], -al * v[1]), 0)); }
          return w;
        };
        const worst = logspace(.05, 2, 30).map(gam => [gam, worstAt(gam)]);
        ctx.plot.clear().fn(g => g ** 3 / 3, { color: HEX.good, width: 2 }).fn(g => g ** 3 / 24, { color: HEX.muted, dash: [5, 4], width: 2 }).line(worst, { color: HEX.ball, width: 2, dots: 3 }).draw();
        ctx.stat(`γ = 0.5, M = 1: exact eigenvector ${(.125 / 3).toFixed(5)} · worst admissible v ${worstAt(.5).toFixed(5)} · Lemma 2 bound ${(.125 / 24).toFixed(5)}`);
      },
    },
  ],
};
