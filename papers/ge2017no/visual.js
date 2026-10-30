// Ge, Jin, Zheng 2017 — one direction of improvement for all low-rank problems: Δ = U − U*R (Definition 6) and the
// Lemma 7 identity; the balancing regularizer of the asymmetric case (Sec. 5). Checked in scripts/check-math/ge2017no.py.
import { def, opt, C, HEX, segPts, hyperbola } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const US = [.9, .45], MS = [US[0] * US[0], US[0] * US[1], US[1] * US[1]];
const L = (a, b) => .5 * ((a * a - MS[0]) ** 2 + 2 * (a * b - MS[1]) ** 2 + (b * b - MS[2]) ** 2);
const G = (a, b) => { const n2 = a * a + b * b; return [2 * (n2 * a - MS[0] * a - MS[1] * b), 2 * (n2 * b - MS[1] * a - MS[2] * b)]; };
const H = (a, b) => { const n2 = a * a + b * b; return [2 * (n2 + 2 * a * a - MS[0]), 2 * (2 * a * b - MS[1]), 2 * (n2 + 2 * b * b - MS[2])]; };   // 2(‖u‖²I + 2uuᵀ − M*)
const F = def({ L, g: G, hess: H, h: v => Math.log1p(v * 4), dom: [-1.4, 1.4, -1.4, 1.4] });
const delta = u => { const p = [u[0] - US[0], u[1] - US[1]], m = [u[0] + US[0], u[1] + US[1]]; return Math.hypot(...p) <= Math.hypot(...m) ? p : m; };   // r = 1: R = ±1
const fro2 = (a, b) => (a * a - MS[0]) ** 2 + 2 * (a * b - MS[1]) ** 2 + (b * b - MS[2]) ** 2;

export default {
  tabs: [
    {
      id: 'delta', name: 'Direction Δ', title: 'At any point, moving toward the aligned truth reveals curvature or descent',
      caption: 'f(u) = ½‖uuᵀ − M*‖² with M* = u*u*ᵀ. Pick any point u (white ring). The violet segment is Δ = u − u*R, the difference to the closest copy of u* (R = ±1). Lemma 7 rewrites the curvature along Δ exactly. At a stationary point the gradient term vanishes and the curvature is at most −‖M − M*‖², so every stationary point other than ±u* is a saddle with Δ as its escape direction.',
      note: 'Symmetric case with H = identity and no regularizer (the paper’s simplest instance). u* = (0.9, 0.45). The identity is checked live; the check script confirms it at 400 random points.',
      legend: [[HEX.white, 'chosen point u'], [HEX.opt2, 'direction Δ (toward u*R)'], [HEX.good, '±u* (global)'], [HEX.ball, 'GD from u']],
      action: 'Release a ball from u', cam: [0, 4.4, 3.8], tgt: [0, .3, 0],
      controls: [{ id: 'a', label: 'u₁', min: -1.3, max: 1.3, step: .01, value: 0 }, { id: 'b', label: 'u₂', min: -1.3, max: 1.3, step: .01, value: 0 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.marker(F, ...US, C.good); s.marker(F, -US[0], -US[1], C.good); s.eigenLines(F, 0, 0, { len: .45, labels: false });
        this.place(ctx);
      },
      place(ctx) {
        const s = ctx.stage, u = [ctx.vals.a, ctx.vals.b], D = delta(u);
        (ctx.objs || []).forEach(o => o && s.remove(o));
        ctx.objs = [s.marker(F, ...u, C.white, .07), s.tube(F, segPts(u[0], u[1], u[0] - D[0], u[1] - D[1]), C.opt2, .016)];
        const [h11, h12, h22] = H(...u), lhs = h11 * D[0] ** 2 + 2 * h12 * D[0] * D[1] + h22 * D[1] ** 2;
        const g = G(...u), dd = (D[0] * D[0]) ** 2 + 2 * (D[0] * D[1]) ** 2 + (D[1] * D[1]) ** 2, e = fro2(...u), rhs = dd - 3 * e + 4 * (g[0] * D[0] + g[1] * D[1]);
        ctx.stat(`Δᵀ∇²f(u)Δ = ${lhs.toFixed(4)} · Lemma 7: ‖ΔΔᵀ‖² − 3‖M − M*‖² + 4⟨∇f, Δ⟩ = ${dd.toFixed(3)} − ${(3 * e).toFixed(3)} + ${(4 * (g[0] * D[0] + g[1] * D[1])).toFixed(3)} = ${rhs.toFixed(4)} · Lemma 6: ‖ΔΔᵀ‖² ≤ 2‖M − M*‖² (${dd.toFixed(3)} ≤ ${(2 * e).toFixed(3)})`);
      },
      drop(ctx) { const s = ctx.stage; s.clearBalls(); s.ball(F, { start: [ctx.vals.a + 1e-3, ctx.vals.b], step: opt.gd(.05), speed: 2, maxSteps: 3000 }); },
      onControl(ctx) { this.place(ctx); },
    },
    {
      id: 'balance', name: 'Balancing', title: 'In the asymmetric case a regularizer removes the scale freedom',
      caption: 'Asymmetric rank-1, 1×1 case: f(u, v) = ½(uv − 1)². Every point of the hyperbola uv = 1 is a global minimum, with any imbalance (u, v) → (cu, v/c), so minima are not isolated and the Hessian has a zero eigenvalue. Adding the paper’s regularizer ⅛(u² − v²)² keeps only the balanced minima (1, 1), (−1, −1) and leaves the origin a strict saddle.',
      note: 'Section 5 reduces the asymmetric problem to the symmetric one with this balancing term (used in Theorems 3–5).',
      legend: [[HEX.ball, 'GD from random starts'], [HEX.good, 'global minima'], [HEX.neg, 'λ < 0 at the origin']],
      action: 'Release balls', cam: [0, 4.6, 3.6], tgt: [0, .3, 0],
      controls: [{ id: 'reg', type: 'seg', label: 'objective', value: 'no', options: [['no', '½(uv − 1)²'], ['yes', '+ ⅛(u² − v²)²']] }],
      build(ctx) {
        const reg = ctx.vals.reg === 'yes';
        ctx.d = def({ L: (u, v) => .5 * (u * v - 1) ** 2 + (reg ? (u * u - v * v) ** 2 / 8 : 0),
          g: (u, v) => [(u * v - 1) * v + (reg ? (u * u - v * v) * u / 2 : 0), (u * v - 1) * u - (reg ? (u * u - v * v) * v / 2 : 0)], h: v => Math.log1p(v * 2), dom: [-2.2, 2.2, -2.2, 2.2] });
        const s = ctx.stage; s.surface(ctx.d); s.eigenLines(ctx.d, 0, 0, { len: .5 });
        if (reg) { s.marker(ctx.d, 1, 1, C.good); s.marker(ctx.d, -1, -1, C.good); } else { s.tube(ctx.d, hyperbola(1, 2.2), C.good, .012); s.tube(ctx.d, hyperbola(1, 2.2, -1), C.good, .012); }
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(5); s.clearBalls(); ctx.bs = [];
        for (let k = 0; k < 10; k++) ctx.bs.push(s.ball(ctx.d, { start: [2 * (2 * R() - 1), 2 * (2 * R() - 1)], step: opt.gd(.05), speed: 4, maxSteps: 4000, r: .05 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        const imb = ctx.bs.map(b => Math.abs(b.p[0] ** 2 - b.p[1] ** 2)), mx = Math.max(...imb);
        ctx.stat(`${ctx.vals.reg === 'yes' ? 'with' : 'without'} the regularizer: losses ≤ ${Math.max(...ctx.bs.map(b => b.loss())).toExponential(1)} · imbalance |u² − v²| at the endpoints up to ${mx.toFixed(3)}`);
      },
    },
  ],
};
