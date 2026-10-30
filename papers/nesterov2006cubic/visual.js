// Nesterov & Polyak 2006 — cubic regularization of Newton's method: x_{k+1} = global minimizer of the cubic model (Eq. 2.4).
// Toy f = x²/2 − y²/2 + y⁴/4 + 0.3xy², constants L = 10 (Hessian) and L1 = 8 (gradient) from scripts/check-math/nesterov2006cubic.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { eig2, clamp } from '../../src/lib/math.js';

const K3 = .3, L = 10;
const fL = (x, y) => .5 * x * x - .5 * y * y + .25 * y ** 4 + K3 * x * y * y;
const fG = (x, y) => [x + K3 * y * y, -y + y ** 3 + 2 * K3 * x * y];
const fH = (x, y) => [1, 2 * K3 * y, -1 + 3 * y * y + 2 * K3 * x];
const YS = Math.sqrt(1 / (1 - 2 * K3 * K3)), XM = -K3 * YS * YS, X0 = [1.2, .03];
const DOM = [-1.6, 1.6, -1.6, 1.6];
const F = def({ L: fL, g: fG, hess: fH, h: v => Math.log1p(Math.max(v + .4, 0) * 3), dom: DOM });

/** Exact global minimizer of gᵀs + ½sᵀHs + (M/6)‖s‖³ in 2D (secular equation, hard case included). */
function cubicStep(g, H, M) {
  const ev = eig2(...H), [[l1, v1], [l2, v2]] = ev;
  const gb = [g[0] * v1[0] + g[1] * v1[1], g[0] * v2[0] + g[1] * v2[1]], lo0 = Math.max(0, -2 * l1 / M) + 1e-14;
  const nr = r => Math.hypot(gb[0] / (l1 + M * r / 2), gb[1] / (l2 + M * r / 2));
  let d;
  if (nr(lo0) < lo0) { const a2 = -gb[1] / (l2 + M * lo0 / 2); d = [Math.sqrt(Math.max(lo0 * lo0 - a2 * a2, 0)), a2]; }
  else { let lo = lo0, hi = lo0 + 1; while (nr(hi) > hi) hi *= 2; for (let i = 0; i < 100; i++) { const m = (lo + hi) / 2; if (nr(m) > m) lo = m; else hi = m; } d = [-gb[0] / (l1 + M * hi / 2), -gb[1] / (l2 + M * hi / 2)]; }
  return [d[0] * v1[0] + d[1] * v2[0], d[0] * v1[1] + d[1] * v2[1]];
}
const newtonStep = (g, H) => { const [a, b, c] = H, det = a * c - b * b; return [-(c * g[0] - b * g[1]) / det, -(-b * g[0] + a * g[1]) / det]; };
const crOpt = M => (p, b, d) => { const s = cubicStep(d.g(...p), d.hess(...p), M); return [p[0] + s[0], p[1] + s[1]]; };
const newtonOpt = (p, b, d) => { const s = newtonStep(d.g(...p), d.hess(...p)); return [p[0] + s[0], p[1] + s[1]]; };
const model = (x, M) => { const g = fG(...x), H = fH(...x), f0 = fL(...x); return (y0, y1) => { const s = [y0 - x[0], y1 - x[1]]; return f0 + g[0] * s[0] + g[1] * s[1] + .5 * (H[0] * s[0] * s[0] + 2 * H[1] * s[0] * s[1] + H[2] * s[1] * s[1]) + M / 6 * Math.hypot(...s) ** 3; }; };
const gn = p => Math.hypot(...fG(...p));

export default {
  tabs: [
    {
      id: 'saddle', name: 'Newton vs cubic', title: 'Plain Newton is pulled into the saddle; the cubic step is not',
      caption: 'All three balls start at (1.2, 0.03), right next to the strict saddle at the origin. Pure Newton (white) jumps onto the saddle and stays. Gradient descent (violet, step 1/L₁) escapes slowly. Cubic regularization with M = L (coral) minimizes an upper model of f each step, so f only goes down and it reaches a minimum in about a dozen steps.',
      note: 'f = x²/2 − y²/2 + y⁴/4 + 0.3xy²; on the box |x|,|y| ≤ 1.6 the Hessian is 10-Lipschitz (sampled sup 9.11) and the gradient 8-Lipschitz. Subproblems are solved exactly.',
      legend: [[HEX.ball, 'cubic regularization, M = L'], [HEX.opt3, 'pure Newton'], [HEX.opt2, 'gradient descent'], [HEX.good, 'minima']],
      action: 'Run again', cam: [2.6, 3.4, 4.2], tgt: [0, .3, 0],
      charts: [{ id: 'g', title: '‖∇f(x_k)‖', xlabel: 'iteration k', ylabel: 'gradient norm', logy: true, ylim: [1e-10, 10], h: 220 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .55 });
        s.marker(F, XM, YS, C.good); s.marker(F, XM, -YS, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        ctx.b = [s.ball(F, { start: X0, step: crOpt(L), speed: .08, maxSteps: 30 }),
          s.ball(F, { start: X0, step: newtonOpt, color: C.opt3, speed: .08, maxSteps: 30 }),
          s.ball(F, { start: X0, step: opt.gd(1 / 8), color: C.opt2, speed: .5, maxSteps: 220 })];
        for (const q of ctx.b) { q.h = [[0, gn(q.p)]]; q.onStep = z => z.h.push([z.steps, Math.max(gn(z.p), 1e-10)]); }
      },
      frame(ctx) {
        const [a, n, d] = ctx.b || []; if (!a) return;
        ctx.charts.g.clear().set({ xlim: [0, Math.max(30, d.steps)] })
          .line(d.h, { color: HEX.opt2, width: 1.4, label: 'GD' }).line(n.h, { color: HEX.opt3, width: 1.6, label: 'Newton' }).line(a.h, { color: HEX.ball, width: 2, label: 'cubic' }).draw();
        const at = p => `(${p.map(v => v.toFixed(3)).join(', ')})`;
        ctx.stat(`cubic: ${at(a.p)}, f = ${a.loss().toFixed(4)} · Newton: ${at(n.p)}, f = ${n.loss().toFixed(4)} · GD step ${d.steps}: f = ${d.loss().toFixed(4)}`);
      },
    },
    {
      id: 'model', name: 'The cubic model', title: 'A Newton model that cannot run off to −∞',
      caption: 'The violet sheet is the model at the current point x: f(x) + ∇f(x)ᵀs + ½sᵀ∇²f(x)s + (M/6)‖s‖³. With M = 0 it is Newton’s quadratic, unbounded below where the Hessian is indefinite. With M ≥ L it lies above f everywhere (Lemma 1), so stepping to its global minimum (ring) can only lower f. Press “Take the step” to iterate.',
      note: 'Same f and box as the first tab; heights are log-scaled for display and the model is clipped at the floor. Right: the slice through x along the most negative Hessian direction.',
      legend: [['#b28cff', 'cubic model m_x'], [HEX.ball, 'current x'], [HEX.opt3, 'model minimizer T_M(x)']],
      cam: [3.4, 3.2, 3.6], tgt: [0, .2, 0],
      controls: [{ id: 'M', type: 'seg', label: 'cubic weight M', value: 10, options: [[0, '0 (Newton)'], [3, '3'], [10, '10 = L'], [25, '25']] }],
      actions: [{ label: 'Take the step', fn: ctx => { const t = ctx.cur.T; if (t) { ctx.cur.x = DOMIN(t); ctx.cur.k++; } ctx.shell.tab.build(ctx); } },
        { label: 'Reset', fn: ctx => { ctx.cur = { x: X0.slice(), k: 0 }; ctx.shell.tab.build(ctx); } }],
      charts: [{ id: 'slice', title: 'slice along the λ_min eigenvector', xlabel: 't', ylabel: 'value', ylim: [-1.5, 2.5], h: 230 }],
      build(ctx) {
        ctx.cur ||= { x: X0.slice(), k: 0 };
        const s = ctx.stage, x = ctx.cur.x, M = +ctx.vals.M; s.clear();
        const Ff = def({ L: fL, g: fG, hess: fH, h: v => clamp(v, -1.6, 2.6) * .55, dom: DOM, fit: 0 });
        const m = model(x, M), Fm = def({ L: m, h: v => clamp(v, -1.6, 2.6) * .55, dom: DOM, fit: 0, N: 70 });
        s.surface(Ff); s.surface(Fm, { grid: false, opacity: .5, colorFn: () => [.7, .55, 1] });
        s.marker(Ff, XM, YS, C.good); s.marker(Ff, XM, -YS, C.good);
        s.dot(Ff.W(x[0], x[1], .03), C.ball, .06);
        const H = fH(...x), [[l1, v1]] = eig2(...H);
        const T = M > 0 ? (st => [x[0] + st[0], x[1] + st[1]])(cubicStep(fG(...x), H, M)) : (Math.abs(H[0] * H[2] - H[1] * H[1]) > 1e-12 ? (st => [x[0] + st[0], x[1] + st[1]])(newtonStep(fG(...x), H)) : null);
        ctx.cur.T = T;
        if (T && F.in(...T)) { s.marker(Ff, T[0], T[1], C.opt3, .09); s.dot(Fm.W(T[0], T[1], .03), C.opt3, .045); }
        const sl = [], ts = [];
        for (let t = -1.6; t <= 1.6001; t += .02) ts.push(t);
        const P = t => [x[0] + t * v1[0], x[1] + t * v1[1]];
        const q = model(x, 0), c = model(x, M);
        ctx.charts.slice.clear().line(ts.map(t => [t, fL(...P(t))]), { color: HEX.white, width: 2, label: 'f' })
          .line(ts.map(t => [t, q(...P(t))]), { color: HEX.muted, dash: [4, 4], width: 1.4, label: 'quadratic' })
          .line(ts.map(t => [t, c(...P(t))]), { color: '#b28cff', width: 2, label: `cubic M=${M}` }).vline(0, { color: HEX.ball, width: 1 }).draw();
        let worst = -Infinity; for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) { const y0 = DOM[0] + 3.2 * i / 40, y1 = DOM[2] + 3.2 * j / 40; worst = Math.max(worst, fL(y0, y1) - m(y0, y1)); }
        const fT = T ? fL(...T) : NaN;
        ctx.stat(`step ${ctx.cur.k} · x = (${x.map(v => v.toFixed(3)).join(', ')}), f(x) = ${fL(...x).toFixed(4)}, λ_min = ${l1.toFixed(3)} · ${T ? `f(T) = ${fT.toFixed(4)}, model min = ${m(...T).toFixed(4)}` : 'no Newton point (singular Hessian)'} · model ${worst <= 1e-9 ? 'is above f on the box' : `dips below f by up to ${worst.toFixed(2)}`}`);
      },
      onControl(ctx) { this.build(ctx); },
      leave(ctx) { ctx.cur = null; },
    },
    {
      id: 'rate', name: 'Gradient rate', mode: '2d', title: 'Theorem 1: ‖∇f‖ falls like k^{−2/3} at worst, versus k^{−1/2} for gradient descent',
      caption: 'Smallest gradient norm seen so far for the two runs of the first tab, against the worst-case guarantees evaluated with this f’s constants (dashed): Theorem 1 for cubic regularization, Nesterov’s textbook bound (1.2.13) for gradient descent with step 1/L₁.',
      note: 'Bounds use L = 10, L₁ = 8, f(x₀) − f* = 1.025. On this easy instance both methods sit far below their worst-case bounds; the theorem is about the rate exponent, not this race.',
      hint: '', legend: [[HEX.ball, 'cubic regularization'], [HEX.opt2, 'gradient descent'], [HEX.muted, 'worst-case bounds']],
      plot: { xlabel: 'iteration k', ylabel: 'min_{i≤k} ‖∇f(x_i)‖', logx: true, logy: true, xlim: [1, 300], ylim: [1e-10, 100], legend: false },
      build(ctx) {
        const D = ctx.data, cm = acc(D.cr_grad), gm = acc(D.gd_grad), Lc = D.L, L1 = D.L1, Dl = D.D;
        ctx.plot.clear()
          .fn(k => Lc * ((8 / 3) * (3 * Dl / (2 * k * Lc)) ** (1 / 3)) ** 2, { color: HEX.ball, dash: [5, 4], width: 1.3, alpha: .7 })
          .fn(k => Math.sqrt(2 * L1 * Dl / (k + 1)), { color: HEX.opt2, dash: [5, 4], width: 1.3, alpha: .7 })
          .line(gm.map((v, i) => [i, v]).slice(1), { color: HEX.opt2, width: 2 })
          .line(cm.map((v, i) => [i, v]).slice(1), { color: HEX.ball, width: 2.4 })
          .text(150, Lc * ((8 / 3) * (3 * Dl / (2 * 150 * Lc)) ** (1 / 3)) ** 2, 'Thm 1 ∝ k^{−2/3}', { color: HEX.ball, align: 'right', dy: -10 })
          .text(150, Math.sqrt(2 * L1 * Dl / 151), 'GD ∝ k^{−1/2}', { color: HEX.opt2, align: 'right', dy: 12 }).draw();
        ctx.stat(`‖∇f‖ < 10⁻⁶ after ${D.kcr} cubic steps vs ${D.kgd} gradient steps · each cubic step needs a Hessian and a small eigen-solve`);
      },
    },
  ],
};
const DOMIN = p => [clamp(p[0], DOM[0], DOM[1]), clamp(p[1], DOM[2], DOM[3])];
const acc = a => { let m = Infinity; return a.map(v => (m = Math.min(m, v))); };
