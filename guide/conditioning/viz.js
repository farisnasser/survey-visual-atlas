// Chapter 4 (survey Sections II-D, II-E; Appendices A-A, C-A, D-B). New visuals 'narrow', 'zoo' and 'pgdstab' (numbers in
// scripts/check-math/guide3_numbers.py); reused crash-course visuals 'precond' (course_adaptive.py) and 'kappa' (course_curvature.py).
import { HEX, C, def, opt } from '../../src/lib/atlas3d.js';
import ad from '../../course/adaptive/viz.js';
import curv from '../../course/curvature/viz.js';

/** H = R(phi) diag(L, 1) R(phi)^T as [a, b, c] = [[a, b], [b, c]] */
// the sign of s is chosen so that 45° gives the survey's Q^T diag(100, 1) Q of Eq. (41), coupling -49.5
const Hrot = (deg, L) => { const t = deg * Math.PI / 180, c = Math.cos(t), s = -Math.sin(t); return [L * c * c + s * s, (L - 1) * c * s, L * s * s + c * c]; };
const eig2 = ([a, b, c]) => { const m = (a + c) / 2, d = Math.hypot((a - c) / 2, b); return [m - d, m + d]; };
/** eigenvalues of K = M^{-1/2} H M^{-1/2} for M = I, diag(H), H */
function Keig(H, kind) {
  if (kind === 'I') return eig2(H);
  if (kind === 'H') return [1, 1];
  const [a, b, c] = H; return eig2([1, b / Math.sqrt(a * c), 1]);
}
const KINDS = [['I', 'GD, M = I', HEX.opt2], ['D', 'diagonal, M = diag(H)', HEX.ball], ['H', 'Newton, M = H', HEX.opt3]];
const rho = (lams, eta) => Math.max(...lams.map(l => Math.abs(1 - eta * l)));

const pgdstab = {
  title: 'When is a preconditioned step stable? Eq. (58) for three choices of M',
  caption: 'The valley ½θᵀHθ has curvatures κ and 1 along axes rotated by the angle. For each preconditioner M, the step θ ← (I − ηM⁻¹H)θ multiplies the error along each eigen-direction of K = M^{-1/2}HM^{-1/2} by 1 − ηλ(K). The curves show the worst factor max|1 − ηλ| against η; below the dashed line at 1 the step contracts.',
  note: 'Exact eigenvalues of the 2 × 2 matrix K; no simulation. At 45° and κ = 100 this is the matrix of the survey\'s Eq. (43): diag(H) = 50.5 I.',
  legend: KINDS.map(k => [k[2], k[1]]),
  mode: '2d',
  plot: { xlabel: 'step size η (log scale)', ylabel: 'worst factor max |1 − ηλ(K)|', logx: true, xlim: [1e-3, 10], ylim: [0, 2], legend: false },
  controls: [
    { id: 'deg', label: 'valley angle (degrees)', min: 0, max: 90, step: 1, value: 45, digits: 0 },
    { id: 'k', type: 'seg', label: 'κ', value: 100, options: [[10, '10'], [100, '100']] },
    { id: 'le', label: 'step size η', min: -3, max: 1, step: .01, value: 0, fmt: v => (10 ** v).toPrecision(3) },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const H = Hrot(ctx.vals.deg, +ctx.vals.k), eta = 10 ** ctx.vals.le, P = ctx.plot.clear();
    P.band(1e-3, 10, { color: 'rgba(0,0,0,0)' });
    P.hline(1, { color: 'rgba(255,255,255,.45)', dash: [4, 4], text: 'contracts below this line' });
    const parts = [];
    for (const [k, name, col] of KINDS) {
      const lam = Keig(H, k), etas = Array.from({ length: 400 }, (_, i) => 10 ** (-3 + 4 * i / 399));
      P.line(etas.map(e => [e, Math.min(rho(lam, e), 2)]), { color: col, width: 2 });
      const emax = 2 / Math.max(...lam);
      P.vline(emax, { color: col, dash: [2, 4], width: 1 });
      const r = rho(lam, eta);
      P.scatter([[eta, Math.min(r, 2)]], { color: col, r: 5 });
      const kap = Math.max(...lam) / Math.min(...lam), best = (kap - 1) / (kap + 1);
      const steps = best < 1e-12 ? 1 : Math.ceil(Math.log(1e-6) / Math.log(best));
      parts.push(`${name.split(',')[0]}: λ(K) = ${lam.map(l => l.toPrecision(3)).join(', ')}, stable for η < ${emax.toPrecision(4)}, factor at this η ${r.toPrecision(3)}${r < 1 ? '' : ' (unstable)'}, best-η steps to 10⁻⁶: ${steps.toLocaleString('en')}`);
    }
    P.vline(eta, { color: 'rgba(255,200,87,.6)', dash: [3, 3] });
    P.draw();
    const [a, b, c] = H;
    ctx.stat(`H = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${b.toFixed(2)}, ${c.toFixed(2)}]] · η = ${eta.toPrecision(3)} · ${parts.join(' · ')}`);
  },
};

// ---------- 'narrow' (survey Section II-D): GD with η = 0.01 on L = ½(100x² + y²), the survey's own example ----------
const NV = { L: (x, y) => .5 * (100 * x * x + y * y), g: (x, y) => [100 * x, y] };
const narrow = {
  title: 'The survey\'s narrow valley: η = 0.01 solves x in one step and leaves y crawling',
  caption: 'L = ½(100x² + y²), the valley of Eq. (8). Gradient descent multiplies x by 1 − 100η and y by 1 − η every step. At the survey\'s η = 0.01 the first factor is exactly 0, so the ball drops straight onto the valley floor in one step, then creeps along it at 0.99 per step. Raise η past 0.02 and the steep direction blows up instead.',
  note: 'Exact gradient (100x, y), no noise, start (1, 1). Height drawn as √L. Chart: |x_t| and |y_t| of the same run on a log scale; an exact zero is drawn at the floor 10⁻⁶. Dashed: the predicted factors |1 − 100η|ᵗ and (1 − η)ᵗ.',
  legend: [[HEX.ball, 'gradient descent'], [HEX.pos, '|x_t| (steep, curvature 100)'], [HEX.flat, '|y_t| (flat, curvature 1)'], [HEX.good, 'minimum']],
  mode: '3d', cam: [1.4, 2.6, 4.4], tgt: [0, .35, 0],
  charts: [{ id: 'xy', title: 'distance from the minimum along each axis', xlabel: 'step t', ylabel: '|x_t|, |y_t|', logy: true, xlim: [0, 500], ylim: [1e-6, 10], h: 200, legend: false }],
  controls: [{ id: 'eta', label: 'step size η', min: .002, max: .024, step: .0005, value: .01, fmt: v => v.toFixed(4) }],
  action: 'Drop the ball',
  build(ctx) {
    ctx.d = def({ L: NV.L, g: NV.g, dom: [-1.2, 1.2, -1.2, 1.2], h: v => Math.sqrt(v), fit: 1.5 });
    ctx.d.hess = () => [100, 0, 1];
    ctx.stage.surface(ctx.d);
    ctx.stage.eigenLines(ctx.d, 0, 0, { len: .9 });
    ctx.stage.marker(ctx.d, 0, 0, C.good, .08);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const eta = ctx.vals.eta, st = ctx.stage; st.clearBalls();
    // the chart uses an unclamped run of the same update (the 3D ball has to stop at the edge of the box if x diverges)
    ctx.sim = [[1, 1]]; for (let t = 0; t < 500; t++) { const [x, y] = ctx.sim[t]; ctx.sim.push([x - eta * 100 * x, y - eta * y]); }
    ctx.ball = st.ball(ctx.d, { start: [1, 1], step: opt.gd(eta), speed: 2, maxSteps: 500,
      stop: b => Math.abs(b.p[0]) >= 1.2 || Math.hypot(...b.p) < 1e-5 });
    ctx.last = -1; this.chart(ctx);
  },
  chart(ctx) {
    const eta = ctx.vals.eta, P = ctx.charts.xy.clear(), fx = Math.abs(1 - 100 * eta), fy = 1 - eta;
    P.fn(t => Math.max(fx ** t, 1e-6), { color: 'rgba(124,196,255,.5)', width: 1.2, dash: [5, 4] });
    P.fn(t => fy ** t, { color: 'rgba(255,200,87,.5)', width: 1.2, dash: [5, 4] });
    const n = ctx.ball && !ctx.ball.done ? ctx.ball.steps : 500, cap = v => Math.min(Math.max(Math.abs(v), 1e-6), 1e3);
    P.line(ctx.sim.slice(0, n + 1).map((p, t) => [t, cap(p[0])]), { color: HEX.pos, width: 2.2 });
    P.line(ctx.sim.slice(0, n + 1).map((p, t) => [t, cap(p[1])]), { color: HEX.flat, width: 2.2 });
    P.draw();
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    if (b.steps !== ctx.last || b.done !== ctx.wasDone) { ctx.last = b.steps; ctx.wasDone = b.done; this.chart(ctx); }
    const eta = ctx.vals.eta, fx = 1 - 100 * eta, fy = 1 - eta, ny = Math.ceil(Math.log(.01) / Math.log(fy));
    const xs = Math.abs(fx) < 1 ? (Math.abs(fx) < 1e-12 ? 'x is exactly 0 after one step' : `x shrinks by ${Math.abs(fx).toFixed(3)} per step${fx < 0 ? ', flipping sign (zig-zag)' : ''}`) : 'x GROWS: η > 2/100, unstable';
    ctx.stat(`η = ${eta.toFixed(4)} · factors 1 − 100η = ${fx.toFixed(3)}, 1 − η = ${fy.toFixed(4)} · ${xs} · y needs ${ny} steps to fall below 0.01 · κ = 100 · ball: step ${b.steps}${Math.abs(b.p[0]) >= 1.2 ? ', stopped at the edge of the box (x diverges; the chart continues the run)' : `, x = ${b.p[0].toExponential(2)}, y = ${b.p[1].toFixed(4)}`}`);
  },
};

// ---------- 'zoo' (survey Section II-E, Fig. 2(e)–(i)): what counts as a stationary point ----------
const sech2 = x => 1 / Math.cosh(x) ** 2;
const ZOO = {
  min: { name: 'local minimum', L: (x, y) => x * x + .6 * y * y, g: (x, y) => [2 * x, 1.2 * y], H: () => [2, 0, 1.2], at: [0, 0], start: [.85, .7] },
  saddle: { name: 'strict saddle', L: (x, y) => x * x - y * y, g: (x, y) => [2 * x, -2 * y], H: () => [2, 0, -2], at: [0, 0], start: [.85, .02] },
  monkey: { name: 'monkey saddle', L: (x, y) => x ** 3 - 3 * x * y * y, g: (x, y) => [3 * x * x - 3 * y * y, -6 * x * y], H: (x, y) => [6 * x, -6 * y, -6 * x], at: [0, 0], start: [.3, .05] },
  max: { name: 'local maximum', L: (x, y) => -(x * x + .6 * y * y), g: (x, y) => [-2 * x, -1.2 * y], H: () => [-2, 0, -1.2], at: [0, 0], start: [.06, .04] },
  shelf: { name: 'almost-flat shelf', L: (x, y) => .6 * (1 - Math.tanh(4 * x)) + .15 * y * y, g: (x, y) => [-2.4 * sech2(4 * x), .3 * y], H: x => [19.2 * sech2(4 * x) * Math.tanh(4 * x), 0, .3], at: [.75, 0], start: [.1, .6] },
};
const lmin = ([a, b, c]) => (a + c) / 2 - Math.hypot((a - c) / 2, b);
function verdict(z, eps, gam) {
  const [x, y] = z.at, gn = Math.hypot(...z.g(x, y)), lm = lmin(z.H(x, y));
  return { gn, lm, stat: gn < 1e-12, fosp: gn <= eps, sosp: gn <= eps && lm >= -gam };
}
const zoo = {
  title: 'Stationary points: a small gradient is not a minimum',
  caption: 'The five local shapes of the survey\'s Fig. 2(e)–(i). The amber marker is the candidate point. An ε-first-order stationary point only needs ‖∇L‖ ≤ ε; an approximate second-order one also needs λmin(H) ≥ −γ, so no direction curves down by more than γ. Change ε and γ and watch which shapes pass. Then drop the ball: gradient descent from a nearby start shows what the point really is.',
  note: 'Exact gradients and Hessians. Ball: gradient descent, η = 0.1, from a start next to the point, stopped at the edge of the box. Heights rescaled for display.',
  legend: [[HEX.flat, 'candidate point'], [HEX.ball, 'gradient descent'], [HEX.pos, 'curves up'], [HEX.neg, 'curves down']],
  mode: '3d', cam: [3.1, 3.4, 3.7], tgt: [0, .1, 0],
  controls: [
    { id: 'kind', type: 'seg', label: 'shape', value: 'saddle', options: Object.entries(ZOO).map(([k, z]) => [k, z.name]) },
    { id: 'le', label: 'ε (gradient tolerance)', min: -3, max: 0, step: .05, value: -1.3, fmt: v => (10 ** v).toPrecision(2) },
    { id: 'gam', label: 'γ (allowed downward curvature)', min: 0, max: 2.5, step: .05, value: .1 },
  ],
  action: 'Drop the ball',
  build(ctx) {
    const z = ZOO[ctx.vals.kind]; ctx.z = z;
    ctx.d = def({ L: z.L, g: z.g, dom: [-1, 1, -1, 1], h: v => v, fit: 1.4 });
    ctx.d.hess = z.H;
    ctx.stage.surface(ctx.d);
    ctx.stage.eigenLines(ctx.d, ...z.at, { len: .7 });
    ctx.stage.marker(ctx.d, ...z.at, C.flat, .09);
    this.drop(ctx);
  },
  onControl(ctx, id) { if (id === 'kind') { ctx.stage.clear(); this.build(ctx); } },
  drop(ctx) {
    ctx.stage.clearBalls();
    ctx.ball = ctx.stage.ball(ctx.d, { start: ctx.z.start, step: opt.gd(.1), speed: 1, maxSteps: 400, stop: b => Math.abs(b.p[0]) >= .999 || Math.abs(b.p[1]) >= .999 || Math.hypot(...ctx.z.g(...b.p)) < 1e-7 });
  },
  frame(ctx) {
    const z = ctx.z, eps = 10 ** ctx.vals.le, gam = ctx.vals.gam, v = verdict(z, eps, gam), b = ctx.ball;
    const yes = t => t ? 'yes' : 'no';
    const at = `(${z.at.map(c => c.toFixed(2)).join(', ')})`;
    let s = `${z.name} at ${at}: ‖∇L‖ = ${v.gn < 1e-12 ? '0' : v.gn.toPrecision(3)}, λmin(H) = ${Math.abs(v.lm) < 1e-12 ? '0' : v.lm.toPrecision(3)} · truly stationary: ${yes(v.stat)} · ε-FOSP (ε = ${eps.toPrecision(2)}): ${yes(v.fosp)} · approximate SOSP (γ = ${gam.toFixed(2)}): ${yes(v.sosp)}`;
    if (b) { const [x, y] = b.p; s += ` · ball: step ${b.steps}, ‖∇L‖ = ${Math.hypot(...z.g(x, y)).toPrecision(3)}, λmin = ${lmin(z.H(x, y)).toPrecision(3)}, L = ${z.L(x, y).toFixed(4)}${Math.abs(x) >= .999 || Math.abs(y) >= .999 ? ', stopped at the edge of the box' : ''}`; }
    ctx.stat(s);
  },
};

export default { narrow, zoo, pgdstab, precond: ad.precond, kappa: curv.kappa };
