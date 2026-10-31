// Stop 6 visuals. Numbers checked in scripts/check-math/course_escape.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng, eig2 } from '../../src/lib/math.js';
import { contours, drawContours, draggable } from '../../src/learn/plot2d.js';

// ---------- escape time from a strict saddle grows like log(1/offset) ----------
const fl = (x, y) => .5 * x * x + .25 * y ** 4 - .5 * y * y, gl = (x, y) => [x, y ** 3 - y];
const escSteps = d => { let p = [1, d], t = 0; while (Math.abs(p[1]) <= .5 && t < 1e6) { const g = gl(...p); p = [p[0] - .1 * g[0], p[1] - .1 * g[1]]; t++; } return t; };
const escape = {
  title: 'How long a saddle holds gradient descent',
  caption: 'f = ½x² + ¼y⁴ − ½y²: a strict saddle at the centre (curvatures 1 and −1) between two minima. GD starts at x = 1, a distance δ off the saddle\'s stable line y = 0. Each step multiplies the offset by 1 + η = 1.1, so the wait grows like log(1/δ), never stuck, but longer for every digit of δ.',
  note: 'η = 0.1. Chart: |y_t| (log scale) against the prediction δ·1.1ᵗ; the run leaves the line once the quartic term matters. Heights rescaled.',
  legend: [[HEX.ball, 'GD'], [HEX.neg, 'λ = −1 (escape direction)'], [HEX.pos, 'λ = 1'], [HEX.good, 'minima']],
  mode: '3d', cam: [3.6, 3.3, 3.9], tgt: [0, .3, 0],
  charts: [{ id: 'y', title: 'distance from the stable line', xlabel: 'step t', ylabel: '|y_t|', logy: true, xlim: [0, 300], ylim: [1e-12, 2], h: 190, legend: false }],
  controls: [{ id: 'ld', label: 'offset δ', min: -12, max: -1, step: .1, value: -6, fmt: v => '10^' + v.toFixed(1) }],
  action: 'Drop the ball',
  build(ctx) {
    ctx.d = def({ L: fl, g: gl, dom: [-1.6, 1.6, -1.6, 1.6], h: v => v, fit: 1.4 });
    ctx.d.hess = (x, y) => [1, 0, 3 * y * y - 1];
    ctx.stage.surface(ctx.d);
    ctx.stage.eigenLines(ctx.d, 0, 0, { len: .9 });
    ctx.stage.marker(ctx.d, 0, 1, C.good, .1); ctx.stage.marker(ctx.d, 0, -1, C.good, .1);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const d = 10 ** ctx.vals.ld, st = ctx.stage;
    st.clearBalls(); ctx.hist = [[0, d]];
    ctx.ball = st.ball(ctx.d, { start: [1, d], step: opt.gd(.1), speed: 1.5, maxSteps: 600, stop: b => Math.hypot(...gl(...b.p)) < 1e-6 });
    ctx.ball.onStep = b => { if (b.steps <= 300) ctx.hist.push([b.steps, Math.abs(b.p[1])]); };
    ctx.need = escSteps(d); ctx.pred = Math.log(.5 / d) / Math.log(1.1);
    this.chart(ctx);
  },
  chart(ctx) {
    const d = 10 ** ctx.vals.ld, P = ctx.charts.y.clear();
    P.fn(t => d * 1.1 ** t, { color: 'rgba(178,140,255,.75)', width: 1.4, dash: [5, 4] });
    P.hline(.5, { color: 'rgba(255,200,87,.6)', dash: [4, 4], text: '|y| = 0.5' });
    P.line(ctx.hist, { color: HEX.ball, width: 2 });
    P.draw();
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    if (b.steps !== ctx.last) { ctx.last = b.steps; this.chart(ctx); }
    ctx.stat(`δ = ${(10 ** ctx.vals.ld).toExponential(0)} · steps until |y| > 0.5: ${ctx.need} · prediction ln(0.5/δ)/ln(1.1) = ${ctx.pred.toFixed(1)} · this ball: step ${b.steps}, f = ${b.loss().toFixed(4)}`);
  },
};

// ---------- the stable set is a thin curve (measure zero) ----------
const K = .25;
const fb = (x, y) => y * y / 2 + Math.cos(x - K * y * y) - 1;
const gb = (x, y) => { const u = x - K * y * y; return [-Math.sin(u), y + 2 * K * y * Math.sin(u)]; };
const runB = (p, T) => { let [x, y] = p; for (let t = 0; t < T; t++) { const g = gb(x, y); x -= .2 * g[0]; y -= .2 * g[1]; } return [x, y]; };
const sideB = p => { const q = runB(p, 300); return Math.sign(q[0] - K * q[1] * q[1]); };
const xStar = y => { let lo = K * y * y - 1.5, hi = K * y * y + 1.5; const s0 = sideB([lo, y]); for (let i = 0; i < 55; i++) { const m = (lo + hi) / 2; if (sideB([m, y]) === s0) lo = m; else hi = m; } return (lo + hi) / 2; };
const DB = [-4, 4, -2.4, 2.4];
const stable = {
  title: 'The stable set: a thin curve',
  caption: 'f = ½y² + cos(x − y²/4) − 1: a strict saddle at the centre, minima at (±π, 0). The tint says which minimum gradient descent reaches from each start. The starts that end at the saddle form only the amber curve, its stable set. Drag the white start close to it: the path slides toward the saddle and still peels away.',
  note: 'η = 0.2, below 1/L on this box (L ≈ 3.1), as Lee et al.\'s theorem requires. The curve is computed by bisection between the two basins.',
  legend: [[HEX.flat, 'stable set of the saddle'], [HEX.ball, 'GD path'], [HEX.neg, 'saddle'], [HEX.good, 'minima']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'y', xlim: [DB[0], DB[1]], ylim: [DB[2], DB[3]], equal: true, legend: false, grid: false },
  build(ctx) {
    const n = 120, z = [], CDB = [-6, 6, -2.4, 2.4];   // drawn area (equal aspect may widen x); outside the box is not tinted
    for (let j = 0; j < n; j++) { const row = [], y = CDB[2] + (CDB[3] - CDB[2]) * (j + .5) / n; for (let i = 0; i < n; i++) { const x = CDB[0] + (CDB[1] - CDB[0]) * (i + .5) / n; row.push(Math.abs(x) > DB[1] ? 0 : sideB([x, y])); } z.push(row); }
    ctx.basin = [z, CDB];
    ctx.curve = []; for (let y = -2.4; y <= 2.4001; y += .1) ctx.curve.push([xStar(y), y]);
    ctx.sets = contours(fb, DB, [-1.9, -1.5, -1, -.5, -.2, .2, .6, 1, 1.5, 2], 120);
    const r = rng(11); let stuck = 0;
    for (let k = 0; k < 2000; k++) { const q = runB([r() * 8 - 4, r() * 4.8 - 2.4], 600); if (Math.hypot(...q) < 1e-3) stuck++; }
    ctx.stuck = stuck;
    ctx.p = [.4, 2];
    draggable(ctx, q => { ctx.p = [Math.max(DB[0], Math.min(DB[1], q[0])), Math.max(DB[2], Math.min(DB[3], q[1]))]; this.draw(ctx); });
    this.draw(ctx);
  },
  draw(ctx) {
    const P = ctx.plot.clear(), [z, CDB] = ctx.basin;
    P.heat(z, CDB, { cell: v => v > 0 ? 'rgba(124,196,255,.22)' : v < 0 ? 'rgba(95,224,176,.16)' : 'rgba(0,0,0,0)' });
    drawContours(P, ctx.sets, { color: 'rgba(232,236,248,.16)' });
    P.line(ctx.curve, { color: HEX.flat, width: 2.6 });
    const path = [ctx.p.slice()]; let [x, y] = ctx.p, hit = null;
    for (let t = 1; t <= 400; t++) { const g = gb(x, y); x -= .2 * g[0]; y -= .2 * g[1]; path.push([x, y]); if (hit === null && fb(x, y) < -1) hit = t; }
    P.line(path, { color: HEX.ball, width: 1.6 }); P.scatter(path.slice(0, 120), { color: HEX.ball, r: 1.8 });
    P.scatter([[Math.PI, 0], [-Math.PI, 0]], { color: HEX.good, r: 5 });
    P.scatter([[0, 0]], { color: HEX.neg, r: 4.5 });
    P.scatter([ctx.p], { color: '#ffffff', r: 5.5 });
    P.draw();
    const end = Math.hypot(x, y) < 1e-3 ? 'the saddle' : `the minimum (${x > 0 ? '+' : '−'}π, 0)`;
    ctx.stat(`start (${ctx.p[0].toFixed(2)}, ${ctx.p[1].toFixed(2)}) · ends at ${end}${hit ? ` · f < −1 after ${hit} steps` : ''} · 2,000 random starts in the box: ${ctx.stuck} end at the saddle`);
  },
};

// ---------- perturbed gradient descent (Jin et al. 2017, Algorithm 2) ----------
const fj = (x, y) => .5 * x * x + Math.cos(y) - 1, gj = (x, y) => [x, -Math.sin(y)];
const ETJ = .2, GTH = 1e-3, TTH = 80, FTH = .05;
function pgdStep(r, seed) {
  return (p, b) => {
    if (!b.rn) { b.rn = rng(seed); b.tn = -TTH - 1; b.xt = null; b.kicks = []; }
    const t = b.steps; let q = p;
    if (Math.hypot(...gj(...q)) <= GTH && t - b.tn > TTH) {
      b.xt = q.slice(); b.tn = t;
      const a = b.rn.normal(), c = b.rn.normal(), n = Math.hypot(a, c), s = r * Math.sqrt(b.rn());
      q = [b.xt[0] + a / n * s, b.xt[1] + c / n * s]; b.kicks.push(t);
    }
    if (b.xt && t - b.tn === TTH && fj(...q) - fj(...b.xt) > -FTH) { b.halt = t; return b.xt.slice(); }
    const g = gj(...q); return [q[0] - ETJ * g[0], q[1] - ETJ * g[1]];
  };
}
const pgd = {
  title: 'Perturbed gradient descent',
  caption: 'Both balls start at (1.5, 0), exactly on the saddle\'s stable line. Plain GD (coral) rolls into the saddle and stays: its gradient is zero. Perturbed GD (violet) adds a random kick of radius r whenever the gradient is tiny, and leaves. At the minimum the next kick does not lower f, so it stops and returns the point.',
  note: 'f = ½x² + cos y − 1 (curvatures 1 and −1 at the saddle, minima at (0, ±π)). η = 0.2, kick when ‖∇f‖ ≤ 10⁻³, at most once per 80 steps; stop if 80 steps after a kick f fell by less than 0.05. y-axis drawn compressed.',
  legend: [[HEX.ball, 'gradient descent'], [HEX.opt2, 'perturbed GD'], [HEX.good, 'minima']],
  mode: '3d', cam: [3.6, 3.4, -3.6], tgt: [0, .2, 0],
  controls: [{ id: 'lr', label: 'kick radius r', min: -3, max: -.5, step: .1, value: -1, fmt: v => (10 ** v).toPrecision(2) }],
  actions: [{ label: 'Run both', fn: ctx => pgd.drop(ctx) }, { label: 'New random kick', fn: ctx => { ctx.seed++; pgd.drop(ctx); } }],
  build(ctx) {
    ctx.seed = 1;
    ctx.d = def({ L: fj, g: gj, dom: [-2, 2, -4, 4], square: true, h: v => v, fit: 1.4 });
    ctx.d.hess = (x, y) => [1, 0, -Math.cos(y)];
    ctx.stage.surface(ctx.d);
    ctx.stage.eigenLines(ctx.d, 0, 0, { len: .8 });
    ctx.stage.marker(ctx.d, 0, Math.PI, C.good, .1); ctx.stage.marker(ctx.d, 0, -Math.PI, C.good, .1);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const st = ctx.stage, r = 10 ** ctx.vals.lr; st.clearBalls();
    ctx.gd = st.ball(ctx.d, { start: [1.5, 0], step: opt.gd(ETJ), speed: .5, maxSteps: 400 });
    ctx.pg = st.ball(ctx.d, { start: [1.5, 0], step: pgdStep(r, ctx.seed), color: C.opt2, speed: .5, maxSteps: 2000, stop: b => b.halt !== undefined });
  },
  frame(ctx) {
    const a = ctx.gd, b = ctx.pg; if (!a || !b) return;
    const ks = b.kicks || [];
    ctx.stat(`kick radius r = ${(10 ** ctx.vals.lr).toPrecision(2)}, seed ${ctx.seed} · GD: step ${a.steps}, f = ${a.loss().toFixed(3)} · perturbed GD: kicks at steps ${ks.length ? ks.join(', ') : '—'}${b.halt !== undefined ? ` · stopped at step ${b.halt}, returns f = ${b.loss().toFixed(3)}` : `, f = ${b.loss().toFixed(3)}`}`);
  },
};

// ---------- negative curvature from gradient differences = power iteration ----------
const DIM = 30;
function powerRun(gamma, T = 5000) {
  const lam = [-gamma]; for (let i = 0; i < DIM - 1; i++) lam.push(i / (DIM - 2));
  const r = rng(5); let v = Array.from({ length: DIM }, () => r.normal()); let n = Math.hypot(...v); v = v.map(a => a / n);
  const grad = x => x.map((a, i) => lam[i] * a + a ** 3), rad = 1e-3, align = [[0, Math.abs(v[0])]];
  let hit = null;
  for (let t = 1; t <= T; t++) {
    const g = grad(v.map(a => rad * a)), g0 = grad(v.map(() => 0));
    v = v.map((a, i) => a - (g[i] - g0[i]) / rad); n = Math.hypot(...v); v = v.map(a => a / n);
    align.push([t, Math.abs(v[0])]);
    if (hit === null && v.reduce((s, a, i) => s + lam[i] * a * a, 0) <= -gamma / 2) hit = t;
    if (hit !== null && t > hit * 3 && t > 50) break;
  }
  return { align, hit };
}
const power = {
  title: 'Finding the way down with gradients only',
  caption: 'At a saddle in d = 30 dimensions with one negative curvature −γ, repeat: v ← v − (∇f(x + rv) − ∇f(x))/r, then rescale v to length 1. The difference of two gradients is ≈ H v, so this is the power method on I − H: the component of v along the negative direction grows by 1 + γ per step and takes over.',
  note: 'f = ½Σλᵢxᵢ² + ¼Σxᵢ⁴ at x = 0, eigenvalues −γ and 29 values from 0 to 1, r = 10⁻³, step 1 = 1/L. Stop rule: vᵀHv ≤ −γ/2.',
  legend: [[HEX.neg, 'share of v along the negative-curvature direction']],
  mode: '2d',
  plot: { xlabel: 'iteration t', ylabel: '|⟨v, e_min⟩|', logx: true, xlim: [1, 5000], ylim: [0, 1.05], legend: false },
  controls: [{ id: 'lg', label: 'negative curvature γ', min: -3, max: 0, step: .05, value: -2, fmt: v => (10 ** v).toPrecision(2) }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const gamma = 10 ** ctx.vals.lg, { align, hit } = powerRun(gamma), P = ctx.plot.clear();
    P.line(align.filter(p => p[0] >= 1), { color: HEX.neg, width: 2.2 });
    if (hit) P.vline(hit, { color: 'rgba(255,200,87,.7)', dash: [4, 4], text: `t = ${hit}` });
    P.draw();
    ctx.stat(`γ = ${gamma.toPrecision(2)} · vᵀHv ≤ −γ/2 after ${hit ?? '> 5000'} iterations (each costs one extra gradient) · smaller γ, slower: the count grows like 1/γ`);
  },
};

// ---------- Newton is attracted to saddles; saddle-free Newton and cubic regularization are not ----------
const C3 = .3;
const fn = (x, y) => .5 * x * x - .5 * y * y + .25 * y ** 4 + C3 * x * y * y;
const gn = (x, y) => [x + C3 * y * y, -y + y ** 3 + 2 * C3 * x * y];
const Hn = (x, y) => [1, 2 * C3 * y, -1 + 3 * y * y + 2 * C3 * x];
function eigBasis(H) { return eig2(...H); }   // [[l1, v1], [l2, v2]], l1 <= l2
function cubicStep(g, H, M) {
  const ev = eigBasis(H), lam = ev.map(e => e[0]), V = ev.map(e => e[1]), gb_ = V.map(v => v[0] * g[0] + v[1] * g[1]);
  let lo = Math.max(0, -2 * lam[0] / M) + 1e-14;
  const nr = r => Math.hypot(...gb_.map((c, i) => c / (lam[i] + M * r / 2)));
  let Db;
  if (nr(lo) < lo) {   // hard case: the gradient has no component along the most negative direction
    Db = gb_.map((c, i) => lam[i] + M * lo / 2 > 1e-12 ? -c / (lam[i] + M * lo / 2) : 0);
    Db[0] = Math.sqrt(Math.max(lo * lo - Db[0] * Db[0] - Db[1] * Db[1], 0));
  } else {
    let hi = lo + 1; while (nr(hi) > hi) hi *= 2;
    for (let i = 0; i < 200; i++) { const m = (lo + hi) / 2; if (nr(m) > m) lo = m; else hi = m; }
    Db = gb_.map((c, i) => -c / (lam[i] + M * hi / 2));
  }
  return [V[0][0] * Db[0] + V[1][0] * Db[1], V[0][1] * Db[0] + V[1][1] * Db[1]];
}
const STEP = {
  gd: p => { const g = gn(...p); return [p[0] - g[0] / 8, p[1] - g[1] / 8]; },
  newton: p => { const g = gn(...p), [a, b, e] = Hn(...p), det = a * e - b * b; return [p[0] - (e * g[0] - b * g[1]) / det, p[1] - (-b * g[0] + a * g[1]) / det]; },
  sfn: p => { const g = gn(...p), ev = eigBasis(Hn(...p)); let s = [0, 0]; for (const [l, v] of ev) { const c = (v[0] * g[0] + v[1] * g[1]) / Math.max(Math.abs(l), .1); s = [s[0] + c * v[0], s[1] + c * v[1]]; } return [p[0] - s[0], p[1] - s[1]]; },
  cr: p => { const s = cubicStep(gn(...p), Hn(...p), 10); return [p[0] + s[0], p[1] + s[1]]; },
};
const conv = b => Math.hypot(...gn(...b.p)) < 1e-6;
const newton = {
  title: 'Newton is drawn to saddles; cubic regularization is not',
  caption: 'f = ½x² − ½y² + ¼y⁴ + 0.3xy² from (1.2, y₀): a strict saddle at the centre and two minima. Newton (white) jumps to the critical point of its quadratic model, and here that is the saddle. Saddle-free Newton divides by |λ| instead of λ; cubic regularization (violet) adds a (M/6)‖Δ‖³ penalty. Both leave; GD (coral) leaves too, slowly.',
  note: 'Cubic regularization: M = 10 ≥ the Hessian-Lipschitz constant on the box, exact model minimizer each step. GD: η = 1/8 (L < 8 on the box). Saddle-free Newton: |λ| floored at 0.1. Heights rescaled.',
  legend: [['#f2f4ff', 'Newton / saddle-free Newton'], [HEX.opt2, 'cubic regularization'], [HEX.ball, 'gradient descent'], [HEX.good, 'minima']],
  mode: '3d', cam: [2.9, 3.0, -3.3], tgt: [0, .15, 0],
  controls: [
    { id: 'kind', type: 'seg', label: 'white ball', value: 'newton', options: [['newton', 'Newton'], ['sfn', 'saddle-free Newton']] },
    { id: 'y0', label: 'start y₀', min: -.3, max: .3, step: .01, value: .03, digits: 2 },
  ],
  action: 'Run all three',
  build(ctx) {
    ctx.d = def({ L: fn, g: gn, dom: [-1.6, 1.6, -1.6, 1.6], h: v => v, fit: 1.4 });
    ctx.d.hess = Hn;
    ctx.stage.surface(ctx.d);
    const ys = Math.sqrt(1 / (1 - 2 * C3 * C3)), xm = -C3 * ys * ys;
    ctx.stage.marker(ctx.d, xm, ys, C.good, .1); ctx.stage.marker(ctx.d, xm, -ys, C.good, .1);
    ctx.stage.marker(ctx.d, 0, 0, C.neg, .08);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const st = ctx.stage, s = [1.2, ctx.vals.y0]; st.clearBalls();
    const mk = (kind, color, speed, max) => st.ball(ctx.d, { start: s, step: p => STEP[kind](p), color, speed, maxSteps: max, stop: conv });
    ctx.balls = { w: mk(ctx.vals.kind, C.opt3, .08, 60), cr: mk('cr', C.opt2, .08, 60), gd: mk('gd', C.ball, 1, 1000) };
  },
  frame(ctx) {
    const B = ctx.balls; if (!B) return;
    const where = b => Math.hypot(...b.p) < 1e-3 ? 'saddle' : fn(...b.p) < -.3 ? 'minimum' : '…';
    const lab = ctx.vals.kind === 'newton' ? 'Newton' : 'saddle-free Newton';
    const one = (name, b) => `${name}: ${b.steps} step${b.steps === 1 ? '' : 's'} → ${b.done ? where(b) : '…'}`;
    ctx.stat(`start (1.20, ${ctx.vals.y0.toFixed(2)}) · ${one(lab, B.w)} · ${one('cubic reg.', B.cr)} · ${one('GD', B.gd)}`);
  },
};

export default { escape, stable, pgd, power, newton };
