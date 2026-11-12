// Chapter 11 (survey Sections IV-E to IV-G). New visuals 'bfgs' (Eq. 34 and the inverse update of Section IV-F, with a backtracking line search) and
// 'cg' (conjugate gradients, Algorithm 1's inner solver); numbers in scripts/check-math/guide3_numbers.py.
// Reused: crash-course 'hvp' (course_curvature.py).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import curv from '../../course/curvature/viz.js';

// ---------- BFGS on the survey's added Fig. 10 objective
const F = (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2;
const Gf = (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)];
const Hf = (x, y) => [[2 - 40 * (y - x * x) + 80 * x * x, -40 * x], [-40 * x, 20]];
const START = [-1.2, 1], TOL = 1e-10;
const surf = def({ L: F, g: Gf, dom: [-1.7, 1.7, -.9, 2.1], h: v => Math.log1p(v) * .9, N: 120 });
const steps = {
  gd: (p) => { const g = Gf(...p); return [p[0] - .01 * g[0], p[1] - .01 * g[1]]; },
  newton: (p) => { const g = Gf(...p), [[a, b], [, c]] = Hf(...p), det = a * c - b * b; return [p[0] - (c * g[0] - b * g[1]) / det, p[1] - (-b * g[0] + a * g[1]) / det]; },
  bfgs: (p, bb) => {
    bb.C = bb.C || [[1, 0], [0, 1]]; bb.evals = bb.evals || 0;
    const g = Gf(...p), Cm = bb.C, d = [-(Cm[0][0] * g[0] + Cm[0][1] * g[1]), -(Cm[1][0] * g[0] + Cm[1][1] * g[1])];
    let a = 1; const f0 = F(...p), slope = g[0] * d[0] + g[1] * d[1];
    while (F(p[0] + a * d[0], p[1] + a * d[1]) > f0 + 1e-4 * a * slope && a > 1e-10) { a /= 2; bb.evals++; }   // Armijo backtracking
    bb.evals++;
    const q = [p[0] + a * d[0], p[1] + a * d[1]], g1 = Gf(...q), s = [q[0] - p[0], q[1] - p[1]], y = [g1[0] - g[0], g1[1] - g[1]];   // Eq. (34)
    const ys = y[0] * s[0] + y[1] * s[1];
    if (ys > 1e-14) {                                                                  // curvature condition y^T s > 0
      const r = 1 / ys, A = [[1 - r * s[0] * y[0], -r * s[0] * y[1]], [-r * s[1] * y[0], 1 - r * s[1] * y[1]]];
      const AC = [[A[0][0] * Cm[0][0] + A[0][1] * Cm[1][0], A[0][0] * Cm[0][1] + A[0][1] * Cm[1][1]], [A[1][0] * Cm[0][0] + A[1][1] * Cm[1][0], A[1][0] * Cm[0][1] + A[1][1] * Cm[1][1]]];
      bb.C = [0, 1].map(i => [0, 1].map(j => AC[i][0] * A[j][0] + AC[i][1] * A[j][1] + r * s[i] * s[j]));   // inverse BFGS update (survey Section IV-F, in words)
    }
    return q;
  },
};
const NAME = { gd: 'GD (η = 0.01)', bfgs: 'BFGS + line search', newton: 'exact Newton' };
const COLS = { gd: [C.opt2, HEX.opt2], bfgs: [C.ball, HEX.ball], newton: [C.opt3, HEX.opt3] };
const bfgs = {
  title: 'BFGS learns the curvature from gradient differences alone',
  caption: 'The curved valley L = (1 − x)² + 10(y − x²)² of the survey\'s Fig. 10, from (−1.2, 1). Gradient descent (violet) needs a small step for the steep walls and crawls along the curved floor. BFGS (coral) never sees a second derivative: it updates an inverse-Hessian estimate C from each step s and gradient change y (Eq. 34 and survey Section IV-F). Exact Newton (white) uses the true Hessian.',
  note: 'BFGS starts from C = I and uses a backtracking (Armijo) line search, as Section IV-F says practical BFGS does; the extra loss evaluations it costs are counted. Newton: exact Hessian, unit steps; it stays positive definite along this path. Heights log-scaled for display.',
  legend: [[HEX.opt2, 'gradient descent'], [HEX.ball, 'BFGS'], [HEX.opt3, 'Newton'], [HEX.good, 'minimum (1, 1)']],
  action: 'Restart', cam: [0, 4.3, 3.6], tgt: [0, .2, 0],
  charts: [{ id: 'f', title: 'loss (log scale)', xlabel: 'step', ylabel: 'L', logy: true, ylim: [1e-16, 100], xlim: [0, 400], h: 220, legend: false }],
  build(ctx) { ctx.stage.surface(surf); ctx.stage.marker(surf, 1, 1, C.good, .09); this.drop(ctx); },
  drop(ctx) {
    const s = ctx.stage; s.clearBalls(); ctx.H = { gd: [], bfgs: [], newton: [] }; ctx.b = {};
    for (const k of ['gd', 'newton', 'bfgs']) {
      const b = ctx.b[k] = s.ball(surf, { start: START, speed: k === 'gd' ? 2 : .25, maxSteps: 400, step: (p, bb) => steps[k](p, bb), color: COLS[k][0], stop: bb => F(...bb.p) < TOL });
      ctx.H[k].push([0, F(...START)]);
      b.onStep = bb => ctx.H[k].push([bb.steps, F(...bb.p)]);
    }
  },
  frame(ctx) {
    const P = ctx.charts.f.clear();
    for (const k of ['gd', 'bfgs', 'newton']) P.line(ctx.H[k].map(h => [h[0], Math.max(h[1], 1e-16)]), { color: COLS[k][1], width: 1.7 });
    P.draw();
    const say = k => { const H = ctx.H[k], last = H[H.length - 1]; return last[1] < TOL ? `below 10⁻¹⁰ at step ${last[0]}` : `L = ${last[1].toExponential(1)} at step ${last[0]}`; };
    ctx.stat(`${NAME.gd}: ${say('gd')} · ${NAME.bfgs}: ${say('bfgs')} (${ctx.b.bfgs.evals ?? 0} loss evaluations in its line searches) · ${NAME.newton}: ${say('newton')}`);
  },
};

// ---------- conjugate gradients vs gradient descent on a d = 60 quadratic
const D = 60;
function cgRun(kappa, K = 200) {
  const lam = Array.from({ length: D }, (_, i) => kappa ** (i / (D - 1)));          // eigenvalues 1 ... kappa (CG is basis-invariant: H diagonal w.l.o.g.)
  const xs = Array.from({ length: D }, (_, i) => Math.cos(1.7 * i + .3));             // fixed solution x*; b = H x*
  const b = lam.map((l, i) => l * xs[i]);
  const errA = x => Math.sqrt(x.reduce((s, xi, i) => s + lam[i] * (xi - xs[i]) ** 2, 0));
  let x = new Array(D).fill(0), r = b.slice(), p = r.slice(), rr = r.reduce((s, v) => s + v * v, 0);
  const e0 = errA(x), cg = [[0, 1]];
  for (let k = 1; k <= K; k++) {
    const Ap = p.map((v, i) => lam[i] * v), al = rr / p.reduce((s, v, i) => s + v * Ap[i], 0);   // one Hessian-vector product
    x = x.map((v, i) => v + al * p[i]); r = r.map((v, i) => v - al * Ap[i]);
    const rr1 = r.reduce((s, v) => s + v * v, 0); p = r.map((v, i) => v + rr1 / rr * p[i]); rr = rr1;
    cg.push([k, Math.max(errA(x) / e0, 1e-17)]);
  }
  const eta = 2 / (1 + kappa); let z = new Array(D).fill(0); const gd = [[0, 1]];
  for (let k = 1; k <= K; k++) { z = z.map((v, i) => v - eta * (lam[i] * v - b[i])); gd.push([k, Math.max(errA(z) / e0, 1e-17)]); }
  return { cg, gd };
}
const cg = {
  title: 'Conjugate gradients: a Newton-quality solve from Hessian-vector products only',
  caption: 'Solve Hx = b for a d = 60 dimensional positive-definite H with eigenvalues spread from 1 to κ. Each conjugate-gradient iteration (coral) uses one product Hv; gradient descent with its best fixed step (violet) also uses one per iteration. Dashed: the classical worst-case bounds, 2((√κ − 1)/(√κ + 1))ᵏ for CG and ((κ − 1)/(κ + 1))ᵏ for GD.',
  note: 'Error measured in the energy norm ‖x − x*‖_H, relative to the start x₀ = 0, as in the bounds. Eigenvalues log-spaced; a diagonal H loses no generality because CG and GD are rotation-invariant. In exact arithmetic CG finishes in at most d = 60 iterations.',
  legend: [[HEX.ball, 'conjugate gradients'], [HEX.opt2, 'gradient descent, η = 2/(1 + κ)']],
  mode: '2d',
  plot: { xlabel: 'iterations (= Hessian-vector products)', ylabel: 'relative error ‖x − x*‖_H (log)', logy: true, xlim: [0, 200], ylim: [1e-12, 2], legend: false },
  controls: [{ id: 'lk', label: 'condition number κ', min: 0, max: 4, step: .05, value: 2, fmt: v => Math.round(10 ** v).toLocaleString('en') }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const kappa = 10 ** ctx.vals.lk, { cg: c, gd } = cgRun(kappa), P = ctx.plot.clear();
    const q = (Math.sqrt(kappa) - 1) / (Math.sqrt(kappa) + 1), qg = (kappa - 1) / (kappa + 1);
    P.line(c.map(([k]) => [k, Math.max(2 * q ** k, 1e-17)]), { color: HEX.ball, width: 1, dash: [4, 4] });
    P.line(gd.map(([k]) => [k, Math.max(qg ** k, 1e-17)]), { color: HEX.opt2, width: 1, dash: [4, 4] });
    P.line(gd, { color: HEX.opt2, width: 2 }); P.line(c, { color: HEX.ball, width: 2.2 });
    P.draw();
    const hit = arr => { const h = arr.find(([, e]) => e < 1e-6); return h ? h[0] : '> 200'; };
    ctx.stat(`κ = ${Math.round(kappa).toLocaleString('en')} · products to reach 10⁻⁶: CG ${hit(c)}, GD ${hit(gd)} · per-iteration bound factors: CG (√κ − 1)/(√κ + 1) = ${q.toFixed(4)}, GD (κ − 1)/(κ + 1) = ${qg.toFixed(4)}`);
  },
};

export default { bfgs, cg, hvp: curv.hvp };
