// Dauphin et al. 2014 — GD vs Newton vs saddle-free Newton on the paper's Fig. 2 functions.
// Checked in scripts/check-math/dauphin2014identifying.py (step sizes, escape counts, GOE statistics).
import { def, opt, C, HEX, segPts, statLabels } from '../../src/lib/atlas3d.js';
import { rng, eigSym } from '../../src/lib/math.js';

const box = 1.6;
const quad = eps => def({ L: (x, y) => 5 * x * x - eps * y * y, g: (x, y) => [10 * x, -2 * eps * y], hess: () => [10, 0, -2 * eps], h: v => Math.sign(v) * Math.log1p(Math.abs(v) * 2), dom: [-box, box, -box, box] });
const monkey = def({ L: (x, y) => x ** 3 - 3 * x * y * y, g: (x, y) => [3 * x * x - 3 * y * y, -6 * x * y], hess: (x, y) => [6 * x, -6 * y, -6 * x], h: v => Math.sign(v) * Math.log1p(Math.abs(v) * 2), dom: [-box, box, -box, box] });
const OPTS = [['GD', C.ball, HEX.ball], ['Newton', C.opt2, HEX.opt2], ['saddle-free Newton', C.opt3, HEX.opt3]];
const outBox = b => Math.max(Math.abs(b.p[0]), Math.abs(b.p[1])) >= box - 1e-6;
const atSaddle = b => Math.hypot(...b.p) < 1e-4;
function race(ctx, S, start, lrs, delta = .1) {
  const s = ctx.stage; s.clearBalls();
  ctx.balls = OPTS.map(([name, col], i) => s.ball(S, {
    start, color: col, speed: .5, maxSteps: 4000,
    step: i === 0 ? opt.gd(lrs[0]) : opt.curv(lrs[i], i === 1 ? 'newton' : 'sfn', delta),
    stop: b => outBox(b) || atSaddle(b),
  }));
}
const report = ctx => ctx.stat(ctx.balls.map((b, i) => `${OPTS[i][0]}: ${b.done ? (atSaddle(b) ? `stuck at saddle (${b.steps})` : `escaped in ${b.steps}`) : `${b.steps}…`}`).join(' · '));

export default {
  tabs: [
    {
      id: 'saddle', name: 'Three optimizers', title: 'Newton walks into the saddle',
      caption: 'The paper’s Fig. 2a surface 5x² − y². GD escapes, Newton is attracted to the saddle, saddle-free Newton (|H|⁻¹∇f) escapes fastest.',
      note: 'Each method uses its own step: GD η = 0.08 (stable needs η < 2/10), Newton and SFN η = 0.5.',
      legend: OPTS.map(o => [o[2], o[0]]).concat([[HEX.neg, 'λ = −2'], [HEX.pos, 'λ = 10']]),
      action: 'Restart', cam: [3.2, 3.1, 3.9], tgt: [0, .5, 0],
      build(ctx) { const s = ctx.stage, S = ctx.S = s.surface(quad(1)); s.eigenLines(S, 0, 0, { len: .8 }); s.marker(S, 0, 0, C.white); this.drop(ctx); },
      drop(ctx) { race(ctx, ctx.S, [.9, .12], [.08, .5, .5]); },
      frame: report,
    },
    {
      id: 'weak', name: 'Weak curvature', title: 'Small |λ| makes gradient descent crawl',
      caption: 'f = 5x² − εy². GD moves along y at a rate set by ε; SFN rescales each direction by 1/|λ|, so it escapes at the same speed for any ε.',
      note: 'SFN floors |λ| at δ = 0.1 (damping), as practical implementations do.',
      legend: [[HEX.ball, 'GD'], [HEX.opt3, 'saddle-free Newton'], [HEX.neg, 'λ = −2ε'], [HEX.pos, 'λ = 10']],
      action: 'Restart', cam: [3.2, 3.1, 3.9], tgt: [0, .5, 0],
      controls: [{ id: 'le', label: 'curvature ε', min: -2, max: 0, step: .05, value: -1.5, fmt: v => (10 ** v).toFixed(3) }],
      build(ctx) { const s = ctx.stage, S = ctx.S = s.surface(quad(10 ** ctx.vals.le)); s.eigenLines(S, 0, 0, { len: .8, eps: 1e-4 }); s.marker(S, 0, 0, C.white); this.drop(ctx); },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, S = ctx.S; s.clearBalls();
        ctx.balls = [s.ball(S, { start: [.9, .12], step: opt.gd(.08), speed: 2, maxSteps: 20000, stop: outBox }),
          s.ball(S, { start: [.9, .12], step: opt.curv(.5, 'sfn', .1), color: C.opt3, speed: .5, maxSteps: 20000, stop: outBox })];
      },
      frame(ctx) { const [a, b] = ctx.balls; ctx.stat(`GD: ${a.done ? 'escaped in ' + a.steps : a.steps + '…'} steps · SFN: ${b.done ? 'escaped in ' + b.steps : b.steps + '…'} steps`); },
    },
    {
      id: 'monkey', name: 'Monkey saddle', title: 'A degenerate saddle',
      caption: 'The paper’s Fig. 2b: x³ − 3xy². The Hessian is zero at the origin, there are three ways down, and Newton is again pulled into the saddle.',
      note: 'Newton and SFN floor |λ| at δ = 0.1; all three use η = 0.03 here.',
      legend: OPTS.map(o => [o[2], o[0]]).concat([[HEX.flat, 'λ = 0 at the origin']]),
      action: 'Restart', cam: [3.0, 3.4, 3.8], tgt: [0, .6, 0],
      build(ctx) {
        const s = ctx.stage, S = ctx.S = s.surface(monkey); s.marker(S, 0, 0, C.flat);
        s.tube(S, segPts(-.5, 0, .5, 0), C.flat, .014); s.tube(S, segPts(0, -.5, 0, .5), C.flat, .014);
        this.drop(ctx);
      },
      drop(ctx) { race(ctx, ctx.S, [.35, .45], [.03, .03, .03]); },
      frame: report,
    },
    {
      id: 'index', name: 'Why saddles dominate', mode: '2d', title: 'In high dimensions, critical points are mostly saddles',
      caption: 'Hessian spectrum at a critical point, modeled as a random symmetric (GOE) matrix shifted by m. Higher error ⇒ smaller shift ⇒ more negative eigenvalues (larger index α).',
      note: 'Illustration of the Bray & Dean picture the paper cites (Sec. 2): we set the shift with the slider; the shift–error relation is model specific.',
      hint: '', legend: [],
      controls: [{ id: 'm', label: 'shift m (lower error →)', min: -1.5, max: 2.3, step: .05, value: .6 }],
      plot: { xlabel: 'Hessian eigenvalue λ', ylabel: 'density', xlim: [-3.6, 4.4], ylim: [0, .48] },
      charts: [{ id: 'p', title: 'P(all N eigenvalues > 0) · GOE, 20k samples', xlabel: 'dimension N', ylabel: 'probability', logy: true, xlim: [.5, 5.5], ylim: [1e-5, 1], legend: false }],
      build(ctx) {
        const r = rng(3), N = 200, A = Array.from({ length: N }, () => new Array(N).fill(0));
        for (let i = 0; i < N; i++) for (let j = i; j < N; j++) { const v = r.normal() / Math.sqrt(N) * (i === j ? Math.SQRT2 : 1); A[i][j] = A[j][i] = v; }
        ctx.ev = eigSym(A, { sweeps: 30 }).values;
        // P(all eigenvalues > 0) for an N x N GOE matrix: Monte Carlo, 20,000 samples per N (scripts/check-math/dauphin2014identifying.py)
        const pts = [[1, .498], [2, .145], [3, .02715], [4, .0024], [5, 5e-5]];
        ctx.pmin = pts; this.draw(ctx);
      },
      onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const m = ctx.vals.m, ev = ctx.ev.map(v => v + m), P = ctx.plot.clear(), neg = ev.filter(v => v < 0).length / ev.length;
        P.band(-3.6, 0, { color: 'rgba(255,93,115,.07)' });
        const bins = 60, lo = -3.6, w = 8 / bins, c = new Array(bins).fill(0);
        ev.forEach(v => { const k = Math.floor((v - lo) / w); if (k >= 0 && k < bins) c[k]++; });
        P.line(c.map((n, k) => [lo + (k + .5) * w, n / (ev.length * w)]), { color: HEX.pos, width: 2, step: true, label: 'eigenvalues, one sample (N = 200)' });
        P.fn(x => { const u = x - m; return Math.abs(u) < 2 ? Math.sqrt(4 - u * u) / (2 * Math.PI) : 0; }, { color: HEX.white, dash: [4, 4], width: 1.4, label: 'semicircle law' });
        P.vline(0, { color: HEX.neg, text: 'λ = 0' });
        P.draw();
        const ch = ctx.charts.p.clear();
        ch.line(ctx.pmin, { color: HEX.ball, width: 2, dots: 3.5 }); ch.draw();
        ctx.stat(`index α = fraction of negative eigenvalues = ${neg.toFixed(2)} · ${neg === 0 ? 'a local minimum' : neg === 1 ? 'a local maximum' : 'a saddle'} (N = 200)`);
      },
    },
  ],
};
