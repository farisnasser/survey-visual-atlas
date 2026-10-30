// Wang, Ji, Zhou, Liang, Tarokh 2019 — SpiderBoost: SPIDER's recursive estimator with a constant step η = 1/(2L),
// and Prox-SpiderBoost. Components, L, q = |S| = √n come from scripts/check-math/wang2019spiderboost.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const fL = (x, y) => .25 * (x * x - 1) ** 2 + .5 * (y - .25) ** 2;
const fG = (x, y) => [x ** 3 - x, y - .25];
const F = def({ L: fL, g: fG, hess: (x, y) => [3 * x * x - 1, 0, 1], h: v => Math.log1p(v * 3), dom: [-1.8, 1.8, -1.6, 1.6] });
const gi = (D, p, i) => [p[0] ** 3 - p[0] + D.a[i] + D.c[i] * p[0], p[1] - .25 + D.b[i] - D.c[i] * p[1]];
const mb = (D, p, I) => { let gx = 0, gy = 0; for (const i of I) { const g = gi(D, p, i); gx += g[0]; gy += g[1]; } return [gx / I.length, gy / I.length]; };
const n = D => D.a.length;

/** Shared SPIDER/SARAH estimator (Eq. 1), counting stochastic first-order oracle calls; `move` turns v into the step. */
const spiderLike = (D, R, move) => (p, b) => {
  b.k = (b.k ?? -1) + 1;
  if (b.k % D.q === 0) { b.v = F.g(...p); b.sfo += n(D); }
  else { const I = Array.from({ length: D.q }, () => Math.floor(R() * n(D))), u = mb(D, p, I), w = mb(D, b.xp, I); b.v = [u[0] - w[0] + b.v[0], u[1] - w[1] + b.v[1]]; b.sfo += 2 * D.q; }
  b.xp = p.slice(); return move(p, b.v, D);
};
const boost = (p, v, D) => [p[0] - v[0] / (2 * D.L), p[1] - v[1] / (2 * D.L)];
const spider = (p, v, D) => { const nv = Math.max(Math.hypot(...v), 1e-12), e = D.eps / D.L; return [p[0] - e * v[0] / nv, p[1] - e * v[1] / nv]; };
const soft = (z, t) => Math.sign(z) * Math.max(Math.abs(z) - t, 0);
const proxB = (p, v, D) => { const e = 1 / (2 * D.L); return [soft(p[0] - e * v[0], e * D.lam), soft(p[1] - e * v[1], e * D.lam)]; };
const subB = (p, v, D) => { const e = 1 / (2 * D.L); return [p[0] - e * (v[0] + D.lam * Math.sign(p[0])), p[1] - e * (v[1] + D.lam * Math.sign(p[1]))]; };

const PSI = def({ L: (x, y) => fL(x, y) + .3 * (Math.abs(x) + Math.abs(y)), h: v => Math.log1p(v * 3), dom: [-1.8, 1.8, -1.6, 1.6], N: 160 });

export default {
  tabs: [
    {
      id: 'boost', name: 'Big steps', title: 'Same estimator as SPIDER, but a constant step 1/(2L) instead of an ε-sized one',
      caption: 'Both balls use the recursive estimator v_k = v_{k−1} + mean over a minibatch of ∇f_i(x_k) − ∇f_i(x_{k−1}), with a full gradient every q = √n steps. Violet is SPIDER: normalized steps of length ε/L. Coral is SpiderBoost: plain steps x − v/(2L). The chart counts stochastic gradient calls; SpiderBoost needs several times fewer to reach ‖∇f‖ < ε.',
      note: 'n = 196 components f_i = (x² − 1)²/4 + (y − ¼)²/2 + a_i x + b_i y + c_i(x² − y²)/2 with zero-mean a, b, c; every ∇f_i is L-Lipschitz on the box with L = 11.1; q = |S| = 14; ε = 0.05.',
      legend: [[HEX.ball, 'SpiderBoost η = 1/(2L)'], [HEX.opt2, 'SPIDER η = ε/L, normalized'], [HEX.good, 'minima (±1, ¼)']],
      action: 'Run again', cam: [0, 4.2, 4.4], tgt: [0, .3, 0],
      charts: [{ id: 'g', title: '‖∇f(x_k)‖ vs stochastic gradient calls', xlabel: 'SFO calls', ylabel: '‖∇f‖', logy: true, ylim: [1e-3, 5], h: 240 }],
      build(ctx) { const s = ctx.stage; s.surface(F); s.marker(F, 1, .25, C.good); s.marker(F, -1, .25, C.good); s.eigenLines(F, 0, .25, { len: .5 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, seed = Date.now() % 1e6; s.clearBalls();
        ctx.b = [s.ball(F, { start: [-1.6, 1.4], step: spiderLike(D, rng(seed), boost), speed: .5, maxSteps: 400 }),
          s.ball(F, { start: [-1.6, 1.4], step: spiderLike(D, rng(seed + 1), spider), color: C.opt2, speed: 3, maxSteps: 4000 })];
        for (const q of ctx.b) { q.sfo = 0; q.h = [[0, Math.hypot(...F.g(...q.p))]]; q.hit = null; q.onStep = z => { const gn = Math.hypot(...F.g(...z.p)); z.h.push([z.sfo, gn]); if (z.hit === null && gn < D.eps) z.hit = z.sfo; }; }
      },
      frame(ctx) {
        const [a, b] = ctx.b || []; if (!a) return;
        ctx.charts.g.clear().set({ xlim: [0, Math.max(3000, Math.min(a.sfo, b.sfo))] }).hline(ctx.data.eps, { color: HEX.muted, dash: [4, 4], text: 'ε' })
          .line(b.h, { color: HEX.opt2, width: 1.4, label: 'SPIDER' }).line(a.h, { color: HEX.ball, width: 1.8, label: 'SpiderBoost' }).draw();
        const t = z => z.hit === null ? 'not yet' : `${z.hit} calls`;
        ctx.stat(`‖∇f‖ < ε: SpiderBoost ${t(a)} · SPIDER ${t(b)} · now: SpiderBoost ${a.sfo} calls, ‖∇f‖ = ${Math.hypot(...F.g(...a.p)).toFixed(4)}`);
      },
    },
    {
      id: 'prox', name: 'Prox step', title: 'Prox-SpiderBoost handles a nonsmooth regularizer exactly',
      caption: 'Surface: Ψ = f + 0.3(|x| + |y|), with kinks along both axes. Coral is Prox-SpiderBoost (Algorithm 2): x ← prox_{ηh}(x − ηv), i.e. soft-thresholding, so it lands on the kink with y exactly 0. White uses the same estimator but just adds a subgradient of h; it chatters around y = 0 and never reaches it.',
      note: 'Same components and η = 1/(2L). Because λ = 0.3 exceeds the pull |0.25| of f toward y = ¼, the composite minimizers are (±0.787, 0). Theorem 2 measures stationarity with the gradient mapping G_η, not ∇Ψ.',
      legend: [[HEX.ball, 'Prox-SpiderBoost'], [HEX.opt3, 'subgradient + same estimator'], [HEX.good, 'minimizers of Ψ']],
      action: 'Run again', cam: [3.4, 3.4, 3.4], tgt: [0, .2, 0],
      charts: [{ id: 'y', title: '|y_k| (distance to the kink)', xlabel: 'iteration k', ylabel: '|y|', logy: true, ylim: [1e-6, 3], h: 220 }],
      build(ctx) { const s = ctx.stage, D = ctx.data; s.surface(PSI); s.marker(PSI, D.xs, 0, C.good); s.marker(PSI, -D.xs, 0, C.good); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, seed = Date.now() % 1e6; s.clearBalls();
        ctx.b = [s.ball(PSI, { start: [-1.6, 1.4], step: spiderLike(D, rng(seed), proxB), speed: .4, maxSteps: 600 }),
          s.ball(PSI, { start: [-1.6, 1.4], step: spiderLike(D, rng(seed), subB), color: C.opt3, speed: .4, maxSteps: 600 })];
        for (const q of ctx.b) { q.sfo = 0; q.h = [[0, Math.abs(q.p[1])]]; q.onStep = z => z.h.push([z.steps, Math.max(Math.abs(z.p[1]), 1e-6)]); }
      },
      frame(ctx) {
        const [a, b] = ctx.b || []; if (!a) return;
        ctx.charts.y.clear().set({ xlim: [0, Math.max(50, a.steps)] }).line(b.h, { color: HEX.opt3, width: 1.2, label: 'subgradient' }).line(a.h, { color: HEX.ball, width: 1.8, label: 'prox' }).draw();
        ctx.stat(`step ${a.steps} · prox: (${a.p[0].toFixed(4)}, ${a.p[1] === 0 ? '0 exactly' : a.p[1].toExponential(2)}) · subgradient: y = ${b.p[1].toExponential(2)}`);
      },
    },
  ],
};
