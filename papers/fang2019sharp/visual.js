// Fang, Lin, Zhang 2019 — plain SGD with an anchor-ball rule (Algorithm 2) reaches second-order stationarity in Õ(ε^{-3.5})
// stochastic gradients; the coupling argument behind it. Checked in scripts/check-math/fang2019sharp.py.
import { def, C, HEX, circlePts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const F = def({ L: (x, y) => .25 * (x * x - 1) ** 2 + .5 * y * y, g: (x, y) => [x ** 3 - x, y], hess: (x, y) => [3 * x * x - 1, 0, 1], h: v => Math.log1p(v * 3), dom: [-1.7, 1.7, -1.5, 1.5] });
const ETA = .02, SIG = 1, B = .25, K0 = 400;
const ballNoise = R => { const a = R() * 2 * Math.PI, r = SIG * Math.sqrt(R()); return [r * Math.cos(a), r * Math.sin(a)]; };

export default {
  tabs: [
    {
      id: 'anchor', name: 'Anchor ball', title: 'Plain SGD, judged by whole runs inside a small ball',
      caption: 'SGD with noise drawn uniformly from a disk (a dispersive distribution) starts on the saddle’s stable set. Algorithm 2 keeps an anchor x₀ and a ball of radius B around it (amber). Leaving the ball moves the anchor there and restarts the count; K₀ consecutive steps inside the ball end the run, and the output is the average of those K₀ iterates (green).',
      note: 'f = (x² − 1)²/4 + y²/2, saddle at 0 with λ = (−1, 1). η = 0.02, B = 0.25, K₀ = 400, noise radius 1. In the proof each exit comes with a guaranteed decrease of f (Prop. 9) and a full stay certifies the averaged output (Prop. 10).',
      legend: [[HEX.ball, 'SGD iterate'], [HEX.flat, 'current anchor ball'], [HEX.good, 'output: average of the last K₀ iterates'], [HEX.neg, 'λ < 0 at the saddle']],
      action: 'Run again', cam: [0, 4.2, 4.2], tgt: [0, .3, 0],
      build(ctx) { const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .6 }); this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6); s.clearBalls(); (ctx.objs || []).forEach(o => o && s.remove(o)); ctx.objs = [];
        const ring = x0 => { if (ctx.ring) s.remove(ctx.ring); ctx.ring = s.tube(F, circlePts(x0[0], x0[1], B), C.flat, .01, .03); };
        ctx.st = { x0: [0, 1.2], k: 0, buf: [], exits: 0, out: null }; ring(ctx.st.x0);
        ctx.b = s.ball(F, { start: [0, 1.2], speed: 4, maxSteps: 60000, maxTrail: 60000, stop: () => !!ctx.st.out, step: p => {
          const S = ctx.st, [gx, gy] = F.g(...p), z = ballNoise(R), q = [p[0] - ETA * (gx + z[0]), p[1] - ETA * (gy + z[1])];
          S.k++; S.buf.push(q);
          if (Math.hypot(q[0] - S.x0[0], q[1] - S.x0[1]) > B) { S.x0 = q.slice(); S.k = 0; S.buf = []; S.exits++; ring(S.x0); }
          if (S.k >= K0) { const m = S.buf.slice(-K0).reduce((a, v) => [a[0] + v[0] / K0, a[1] + v[1] / K0], [0, 0]); S.out = m; ctx.objs.push(s.marker(F, ...m, C.good, .09)); }
          return q;
        } });
      },
      frame(ctx) {
        const S = ctx.st, b = ctx.b; if (!b) return;
        ctx.stat(S.out ? `stopped after ${b.steps} SGD steps and ${S.exits} anchor moves · output (${S.out.map(v => v.toFixed(3)).join(', ')}), ‖∇f‖ = ${Math.hypot(...F.g(...S.out)).toFixed(3)}, λ_min = ${Math.min(3 * S.out[0] ** 2 - 1, 1).toFixed(2)}`
          : `step ${b.steps} · anchor moves ${S.exits} · steps inside the current ball ${S.k}/${K0}`);
      },
    },
    {
      id: 'coupling', name: 'Coupling', mode: '2d', title: 'Two runs with the same noise cannot both stay near a saddle',
      caption: 'The notes’ example on q(u, v) = (−u² + v²)/2 with η = 0.1: two SGD runs start 0.002 apart along the unstable direction u and use identical noise. The noise cancels in their difference, which grows exactly as 0.002·1.1^k. After k = 25 steps they are 0.0217 apart, more than the ball diameter 2B = 0.02, so at least one has left: the set of “stuck” starts is a thin slab, and dispersive noise rarely lands the iterate in it.',
      note: 'Pedagogical quadratic (not lower bounded); Appendix C extends the argument with Hessian-Lipschitz and component-smoothness bounds. Right: both paths in the (u, v) plane with the ball of radius B = 0.01.',
      hint: '', legend: [[HEX.ball, 'distance between the coupled runs'], [HEX.neg, '2B = 0.02']],
      plot: { xlabel: 'step k', ylabel: '‖x_k − x′_k‖', logy: true, xlim: [0, 40], ylim: [1e-3, .2] },
      charts: [{ id: 'paths', title: 'the two runs (same noise)', xlabel: 'u', ylabel: 'v', xlim: [-.06, .06], ylim: [-.06, .06], h: 260, legend: false }],
      actions: [{ label: 'New noise', fn: ctx => ctx.shell.tab.build(ctx) }],
      build(ctx) {
        const R = rng(Date.now() % 1e6), A = [[.001, .002]], Bp = [[-.001, .002]], dist = [[0, .002]]; let k25 = null;
        for (let k = 1; k <= 40; k++) {
          const n1 = .01 * R.normal(), n2 = .01 * R.normal(), a = A[k - 1], b = Bp[k - 1];
          A.push([1.1 * a[0] - .1 * n1, .9 * a[1] - .1 * n2]); Bp.push([1.1 * b[0] - .1 * n1, .9 * b[1] - .1 * n2]);
          const d = Math.hypot(A[k][0] - Bp[k][0], A[k][1] - Bp[k][1]); dist.push([k, d]); if (k25 == null && d > .02) k25 = k;
        }
        ctx.plot.clear().hline(.02, { color: HEX.neg, dash: [5, 4], text: '2B' }).fn(k => .002 * 1.1 ** k, { color: HEX.muted, dash: [2, 4] }).line(dist, { color: HEX.ball, width: 2.4, dots: 3 }).vline(k25, { color: HEX.muted, text: `k = ${k25}` }).draw();
        ctx.charts.paths.clear().line(circleXY(.01), { color: HEX.flat, width: 1.2 }).line(A, { color: HEX.ball, width: 1.6 }).line(Bp, { color: HEX.opt2, width: 1.6 }).draw();
        ctx.stat(`first k with distance > 2B: ${k25} · distance then ${dist[k25][1].toFixed(7)} (= 0.002·1.1^${k25} = ${(.002 * 1.1 ** k25).toFixed(7)})`);
      },
    },
  ],
};
function circleXY(r) { return Array.from({ length: 81 }, (_, i) => [r * Math.cos(i / 40 * Math.PI), r * Math.sin(i / 40 * Math.PI)]); }
