// Defazio et al. 2024 — Schedule-Free SGD: gradients at y = (1 − β)z + βx, SGD on z, x = running average of z.
// Eqs. (3)-(5), the notes' worked steps and the anytime comparison are checked in scripts/check-math/defazio2024road.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const A = [2, .25], SIG = .8, START = [1.8, -1.6];
const quad = def({ L: (x, y) => .5 * (A[0] * x * x + A[1] * y * y), g: (x, y) => [A[0] * x, A[1] * y], dom: [-2.2, 2.2, -2.2, 2.2], h: v => Math.log1p(v) });
const loss = p => quad.L(p[0], p[1]);

export default {
  tabs: [
    {
      id: 'three', name: 'Three sequences', title: 'Step with one sequence, report another, take gradients in between',
      caption: 'Schedule-Free SGD on a noisy quadratic. The base sequence z (violet) takes plain SGD steps with a constant learning rate and keeps jittering. The reported sequence x (coral) is the running average of all z’s. Gradients are taken at y (white), a mix that sits mostly on x: y = (1 − β)z + βx. No learning-rate schedule and no stopping time are needed, yet x keeps getting closer to the minimum.',
      note: 'Loss ½(2x² + 0.25y²), Gaussian gradient noise σ = 0.8 per coordinate, γ = 0.4, c_{t+1} = 1/(t + 1). β = 0 is Polyak–Ruppert averaging, β = 1 primal averaging.',
      legend: [[HEX.opt2, 'z: base SGD sequence'], [HEX.opt3, 'y: where gradients are taken'], [HEX.ball, 'x: running average (output)']],
      action: 'Restart', cam: [0, 4.4, 4.2], tgt: [0, .1, 0],
      controls: [{ id: 'beta', type: 'seg', label: 'β', value: .9, options: [[0, '0'], [.5, '0.5'], [.9, '0.9'], [.98, '0.98']] }],
      charts: [{ id: 'L', title: 'loss of x (coral) and z (violet), log', xlabel: 'step', ylabel: 'f', logy: true, ylim: [1e-5, 10], xlim: [0, 3000], h: 230, legend: false }],
      build(ctx) { ctx.stage.surface(quad); ctx.stage.marker(quad, 0, 0, C.good, .1); this.drop(ctx); },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, beta = +ctx.vals.beta, R = rng(5), st = { z: START.slice(), x: START.slice(), y: START.slice(), t: 0 }; s.clearBalls(); ctx.st = st; ctx.H = { x: [], z: [] };
        ctx.bx = s.ball(quad, { start: START, speed: 3, maxSteps: 3000, maxTrail: 3000, step: () => {
          st.t++; st.y = [(1 - beta) * st.z[0] + beta * st.x[0], (1 - beta) * st.z[1] + beta * st.x[1]];
          const g = quad.g(...st.y); st.z = [st.z[0] - .4 * (g[0] + SIG * R.normal()), st.z[1] - .4 * (g[1] + SIG * R.normal())];
          const c = 1 / (st.t + 1); st.x = [(1 - c) * st.x[0] + c * st.z[0], (1 - c) * st.x[1] + c * st.z[1]];
          ctx.H.x.push([st.t, loss(st.x)]); ctx.H.z.push([st.t, loss(st.z)]); return st.x.slice();
        } });
        ctx.bz = s.ball(quad, { start: START, speed: 3, maxSteps: 3000, maxTrail: 400, color: C.opt2, step: () => st.z.slice() });
        ctx.by = s.ball(quad, { start: START, speed: 3, maxSteps: 3000, maxTrail: 1, trail: false, r: .045, color: C.opt3, step: () => st.y.slice() });
      },
      frame(ctx) {
        const thin = a => a.filter((_, i) => i % 3 === 0);
        ctx.charts.L.clear().line(thin(ctx.H.z), { color: HEX.opt2, width: 1, alpha: .7 }).line(ctx.H.x, { color: HEX.ball, width: 2 }).draw();
        const n = ctx.H.x.length, lz = ctx.H.z.slice(-300).reduce((a, h) => a + h[1], 0) / Math.max(1, Math.min(300, n));
        ctx.stat(`step ${ctx.st.t} · β = ${ctx.vals.beta} · loss of x ${n ? ctx.H.x[n - 1][1].toExponential(2) : '–'} · recent mean loss of z ${lz.toExponential(2)}`);
      },
    },
    {
      id: 'anytime', name: 'No horizon needed', mode: '2d', title: 'One run matches schedules tuned for each stopping time',
      caption: 'Noisy 20-dimensional least squares. Each dashed curve is SGD with a linear-decay schedule tuned for one stopping time T (250, 1000 or 4000 steps), so it only does well at its own T. A single Schedule-Free run (coral), which never knows T, is within 1.6× of the tuned schedule at T = 250 and about 1.8× better at T = 1000 and 4000. Constant-step SGD (grey) stalls at its noise floor.',
      note: 'Offline: 2000 samples, batch 4, each method’s step chosen from the same grid; Schedule-Free uses β = 0.9 and one step for all horizons. The paper reports the same pattern on deep-learning benchmarks and an AlgoPerf 2024 win (self-tuning track) for Schedule-Free AdamW.',
      hint: '', legend: [[HEX.ball, 'Schedule-Free SGD'], [HEX.opt2, 'linear decay, tuned per T'], [HEX.muted, 'constant-step SGD']],
      plot: { xlabel: 'step', ylabel: 'F(x) − F* (log)', logy: true, xlim: [0, 4000], ylim: [1e-4, 10], legend: false },
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear(), xs = a => a.map((v, i) => [10 * i + 1, v]);
        P.line(xs(D.const.gap), { color: HEX.muted, width: 1.2 });
        for (const T of D.Ts) { P.line(xs(D.lin[T].gap).filter(p => p[0] <= T), { color: HEX.opt2, width: 1.4, dash: [5, 3] }); P.vline(T, { color: HEX.opt2, dash: [2, 4], alpha: .5, text: `T = ${T}` }); }
        P.line(xs(D.sf.gap), { color: HEX.ball, width: 2.2 }).draw();
        ctx.stat(D.Ts.map(T => `T = ${T}: tuned linear decay ${D.lin[T].end.toExponential(2)} (γ ${D.lin[T].gamma}) vs Schedule-Free ${D.lin[T].sf.toExponential(2)}`).join(' · '));
      },
    },
  ],
};
