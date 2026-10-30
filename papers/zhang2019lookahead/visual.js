// Zhang, Lucas, Hinton, Ba 2019 — Lookahead: k fast steps, then the slow weights move a fraction α towards them.
// Noisy quadratic model (Sec. 3.1) and Proposition 2 are checked in scripts/check-math/zhang2019lookahead.py
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const A = [1, .3], S2 = [1, 1], GAM = .6;    // L^(x) = 1/2 (x - c)^T A (x - c), c ~ N(0, diag S2)
const quad = def({ L: (x, y) => .5 * (A[0] * (x * x + S2[0]) + A[1] * (y * y + S2[1])), g: (x, y) => [A[0] * x, A[1] * y], dom: [-3, 3, -3, 3], h: v => v });
const vSGD = (g, a, s2) => { const r = 1 - g * a; return g * g * a * a * s2 / (1 - r * r); };
const vLA = (g, a, s2, k, al) => { const r = 1 - g * a; return al * al * (1 - r ** (2 * k)) / (al * al * (1 - r ** (2 * k)) + 2 * al * (1 - al) * (1 - r ** k)) * vSGD(g, a, s2); };
const excess = (x, y) => .5 * (A[0] * x * x + A[1] * y * y);
const sgStep = (R, p) => [p[0] - GAM * A[0] * (p[0] - Math.sqrt(S2[0]) * R.normal()), p[1] - GAM * A[1] * (p[1] - Math.sqrt(S2[1]) * R.normal())];

export default {
  tabs: [
    {
      id: 'noisy', name: 'Noisy quadratic', title: 'Fast weights jitter, slow weights stay calm',
      caption: 'The paper’s noisy quadratic: every step sees the bowl shifted by random noise. SGD (violet) keeps bouncing around the minimum. Lookahead’s fast weights (white) do the same for k steps, then the slow weights (coral) move only a fraction α of the way to where they ended, and the fast weights restart there. The slow weights settle much closer to the minimum at the same learning rate.',
      note: 'Expected loss ½Σaᵢ(xᵢ² + σᵢ²), a = (1, 0.3), σ² = (1, 1), learning rate 0.6, fresh noise every step. Readouts: time-averaged excess loss ½Σaᵢxᵢ² against Proposition 2.',
      legend: [[HEX.opt2, 'SGD'], [HEX.opt3, 'Lookahead fast weights'], [HEX.ball, 'Lookahead slow weights']],
      action: 'Restart', cam: [0, 4.4, 4.4], tgt: [0, .1, 0],
      controls: [{ id: 'k', type: 'seg', label: 'inner steps k', value: 5, options: [[1, '1'], [5, '5'], [10, '10'], [20, '20']] }, { id: 'al', label: 'slow step α', min: .1, max: 1, step: .05, value: .5 }],
      build(ctx) { ctx.stage.surface(quad); ctx.stage.marker(quad, 0, 0, C.good, .1); this.drop(ctx); },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, k = +ctx.vals.k, al = ctx.vals.al, R1 = rng(11), R2 = rng(12), st = [2.4, 2.4]; s.clearBalls();
        ctx.acc = { s: 0, l: 0, n: 0 };
        ctx.sgd = s.ball(quad, { start: st, speed: 2, maxSteps: 1e6, step: p => sgStep(R1, p), color: C.opt2, maxTrail: 400 });
        const la = ctx.la = s.ball(quad, { start: st, speed: 2, maxSteps: 1e6, maxTrail: 3000, step: (p, b) => {
          b.th = sgStep(R2, b.th || p.slice()); b.i = (b.i || 0) + 1;
          if (b.i % k) return p;
          const q = [p[0] + al * (b.th[0] - p[0]), p[1] + al * (b.th[1] - p[1])]; b.th = q.slice(); return q;
        } });
        ctx.fast = s.ball(quad, { start: st, speed: 2, maxSteps: 1e6, color: C.opt3, r: .04, maxTrail: 300, step: () => (la.th || la.p).slice() });
        ctx.sgd.onStep = b => { if (b.steps > 200) { ctx.acc.s += excess(...b.p); ctx.acc.l += excess(...la.p); ctx.acc.n++; } };
      },
      frame(ctx) {
        const k = +ctx.vals.k, al = ctx.vals.al, n = Math.max(1, ctx.acc.n);
        const ps = .5 * (A[0] * vSGD(GAM, A[0], S2[0]) + A[1] * vSGD(GAM, A[1], S2[1])), pl = .5 * (A[0] * vLA(GAM, A[0], S2[0], k, al) + A[1] * vLA(GAM, A[1], S2[1], k, al));
        ctx.stat(`step ${ctx.sgd.steps} · mean excess loss: SGD ${(ctx.acc.s / n).toFixed(3)} (Prop. 2: ${ps.toFixed(3)}) · Lookahead slow ${(ctx.acc.l / n).toFixed(3)} (Prop. 2: ${pl.toFixed(3)})`);
      },
    },
    {
      id: 'ratio', name: 'Variance ratio', mode: '2d', title: 'Lookahead’s steady-state variance is always smaller',
      caption: 'Proposition 2: at the same learning rate, the slow weights’ stationary variance is V_SGD times a factor that is below 1 for every α in (0, 1) and equals 1 at α = 1 (plain SGD). Smaller α and more inner steps k shrink it further. The paper adds that tuned settings reach SGD’s risk level faster (Fig. 3).',
      note: 'Per coordinate with r = 1 − γa. The formulas match a 200 000-run Monte Carlo to within 2% in the check script.',
      hint: '', legend: [[HEX.pos, 'k = 1'], [HEX.ball, 'k = 5'], [HEX.opt2, 'k = 20']],
      controls: [{ id: 'ga', label: 'γ · a (step × curvature)', min: .05, max: 1.9, step: .05, value: .6 }],
      plot: { xlabel: 'slow step α', ylabel: 'V_LA / V_SGD', xlim: [0, 1], ylim: [0, 1.05], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const ga = ctx.vals.ga, P = ctx.plot.clear().hline(1, { color: HEX.muted, dash: [4, 4], text: 'SGD' });
        [[1, HEX.pos], [5, HEX.ball], [20, HEX.opt2]].forEach(([k, c]) => P.fn(al => al <= 0 ? NaN : vLA(ga, 1, 1, k, al) / vSGD(ga, 1, 1), { color: c, width: 2.2 }));
        P.draw();
        ctx.stat(`γa = ${ga.toFixed(2)} · at α = 0.5: k = 1 → ${(vLA(ga, 1, 1, 1, .5) / vSGD(ga, 1, 1)).toFixed(3)}, k = 5 → ${(vLA(ga, 1, 1, 5, .5) / vSGD(ga, 1, 1)).toFixed(3)}, k = 20 → ${(vLA(ga, 1, 1, 20, .5) / vSGD(ga, 1, 1)).toFixed(3)}`);
      },
    },
  ],
};
