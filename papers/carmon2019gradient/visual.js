// Carmon & Duchi 2019 — plain gradient descent finds the global minimum of the non-convex cubic-regularized quadratic
// f(x) = ½xᵀAx + bᵀx + (ρ/3)‖x‖³. Checked in scripts/check-math/carmon2019gradient.py (data.json: the d = 1000 runs).
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const LAM = [-.5, .6], RHO = 1, ETA = .1;
const mk = b => {
  const L = (x, y) => .5 * (LAM[0] * x * x + LAM[1] * y * y) + b[0] * x + b[1] * y + RHO / 3 * Math.hypot(x, y) ** 3;
  const g = (x, y) => { const r = Math.hypot(x, y); return [LAM[0] * x + b[0] + RHO * r * x, LAM[1] * y + b[1] + RHO * r * y]; };
  return def({ L, g, h: v => v, dom: [-1, 1, -1, 1] });
};
/** All stationary points: x = −b/(λ + ρr) with |x| = r (roots of the secular function, skipping poles), plus the hard-case circle. */
function stationary(b) {
  const out = [], ph = r => Math.hypot(b[0] / (LAM[0] + RHO * r), b[1] / (LAM[1] + RHO * r)) - r;
  let prev = 1e-6, pv = ph(prev);
  for (let k = 1; k <= 20000; k++) {
    const r = 1e-6 + 3 * k / 20000, v = ph(r);
    if (Math.sign(v) !== Math.sign(pv) && Math.sign(LAM[0] + RHO * r) === Math.sign(LAM[0] + RHO * prev) && Math.sign(LAM[1] + RHO * r) === Math.sign(LAM[1] + RHO * prev) && isFinite(v) && isFinite(pv)) {
      let lo = prev, hi = r; for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (Math.sign(ph(m)) === Math.sign(ph(lo))) lo = m; else hi = m; }
      const rr = (lo + hi) / 2; out.push([-b[0] / (LAM[0] + RHO * rr), -b[1] / (LAM[1] + RHO * rr)]);
    }
    prev = r; pv = v;
  }
  if (Math.abs(b[0]) < 1e-15) { const rr = -LAM[0] / RHO, x2 = -b[1] / (LAM[1] + RHO * rr); if (x2 * x2 < rr * rr) { const x1 = Math.sqrt(rr * rr - x2 * x2); out.push([x1, x2], [-x1, x2]); } }
  return out;
}

export default {
  tabs: [
    {
      id: 'gd', name: 'GD finds the global step', title: 'Saddles and a bad local minimum, yet gradient descent from 0 ends at the global minimum',
      caption: 'f = ½xᵀAx + bᵀx + (ρ/3)‖x‖³ with A = diag(−0.5, 0.6) is the subproblem of cubic-regularized Newton. Started at x₀ = 0 (Assumption B), gradient descent keeps ‖x_t‖ non-decreasing (Lemma 2.3) and converges to the global minimum whenever b has a component b⁽¹⁾ ≠ 0 on the bottom eigenvector. Set b⁽¹⁾ = 0 (hard case): GD stays on the x₂-axis and stops at a saddle; a tiny random perturbation of b fixes it (Theorem 3.2).',
      note: 'ρ = 1, η = 0.1 (Assumption A), b = (b⁽¹⁾, −0.5). Markers: every stationary point; green = global minimum. The perturbation adds 10⁻³·q, q a random unit vector.',
      legend: [[HEX.ball, 'GD on f'], [HEX.opt2, 'GD on the perturbed b̃'], [HEX.good, 'global minimum'], [HEX.white, 'other stationary points']],
      action: 'Run again', cam: [0, 4.4, 3.8], tgt: [0, .3, 0],
      controls: [{ id: 'b1', label: 'b⁽¹⁾ (component on v₁)', min: 0, max: .2, step: .005, value: .06, digits: 3 }],
      charts: [{ id: 'nrm', title: '‖x_t‖ (non-decreasing, Lemma 2.3)', xlabel: 'step t', ylabel: '‖x_t‖', ylim: [0, .8], h: 170 },
        { id: 'gap', title: 'f(x_t) − f*', xlabel: 'step t', ylabel: 'gap', logy: true, ylim: [1e-12, 1], h: 170, legend: false }],
      build(ctx) {
        const b = [ctx.vals.b1, -.5]; ctx.d = mk(b); const s = ctx.stage; s.surface(ctx.d);
        const S = stationary(b); ctx.fstar = Math.min(...S.map(p => ctx.d.L(...p)));
        S.forEach(p => s.marker(ctx.d, ...p, Math.abs(ctx.d.L(...p) - ctx.fstar) < 1e-9 ? C.good : C.white, .07));
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, d = ctx.d, b = [ctx.vals.b1, -.5]; s.clearBalls();
        const th = (Date.now() % 1000) / 1000 * 2 * Math.PI, bt = [b[0] + 1e-3 * Math.cos(th), b[1] + 1e-3 * Math.sin(th)], dt = mk(bt);
        const step = dd => (p) => { const [gx, gy] = dd.g(...p); return [p[0] - ETA * gx, p[1] - ETA * gy]; };
        ctx.a = s.ball(d, { start: [0, 0], step: step(d), speed: 2, maxSteps: 3000 });
        ctx.p = ctx.vals.b1 === 0 ? s.ball(d, { start: [0, 0], step: step(dt), color: C.opt2, speed: 2, maxSteps: 3000 }) : null;
        for (const q of [ctx.a, ctx.p].filter(Boolean)) { q.h = [[0, 0, 1]]; q.onStep = z => z.h.push([z.steps, Math.hypot(...z.p), Math.max(d.L(...z.p) - ctx.fstar, 1e-12)]); }
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        const a = ctx.a, p = ctx.p; if (!a) return;
        const xm = Math.max(40, a.steps);
        const N = ctx.charts.nrm.clear().set({ xlim: [0, xm] }).line(a.h.map(r => [r[0], r[1]]), { color: HEX.ball, width: 2, label: 'GD' });
        const G = ctx.charts.gap.clear().set({ xlim: [0, xm] }).line(a.h.map(r => [r[0], r[2]]), { color: HEX.ball, width: 2 });
        if (p) { N.line(p.h.map(r => [r[0], r[1]]), { color: HEX.opt2, width: 1.6, label: 'perturbed' }); G.line(p.h.map(r => [r[0], r[2]]), { color: HEX.opt2, width: 1.6 }); }
        N.draw(); G.draw();
        let mono = true; for (let i = 1; i < a.h.length; i++) if (a.h[i][1] < a.h[i - 1][1] - 1e-12) mono = false;
        const gp = q => { const v = q.loss() - ctx.fstar; return v < 1e-12 ? '< 10⁻¹²' : v.toExponential(2); };
        ctx.stat(`GD: f − f* = ${gp(a)}, ‖x‖ monotone: ${mono ? 'yes' : 'no'}${ctx.vals.b1 === 0 ? ` · hard case: x₁ stays ${a.p[0]} (a saddle); perturbed GD: f − f* = ${gp(p)} (off only by the 10⁻³ perturbation)` : ''}`);
      },
    },
    {
      id: 'regimes', name: 'Stalls', mode: '2d', title: 'Small ‖b‖: GD lingers near weakly curved saddles before converging',
      caption: 'A d = 1000 instance like the paper’s Fig. 2: λ₁ = −0.2, the rest equally spaced in [−0.18, 1], b ∝ (0.01, 1, 1, …), x₀ = 0. Large ‖b‖ converges linearly at once. Small ‖b‖ makes the problem ill-conditioned: the gap sits on a plateau near a saddle whose norm is close to γ/ρ = 1, where the negative curvature is weak, then drops.',
      note: 'Our constants: ρ = 0.2, η = 0.1, 6000 steps (the paper’s exact Fig. 2 parameters are not reproduced). ‖x_t‖ is non-decreasing in every run (Lemma 2.3).',
      hint: '', legend: [],
      controls: [{ id: 'view', type: 'seg', label: 'show', value: 'gap', options: [['gap', 'f(x_t) − f*'], ['norm', '‖x_t‖']] }],
      plot: { xlabel: 'iteration t', ylabel: '', logy: true, xlim: [0, 6000] },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, P = ctx.plot.clear(), cols = [HEX.good, HEX.pos, HEX.ball, HEX.opt2, HEX.flat], gap = ctx.vals.view === 'gap';
        P.set({ ylabel: gap ? 'f(x_t) − f*' : '‖x_t‖', logy: gap, ylim: gap ? [1e-16, 10] : [0, 2] });
        Object.entries(D.runs).forEach(([bn, R], i) => P.line((gap ? R.gap : R.norm).map((v, k) => [k * D.every, v]), { color: cols[i], width: 2, label: `‖b‖ = ${bn}` }));
        P.draw();
        const st = Object.entries(D.runs).map(([bn, R]) => { const i = R.gap.findIndex(v => v < 1e-8); return `${bn}: ${i < 0 ? `not within ${D.T}` : i * D.every}`; }).join(' · ');
        ctx.stat(`iterations until f − f* < 10⁻⁸ — ‖b‖ = ${st}`);
      },
    },
  ],
};
