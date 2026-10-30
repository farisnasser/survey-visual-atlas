// Zhang & Li 2021 — gradient-only Hessian power method (Algorithm 1) vs perturbed GD, on the paper's Fig. 1 landscape.
// The 300 initial samples, the share curves and the stall fractions come from scripts/check-math/zhang2021escape.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const fL = (x, y) => x ** 4 / 16 - x * x / 2 + 9 * y * y / 8;
const fG = (x, y) => [x ** 3 / 4 - x, 9 * y / 4];
const F = def({ L: fL, g: fG, hess: (x, y) => [.75 * x * x - 1, 0, 2.25], h: v => Math.log1p(Math.max(v + 1.02, 0) * 2.5), dom: [-2.6, 2.6, -1, 1], square: true });
const T1 = 30, T2 = 90, SHOW = .45;   // directions of Algorithm 1 are drawn on a circle of radius SHOW (actual r = 0.1)
const rgb = h => [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16) / 255);
const aligned = v => Math.abs(v[0]) / Math.hypot(...v) > Math.cos(5 * Math.PI / 180);

export default {
  tabs: [
    {
      id: 'find', name: 'Find the curvature', title: 'A power method made of gradient differences finds the escape direction in 30 steps',
      caption: 'Both clouds start as 300 uniform samples in a ball of radius r = 0.1 around the saddle of f = x₁⁴/16 − x₁²/2 + 9x₂²/8 (the paper’s Fig. 1). Coral: Algorithm 1 repeatedly applies y ← y − η‖y‖/r·(∇f(r·y/‖y‖) − ∇f(0)), a renormalized Hessian power step, and its directions collapse onto ±e₁. At t = 30 each coral sample kicks along its direction and continues with plain GD; violet (perturbed GD) runs GD all 90 steps.',
      note: 'η = 0.05, r = 0.1, M = 300 as in the paper. Coral dots before t = 30 show directions drawn at radius 0.45 for visibility. The kick length 0.27 = ¼√(ε/ρ) of Lemma 6 is our choice for the equal-budget comparison.',
      legend: [[HEX.ball, 'Algorithm 1 (then kick + GD)'], [HEX.opt2, 'perturbed GD'], [HEX.good, 'minima (±2, 0)']],
      action: 'Run again', cam: [0, 4.4, 4.2], tgt: [0, .5, 0],
      charts: [{ id: 'al', title: 'share within 5° of ±e₁', xlabel: 'iteration t', ylabel: 'share', xlim: [0, 90], ylim: [0, 1.02], h: 170 },
        { id: 'hist', title: 'decrease f(0) − f at t = 90', xlabel: 'decrease', ylabel: 'paths', xlim: [0, 1.02], h: 170 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F); s.eigenLines(F, 0, 0, { len: .7 });
        s.marker(F, 2, 0, C.good); s.marker(F, -2, 0, C.good);
        this.drop(ctx);
      },
      drop(ctx) {
        const D = ctx.data;
        ctx.st = { t: 0, acc: 0, Y: D.y0.map(p => p.slice()), X: D.x0.map(p => p.slice()), sa: [], sp: [], pts: null };
        this.record(ctx); this.draw(ctx);
      },
      step(ctx) {
        const S = ctx.st, eta = ctx.data.eta, r = ctx.data.r, g0 = fG(0, 0);
        if (S.t < T1) S.Y = S.Y.map(y => { const n = Math.hypot(...y), gg = fG(r * y[0] / n, r * y[1] / n); return [y[0] - eta * n / r * (gg[0] - g0[0]), y[1] - eta * n / r * (gg[1] - g0[1])]; });
        else {
          if (S.t === T1) { const k = ctx.data.kick; S.Y = S.Y.map(y => { const n = Math.hypot(...y), a = [k * y[0] / n, k * y[1] / n], b = [-a[0], -a[1]]; return fL(...a) <= fL(...b) ? a : b; }); }
          S.Y = S.Y.map(p => { const gg = fG(...p); return [p[0] - eta * gg[0], p[1] - eta * gg[1]]; });
        }
        S.X = S.X.map(p => { const gg = fG(...p); return [p[0] - eta * gg[0], p[1] - eta * gg[1]]; });
        S.t++; this.record(ctx);
      },
      record(ctx) { const S = ctx.st; S.sa.push([S.t, S.t <= T1 ? S.Y.filter(aligned).length / S.Y.length : NaN]); S.sp.push([S.t, S.X.filter(aligned).length / S.X.length]); },
      draw(ctx) {
        const s = ctx.stage, S = ctx.st;
        if (S.pts) S.pts.forEach(p => s.remove(p));
        const ya = S.t < T1 ? S.Y.map(y => { const n = Math.hypot(...y); return [SHOW * y[0] / n, SHOW * y[1] / n]; }) : S.Y;
        S.pts = [s.points(ya.map(p => F.W(...F.clampP(p), .04).toArray()), ya.map(() => rgb(HEX.ball)), .07),
          s.points(S.X.map(p => F.W(...F.clampP(p), .04).toArray()), S.X.map(() => rgb(HEX.opt2)), .07)];
        ctx.charts.al.clear().line(S.sp, { color: HEX.opt2, width: 1.6, label: 'perturbed GD' }).line(S.sa, { color: HEX.ball, width: 2, label: 'Algorithm 1' }).vline(T1, { color: HEX.muted, dash: [3, 3] }).draw();
        const dec = P => P.map(p => -fL(...p));
        ctx.charts.hist.clear().hist(dec(S.X), { color: HEX.opt2, bins: 34, range: [0, 1.02], alpha: .75 }).hist(dec(S.Y.map(p => S.t < T1 ? [0, 0] : p)), { color: HEX.ball, bins: 34, range: [0, 1.02], alpha: .6 }).vline(.9, { color: HEX.muted, dash: [3, 3], text: '0.9' }).draw();
        const stall = P => P.filter(p => -fL(...p) <= .9).length / P.length;
        ctx.stat(`t = ${S.t} · ${S.t <= T1 ? `Algorithm 1 directions within 5° of ±e₁: ${(100 * S.sa[S.sa.length - 1][1]).toFixed(1)}%` : 'Algorithm 1 kicked at t = 30, now plain GD'} · paths with decrease ≤ 0.9: Alg 1 ${S.t < T1 ? '—' : (100 * stall(S.Y)).toFixed(1) + '%'}, PGD ${(100 * stall(S.X)).toFixed(1)}%`);
      },
      frame(ctx, dt) {
        const S = ctx.st; if (!S || S.t >= T2) return;
        S.acc += dt * 15; let moved = false;
        while (S.acc >= 1 && S.t < T2) { S.acc -= 1; this.step(ctx); moved = true; }
        if (moved) this.draw(ctx);
      },
    },
    {
      id: 'rates', name: 'Dimension dependence', mode: '2d', title: 'Same 1/ε order as perturbed GD, fewer powers of log n',
      caption: 'Iteration orders (Table 1 of the paper, constants and log 1/ε factors dropped) to reach an ε-approximate second-order stationary point in dimension n. The gain of Algorithm 1 is polynomial in log n at the same power of 1/ε; in the stochastic setting its ε⁻⁴ is worse in ε than nested-loop methods (Neon2: log²n/ε³·⁵).',
      note: 'Orders only. Deterministic: PGD [Jin et al.] log⁴n/ε², PAGD log⁶n/ε^1.75, this work log n/ε^1.75. Stochastic: PSGD log¹⁵n/ε⁴, this work log²n/ε⁴, Neon2 log²n/ε³·⁵.',
      hint: '', legend: [[HEX.opt3, 'PGD / PSGD'], [HEX.opt2, 'PAGD / Neon2'], [HEX.ball, 'this work']],
      controls: [{ id: 'set', type: 'seg', label: 'setting', value: 'det', options: [['det', 'deterministic'], ['sto', 'stochastic']] },
        { id: 'e', label: 'accuracy ε', min: -4, max: -1, step: .5, value: -2, fmt: v => `10^${(+v).toFixed(1)}` }],
      plot: { xlabel: 'dimension n', ylabel: 'iterations (order)', logx: true, logy: true, xlim: [10, 1e10], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const e = 10 ** ctx.vals.e, det = ctx.vals.set === 'det', ln = Math.log;
        const rows = det ? [[4, 2, HEX.opt3], [6, 1.75, HEX.opt2], [1, 1.75, HEX.ball]] : [[15, 4, HEX.opt3], [2, 3.5, HEX.opt2], [2, 4, HEX.ball]];
        const P = ctx.plot.clear();
        rows.forEach(([a, b, c]) => P.fn(n => ln(n) ** a / e ** b, { color: c, width: c === HEX.ball ? 2.6 : 2 }));
        P.draw();
        const n = 1e6, v = rows.map(([a, b]) => (ln(n) ** a / e ** b).toExponential(1));
        ctx.stat(`n = 10⁶, ε = 10^${ctx.vals.e}: ${det ? 'PGD' : 'PSGD'} ${v[0]} · ${det ? 'PAGD' : 'Neon2'} ${v[1]} · this work ${v[2]}`);
      },
    },
  ],
};
