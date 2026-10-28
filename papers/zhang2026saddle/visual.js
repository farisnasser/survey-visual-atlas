// Zhang, Saxe, Latham 2026 — saddle-to-saddle dynamics. Checked in scripts/check-math/zhang2026saddle.py
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

// exact loss of a 2-layer linear net on the balanced, aligned invariant manifold W1 = W2 = diag(a1, a2), target diag(s1, s2)
const manifold = (s1, s2) => def({
  L: (a, b) => .5 * (s1 - a * a) ** 2 + .5 * (s2 - b * b) ** 2,
  g: (a, b) => [-2 * a * (s1 - a * a), -2 * b * (s2 - b * b)],
  hess: (a, b) => [-2 * s1 + 6 * a * a, 0, -2 * s2 + 6 * b * b],
  h: L => Math.log1p(L) * 1.05, dom: [-1.8, 1.8, -1.8, 1.8],
});

/** Linear net f = sum_i v_i u_i^T z with whitened 3-d inputs, target map diag(s); returns per-mode strength over time. */
function linearRun(s, H, sig, steps, lr, seed) {
  const r = rng(seed), U = Array.from({ length: H }, () => [0, 0, 0].map(() => sig * r.normal())), V = Array.from({ length: H }, () => [0, 0, 0].map(() => sig * r.normal()));
  const loss = [], modes = [[], [], []];
  for (let t = 0; t <= steps; t++) {
    const W = [0, 1, 2].map(i => [0, 1, 2].map(j => U.reduce((acc, u, h) => acc + V[h][i] * u[j], 0)));   // W = sum v u^T
    const E = W.map((row, i) => row.map((w, j) => (i === j ? s[i] : 0) - w));
    if (t % 10 === 0) { loss.push([t * lr, .5 * E.flat().reduce((a, v) => a + v * v, 0)]); W.forEach((row, i) => modes[i].push([t * lr, row[i]])); }
    for (let h = 0; h < H; h++) {
      const u = U[h], v = V[h], gv = E.map(row => row.reduce((a, e, j) => a + e * u[j], 0)), gu = [0, 1, 2].map(j => E.reduce((a, row, i) => a + row[j] * v[i], 0));
      for (let k = 0; k < 3; k++) { v[k] += lr * gv[k]; u[k] += lr * gu[k]; }
    }
  }
  return { loss, modes };
}
/** Quadratic-in-weights net f(x) = sum_i v_i (u_i . x)^2 on 2-d Gaussian data, teacher x1^2 - 0.6 x2^2. Per-unit norms. */
function quadRun(H, sig, steps, lr, seed) {
  const r = rng(seed), X = Array.from({ length: 200 }, () => [r.normal(), r.normal()]), y = X.map(x => x[0] ** 2 - .6 * x[1] ** 2);
  const u = Array.from({ length: H }, () => [sig * r.normal(), sig * r.normal()]), v = Array.from({ length: H }, () => sig * r.normal());
  const loss = [], norms = Array.from({ length: H }, () => []), snaps = [];
  for (let t = 0; t <= steps; t++) {
    const Z = X.map(x => u.map(w => w[0] * x[0] + w[1] * x[1])), e = Z.map((z, n) => z.reduce((a, zi, i) => a + v[i] * zi * zi, 0) - y[n]);
    if (t % 10 === 0) { loss.push([t * lr, .5 * e.reduce((a, x) => a + x * x, 0) / X.length]); u.forEach((w, i) => norms[i].push([t * lr, Math.sqrt(v[i] ** 2 + w[0] ** 2 + w[1] ** 2)])); }
    if (t % 40 === 0) snaps.push(u.map((w, i) => [w[0] * Math.sqrt(Math.abs(v[i])), w[1] * Math.sqrt(Math.abs(v[i]))]));
    for (let i = 0; i < H; i++) {
      let gv = 0, g0 = 0, g1 = 0;
      for (let n = 0; n < X.length; n++) { const z = Z[n][i]; gv += z * z * e[n]; g0 += 2 * e[n] * v[i] * z * X[n][0]; g1 += 2 * e[n] * v[i] * z * X[n][1]; }
      v[i] -= lr * gv / X.length; u[i][0] -= lr * g0 / X.length; u[i][1] -= lr * g1 / X.length;
    }
  }
  return { loss, norms, snaps };
}

export default {
  tabs: [
    {
      id: 'manifold', name: 'Saddle to saddle', title: 'From one saddle to the next, then the minimum',
      caption: 'Exact loss of a two-layer linear network restricted to an invariant manifold (one coordinate per data mode). From tiny weights the ball first learns the strong mode, pauses at a width-1 saddle, then learns the weak mode.',
      note: 'Balanced, aligned weights W₁ = W₂ = diag(a₁, a₂) stay so under gradient flow (Thm. 3), so this 2D surface is the true loss along the path.',
      legend: [[HEX.ball, 'gradient descent'], ['#ffffff', 'saddles (width 0 and width 1)'], [HEX.good, 'global minima'], [HEX.neg, 'λ < 0'], [HEX.pos, 'λ > 0']],
      action: 'Restart from tiny weights', cam: [3.2, 3.6, 4.4], tgt: [0, .4, 0],
      controls: [{ id: 's2', label: 'weak mode s₂', min: .15, max: 1.6, step: .05, value: .6 }],
      charts: [{ id: 'l', title: 'loss along the run', xlabel: 'step', ylabel: 'loss', h: 230, legend: false }],
      build(ctx) {
        const s1 = 2, s2 = ctx.vals.s2, S = ctx.S = manifold(s1, s2), s = ctx.stage; s.surface(S);
        s.marker(S, 0, 0, C.white); s.marker(S, Math.sqrt(s1), 0, C.white);
        for (const sx of [1, -1]) for (const sy of [1, -1]) s.marker(S, sx * Math.sqrt(s1), sy * Math.sqrt(s2), C.good, .08);
        s.eigenLines(S, Math.sqrt(s1), 0, { len: .5 });
        s.label('width-1 saddle', S.W(Math.sqrt(s1), 0, .55), '', HEX.white); s.label('width-0 saddle', S.W(0, 0, .45), '', HEX.white);
        this.drop(ctx);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) { const s = ctx.stage; s.clearBalls(); ctx.lh = []; ctx.b = s.ball(ctx.S, { start: [.004, .003], step: (p, b, d) => { const q = opt.gd(.02)(p, b, d); ctx.lh.push([b.steps, d.L(...q)]); return q; }, speed: 3, maxSteps: 6000, stop: b => b.loss() < 1e-9 }); },
      frame(ctx) {
        ctx.charts.l.clear().line(ctx.lh, { color: HEX.ball, width: 2 }).hline(.5 * ctx.vals.s2 ** 2, { color: HEX.muted, dash: [4, 4], text: 'width-1 saddle: s₂²/2' }).draw();
        const b = ctx.b; if (b) ctx.stat(`step ${b.steps} · a₁² = ${(b.p[0] ** 2).toFixed(3)} (→ s₁ = 2) · a₂² = ${(b.p[1] ** 2).toFixed(3)} (→ s₂ = ${ctx.vals.s2})`);
      },
    },
    {
      id: 'growth', name: 'Two mechanisms', mode: '2d', title: 'Data picks the order (linear) or initialization does (quadratic)',
      caption: 'Left: a linear network learns data modes one at a time, ordered by singular value. Right: a quadratic-in-weights network grows one unit at a time, ordered by which unit started largest. Both show stage-like loss curves.',
      note: 'Linear: whitened 3-d inputs, target diag(s), s ∝ n^(−θ). Quadratic: f = Σᵢ vᵢ(uᵢ·x)², teacher x₁² − 0.6x₂². Both from small Gaussian init, trained live.',
      hint: '', legend: [],
      controls: [{ id: 'th', label: 'power-law θ', min: 0, max: 2, step: .1, value: 1, live: false }, { id: 'ls', label: 'init scale', min: -4, max: -1, step: .25, value: -3, live: false, fmt: v => `1e${(+v).toFixed(1)}` }],
      actions: [{ label: 'Re-run', fn: ctx => ctx.shell.tab.compute(ctx) }],
      plot: { xlabel: 'time (step × lr)', ylabel: 'linear net: mode strength', xlim: [0, 40], ylim: [0, 1.05] },
      charts: [
        { id: 'q', title: 'quadratic net: per-unit size', xlabel: 'time', ylabel: '‖(vᵢ, uᵢ)‖', h: 230, legend: false },
        { id: 'loss', title: 'loss (both nets)', xlabel: 'time', ylabel: 'loss', logy: true, h: 190 },
      ],
      build(ctx) { this.compute(ctx); }, onControl(ctx) { this.compute(ctx); },
      compute(ctx) {
        const th = ctx.vals.th, sig = 10 ** ctx.vals.ls, raw = [1, 2, 3].map(n => n ** -th), Z = raw.reduce((a, b) => a + b, 0), s = raw.map(v => v / Z * 2);
        const L = linearRun(s, 6, sig, 4000, .01, 3), Q = quadRun(6, sig * 10, 6000, .05, 4), cols = [HEX.ball, HEX.flat, HEX.pos];
        const P = ctx.plot.clear().set({ ylim: [0, Math.max(...s) * 1.08] });
        L.modes.forEach((m, i) => P.line(m, { color: cols[i], width: 2, label: `mode s = ${s[i].toFixed(2)}` }).hline(s[i], { color: cols[i], dash: [3, 5], alpha: .5, noRange: true }));
        P.draw();
        const qc = [HEX.ball, HEX.flat, HEX.pos, HEX.opt2, HEX.good, HEX.neg];
        const q = ctx.charts.q.clear(); Q.norms.forEach((n, i) => q.line(n, { color: qc[i], width: 1.8 })); q.draw();
        ctx.charts.loss.clear().line(L.loss.map(p => [p[0], Math.max(p[1], 1e-12)]), { color: HEX.pos, width: 2, label: 'linear' }).line(Q.loss.map(p => [p[0], Math.max(p[1], 1e-12)]), { color: HEX.opt2, width: 2, label: 'quadratic' }).draw();
        const fin = Q.norms.map(n => n[n.length - 1][1]).sort((a, b) => b - a);
        ctx.stat(`θ = ${th}: singular values ${s.map(v => v.toFixed(2)).join(', ')} · quadratic net ends with ${fin.filter(v => v > .3).length} large unit(s), rest ≤ ${fin[2].toFixed(2)}`);
      },
    },
  ],
};
