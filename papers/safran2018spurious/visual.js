// Safran & Shamir 2018 — spurious local minima of two-layer ReLU nets (population loss, closed form).
// Experiments + the spurious point come from scripts/check-math/safran2018spurious.py (data.json); the slice is exact.
import { def, C, HEX, statLabels } from '../../src/lib/atlas3d.js';

// E[[w.x]+[v.x]+] for x ~ N(0, I): |w||v|/(2pi) (sin t + (pi - t) cos t)
function popLoss(k) {
  const V = Array.from({ length: k }, (_, i) => Array.from({ length: k }, (_, j) => +(i === j)));
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0), nrm = a => Math.sqrt(dot(a, a));
  const pair = (w, u) => { const nw = nrm(w), nu = nrm(u); if (nw < 1e-300 || nu < 1e-300) return [0, 0, 0]; const c = Math.max(-1, Math.min(1, dot(w, u) / (nw * nu))), t = Math.acos(c); return [nw * nu / (2 * Math.PI) * (Math.sin(t) + (Math.PI - t) * c), nu * Math.sin(t) / (2 * Math.PI * nw), (Math.PI - t) / (2 * Math.PI)]; };
  const cVV = k * .5 + k * (k - 1) / (2 * Math.PI);
  const F = W => { let s = 0; for (const a of W) { for (const b of W) s += pair(a, b)[0]; for (const v of V) s -= 2 * pair(a, v)[0]; } return .5 * (s + cVV); };
  const grad = W => W.map(w => { const g = new Array(k).fill(0); for (const [U, sg] of [[W, 1], [V, -1]]) for (const u of U) { const [, a, b] = pair(w, u); for (let j = 0; j < k; j++) g[j] += sg * (a * w[j] + b * u[j]); } return g; });
  return { F, grad };
}

export default {
  tabs: [
    {
      id: 'slice', name: 'Spurious minimum', title: 'A real bad local minimum',
      caption: 'Exact population loss of a two-layer ReLU net with k = n = 8 on the plane through a spurious minimum found by GD (white, F ≈ 0.022) and the nearest global minimum (green, F = 0). A barrier separates them.',
      note: 'Height ∝ √F; the two slice axes are drawn at different scales. Balls run gradient descent restricted to this plane, so they see exactly this surface.',
      legend: [[HEX.ball, 'GD from near the spurious min'], [HEX.opt2, 'GD from near the global min'], ['#ffffff', 'spurious minimum'], [HEX.good, 'global minimum']],
      action: 'Release balls', cam: [0, 3.5, 4.6], tgt: [0, .35, 0],
      build(ctx) {
        const S = ctx.data.slice, k = S.k, P = popLoss(k);
        const at = (a, b) => S.Ws.map((w, i) => w.map((v, j) => v + a * S.d1[i][j] + b * S.d2[i][j]));
        const flat = W => W.flat(), dot = (x, y) => x.reduce((s, v, i) => s + v * y[i], 0), D1 = flat(S.d1), D2 = flat(S.d2);
        const d = ctx.d = def({ L: (a, b) => P.F(at(a, b)), g: (a, b) => { const G = flat(P.grad(at(a, b))); return [dot(G, D1), dot(G, D2)]; }, h: F => Math.sqrt(Math.max(F, 0)), dom: [S.a[0], S.a[1], -.45, .45], N: 72, square: true });
        const s = ctx.stage; s.surface(d);
        s.marker(d, 0, 0, C.white, .08); s.marker(d, 1, 0, C.good, .08);
        s.label(`spurious · F = ${d.L(0, 0).toFixed(4)}`, d.W(0, 0, .3), '', HEX.white); s.label('global · F = 0', d.W(1, 0, .3), '', HEX.good);
        ctx.check = Math.abs(d.L(0, 0) - S.F_spurious);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, d = ctx.d; s.clearBalls();
        const gd = lr => (p, b, dd) => { const [ga, gb] = dd.g(p[0], p[1]); return [p[0] - lr * ga, p[1] - lr * gb]; };
        s.ball(d, { start: [.22, .3], step: gd(.05), speed: 1, maxSteps: 3000, stop: b => Math.hypot(...d.g(...b.p)) < 1e-7 });
        s.ball(d, { start: [.62, -.35], step: gd(.05), speed: 1, maxSteps: 3000, color: C.opt2, stop: b => Math.hypot(...d.g(...b.p)) < 1e-7 });
      },
      frame(ctx) { const [a, b] = ctx.stage.balls; if (a) ctx.stat(`coral F = ${a.loss().toFixed(4)} · violet F = ${b.loss().toExponential(1)} · λ_min(∇²F) at the spurious point = ${ctx.data.slice.lam_min.toFixed(3)} > 0`); },
    },
    {
      id: 'odds', name: 'How often', mode: '2d', title: 'GD often ends at a bad minimum',
      caption: 'Share of gradient-descent runs ending at a spurious minimum when student width n equals teacher width k. The paper’s Table 1 (bars) and our replication with the same protocol (dots).',
      note: 'Protocol: Xavier init N(0, I/k), step 0.1, run to a stationary point, ignore F < 10⁻³. Ours: 600/400/300/200 runs for k = 6/8/10/12.',
      hint: '', legend: [[HEX.pos, 'paper, 1000 runs each'], [HEX.ball, 'this replication']],
      plot: { xlabel: 'k = n (neurons)', ylabel: '% runs at a spurious minimum', ylim: [0, 100], legend: false },
      build(ctx) {
        const T = ctx.data.paper_table1, ks = Object.keys(T).map(Number), E = ctx.data.experiments;
        const P = ctx.plot.clear().set({ xlim: [-.6, ks.length - .4], xfmt: v => String(ks[Math.round(v)] ?? '') });
        P.bars(ks.map(k => ({ y: T[k], color: 'rgba(124,196,255,.55)' })), { width: .7 });
        const ours = Object.values(E).filter(e => e.k === e.n).map(e => [ks.indexOf(e.k), 100 * e.spurious_frac]);
        P.scatter(ours, { color: HEX.ball, r: 5.5 });
        P.draw();
        const ex = Object.values(E).filter(e => e.n > e.k).map(e => `k=${e.k}, n=${e.n}: ${(100 * e.spurious_frac).toFixed(1)}%`).join(' · ');
        ctx.stat(`ours: ${ours.map(([i, v]) => `k=${ks[i]}: ${v.toFixed(1)}%`).join(' · ')} · extra neurons — ${ex}`);
      },
    },
    {
      id: 'cdf', name: 'Final losses', mode: '2d', title: 'Where gradient descent stops',
      caption: 'Cumulative share of runs whose final loss is ≤ the x-value (the paper’s Fig. 1 plots the same quantity). Curves that jump late put mass on bad minima; one or two extra neurons remove that mass in these runs.',
      hint: '', legend: [],
      plot: { xlabel: 'final population loss F', ylabel: 'fraction of runs with F ≤ x', xlim: [0, .045], ylim: [0, 1.02] },
      build(ctx) {
        const E = ctx.data.experiments, cols = [HEX.opt2, HEX.flat, HEX.ball, HEX.neg, HEX.pos, HEX.good], P = ctx.plot.clear();
        Object.values(E).forEach((e, i) => {
          const v = e.final, pts = [[0, 0]]; v.forEach((x, j) => pts.push([Math.min(x, .045), (j + 1) / v.length]));
          P.line(pts, { color: cols[i % cols.length], width: 2, step: true, label: `k=${e.k}, n=${e.n}` });
        });
        P.draw();
      },
    },
  ],
};
