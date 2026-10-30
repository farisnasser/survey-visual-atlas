// Azizian, Iutzeler, Malick, Mertikopoulos 2025 — global convergence time of SGD via large deviations, on the paper's
// modified three-hump camel (Fig. 1). Critical points, transition costs and the offline hitting-time sweep:
// scripts/check-math/azizian2025global.py (data.json).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const f = (x, y) => 2 * x ** 6 / 13 + x ** 5 / 8 - 91 * x ** 4 / 64 - x ** 3 / 2 + 42 * x * x / 16 + 5 * y * y / 4 + x * y;
const g = (x, y) => [12 * x ** 5 / 13 + 5 * x ** 4 / 8 - 91 * x ** 3 / 16 - 1.5 * x * x + 42 * x / 8 + y, 2.5 * y + x];
const CAMEL = def({ L: f, g, h: v => Math.log(v + 7.6), dom: [-3.1, 2.7, -1.7, 1.7], square: true, N: 120 });

function drawGraph(P, nodes, edges, hi) {
  P.custom((c, p) => {
    c.font = '12px JetBrains Mono'; c.textAlign = 'center';
    for (const e of edges) {
      const a = nodes[e[0]], b = nodes[e[1]], on = hi.includes(e[0] + e[1]), prn = hi.includes('!' + e[0] + e[1]);
      const ax = p.sx(a.x), ay = p.sy(a.y), bx = p.sx(b.x), by = p.sy(b.y), off = e[3] || 0, mx = (ax + bx) / 2, my = (ay + by) / 2 - off;
      c.strokeStyle = on ? HEX.good : prn ? HEX.flat : 'rgba(142,153,189,.5)'; c.lineWidth = on || prn ? 3 : 1.4; c.setLineDash(prn ? [6, 4] : []);
      c.beginPath(); c.moveTo(ax, ay); c.quadraticCurveTo(mx, my, bx, by); c.stroke(); c.setLineDash([]);
      const t = .62, qx = (1 - t) ** 2 * ax + 2 * (1 - t) * t * mx + t * t * bx, qy = (1 - t) ** 2 * ay + 2 * (1 - t) * t * my + t * t * by;
      c.fillStyle = c.strokeStyle; c.beginPath(); c.arc(qx, qy, 4, 0, 7); c.fill();
      c.fillStyle = '#e8ecff'; c.fillText(e[2], (ax + 2 * mx + bx) / 4, (ay + 2 * my + by) / 4 - 6);
    }
    for (const [k, n] of Object.entries(nodes)) {
      c.fillStyle = n.goal ? HEX.good : '#2a3157'; c.strokeStyle = n.goal ? HEX.good : HEX.white; c.lineWidth = 1.5;
      c.beginPath(); c.arc(p.sx(n.x), p.sy(n.y), 18, 0, 7); c.fill(); c.stroke();
      c.fillStyle = n.goal ? '#0b0e1a' : '#fff'; c.fillText(n.label || k, p.sx(n.x), p.sy(n.y) + 4);
    }
  });
}

export default {
  tabs: [
    {
      id: 'camel', name: 'Hopping basins', title: 'How long until SGD first reaches the global minimum?',
      caption: 'The paper’s test function (Fig. 1): three minima and two saddles. SGD with Gaussian noise starts in the middle basin p₃ and runs until it is within δ = 0.1 of the global minimum p₁ (green). It usually falls first into the right basin p₅, which is cheaper to enter, and must climb out of it again: this rare detour sets the exponent of the mean hitting time.',
      note: 'f = 2x₁⁶/13 + x₁⁵/8 − 91x₁⁴/64 − x₁³/2 + 42x₁²/16 + 5x₂²/4 + x₁x₂; minimizer p₁ = (−2.533, 1.013) from the formula (the paper prints (−2.573, 1.029)). Noise σ = 11; height shown as log(f + 7.6).',
      legend: [[HEX.ball, 'SGD'], [HEX.good, 'global minimum p₁'], [HEX.white, 'local minima p₃, p₅'], [HEX.neg, 'saddles p₂, p₄']],
      action: 'Run again', cam: [.8, 3.8, 4.6], tgt: [0, .2, 0],
      controls: [{ id: 'eta', label: 'step η', min: .011, max: .02, step: .0005, value: .02, digits: 4 }],
      build(ctx) {
        const s = ctx.stage, D = ctx.data; s.surface(CAMEL);
        D.crit.forEach(([x, y, v], i) => { const k = D.kinds[i]; s.marker(CAMEL, x, y, i === 0 ? C.good : k === 'min' ? C.white : C.neg, .09); s.label(`p${i + 1}`, CAMEL.W(x, y, .35), '', i === 0 ? HEX.good : k === 'min' ? HEX.white : HEX.neg); });
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, D = ctx.data, R = rng(Date.now() % 1e6), p1 = D.crit[0], sig = D.sigma; s.clearBalls(); ctx.visits = { p5: 0 }; ctx.inP5 = false;
        ctx.b = s.ball(CAMEL, { start: [0, 0], speed: 40, maxSteps: 3e6, maxTrail: 3000, stop: b => Math.hypot(b.p[0] - p1[0], b.p[1] - p1[1]) < D.delta, step: p => {
          const [gx, gy] = g(...p), e = ctx.vals.eta, q = [p[0] - e * (gx + sig * R.normal()), p[1] - e * (gy + sig * R.normal())];
          const inP5 = q[0] > .894; if (inP5 && !ctx.inP5) ctx.visits.p5++; ctx.inP5 = inP5; return q;
        } });
      },
      onControl(ctx) { this.drop(ctx); },
      frame(ctx) {
        const b = ctx.b, D = ctx.data; if (!b) return;
        const E = D.E / D.sigma ** 2;
        ctx.stat(`${b.done ? `reached p₁ after τ = ${b.steps.toLocaleString()} steps` : `step ${b.steps.toLocaleString()}`} · crossings to the p₅ side of the saddle p₄ (x₁ > 0.894): ${ctx.visits.p5} · theory: log E[τ] ≈ E/η = ${(E / ctx.vals.eta).toFixed(2)} (up to subexponential factors)`);
      },
    },
    {
      id: 'graph', name: 'Transition graph', mode: '2d', title: 'Energy = cheapest forest − cheapest pruning',
      caption: 'Nodes are minima, edges the least action to move between them. For Gaussian noise of variance σ², climbing a barrier Δf costs 2Δf/σ², descending is free. Green: the cheapest forest that sends every node to the target. Amber dashed: the cheapest pruning edge, a move that leaves the start unable to reach the target. Their difference is the energy E(Q | x) in Theorem 1’s exp(E/η).',
      note: 'Camel costs from the exact critical values (offline). The notes’ three-state example is solved exactly: its mean time T satisfies η·log T → 4.',
      hint: '', legend: [[HEX.good, 'cheapest forest'], [HEX.flat, 'cheapest pruning'], [HEX.muted, 'other transitions']],
      controls: [{ id: 'g', type: 'seg', label: 'graph', value: 'camel', options: [['camel', 'camel (units 1/σ²)'], ['notes', 'notes’ 3-state chain']] }],
      plot: { xlim: [-1.4, 1.4], ylim: [-1, 1.2], legend: false, grid: false, xfmt: () => '', yfmt: () => '' },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const P = ctx.plot.clear();
        if (ctx.vals.g === 'camel') {
          const B = ctx.data.B, n = { p1: { x: -1, y: 0, goal: true, label: 'p₁' }, p3: { x: 0, y: 0, label: 'p₃' }, p5: { x: 1, y: 0, label: 'p₅' } };
          const ed = [['p3', 'p1', B['p3-p1'].toFixed(2), 30], ['p3', 'p5', B['p3-p5'].toFixed(2), 30], ['p5', 'p3', B['p5-p3'].toFixed(2), -30], ['p5', 'p1', B['p5-p1'].toFixed(2), 110]];
          drawGraph(P, n, ed, ['p3p1', 'p5p3', '!p3p5']); P.draw();
          ctx.stat(`forest {p₃→p₁, p₅→p₃} = ${(B['p3-p1'] + B['p5-p3']).toFixed(3)} · pruning p₃→p₅ = ${B['p3-p5'].toFixed(3)} · E(Q | p₃) = ${ctx.data.E.toFixed(3)}/σ² = ${(ctx.data.E / ctx.data.sigma ** 2).toFixed(4)} for σ = ${ctx.data.sigma}`);
        } else {
          const n = { G: { x: -1, y: 0, goal: true }, A: { x: 0, y: 0 }, B: { x: 1, y: 0 } };
          drawGraph(P, n, [['A', 'G', '2', 30], ['A', 'B', '3', 30], ['B', 'A', '5', -30], ['B', 'G', '8', 110]], ['AG', 'BA', '!AB']); P.draw();
          const T = e => { const [u, v, w, z] = [2, 3, 5, 8].map(c => Math.exp(-c / e)); return (w + z + v) / (u * w + u * z + v * z); };
          ctx.stat(`forests {AG, BA} = 7, {AG, BG} = 10, {AB, BG} = 11 · cheapest pruning AB = 3 · E = 4 · exact T: η = 0.5 → ${T(.5).toFixed(2)} (η log T = ${(.5 * Math.log(T(.5))).toFixed(4)}), η = 0.25 → ${T(.25).toFixed(2)} (${(.25 * Math.log(T(.25))).toFixed(5)})`);
        }
      },
    },
    {
      id: 'scaling', name: 'Exponential scaling', mode: '2d', title: 'Mean hitting time grows like exp(E/η)',
      caption: 'Offline: 400 SGD runs per step size from p₃ until within 0.1 of p₁. log(η·E[τ]) is linear in 1/η with slope close to the predicted energy E = 9.775/σ². Halving η multiplies the time by roughly exp(E/η) again: global convergence is exponentially slow when there are spurious minima.',
      note: 'σ = 11. Plotting η·τ (SGD “time”) removes the 1/η factor of an iteration count; such subexponential factors are absorbed by the ±a in Theorem 1. The fitted slope of plain log E[τ] is also within 25% of E.',
      hint: '', legend: [[HEX.ball, 'measured, 400 runs each'], [HEX.muted, 'slope E (theory)'], [HEX.pos, 'least-squares fit']],
      plot: { xlabel: '1/η', ylabel: 'log(η · E[τ])', xlim: [45, 105], ylim: [2.5, 8] },
      build(ctx) {
        const D = ctx.data, E = D.E / D.sigma ** 2, pts = D.rows.map(r => [1 / r[0], r[1] + Math.log(r[0])]);
        const mx = pts.reduce((s, p) => s + p[0], 0) / pts.length, my = pts.reduce((s, p) => s + p[1], 0) / pts.length;
        ctx.plot.clear().fn(x => my + E * (x - mx), { color: HEX.muted, dash: [5, 4], width: 2 }).fn(x => D.icpt + D.slope * x, { color: HEX.pos, width: 1.4 }).scatter(pts, { color: HEX.ball, r: 5 }).draw();
        ctx.stat(`predicted slope E = ${E.toFixed(4)} · fitted ${D.slope.toFixed(4)} (log E[τ] in iterations: ${D.slope_iter.toFixed(4)}) · E[τ] ≈ ${Math.exp(D.rows[0][1]).toPrecision(3)} steps at η = ${D.rows[0][0]}, ≈ ${Math.exp(D.rows[D.rows.length - 1][1]).toPrecision(3)} at η = ${D.rows[D.rows.length - 1][0]}`);
      },
    },
  ],
};
