// Chizat & Bach 2018 — particle gradient flow of a one-hidden-layer ReLU network (setup of Fig. 2).
// Trajectories are replayed from scripts/check-math/chizat2018global.py (data.json): u_j = |w_j|θ_j and sign(w_j),
// so the network is f(x) = (1/m) Σ sign_j · relu(u_j · x) exactly (ReLU is positively homogeneous).
import { def, C, HEX } from '../../src/lib/atlas3d.js';

const relu = z => Math.max(z, 0);
const teacher = D => (x, y) => D.TH.reduce((s, t, k) => s + D.WS[k] * relu(t[0] * x + t[1] * y), 0) / 4;
const student = P => (x, y) => P.reduce((s, u) => s + u[2] * relu(u[0] * x + u[1] * y), 0) / P.length;
const DOM = [-2, 2, -2, 2], H = v => v * .8;
const MS = [[5, '5'], [10, '10'], [20, '20'], [100, '100']];
const COLS = { 5: HEX.opt3, 10: HEX.opt2, 20: '#7cc4ff', 100: HEX.ball };

export default {
  tabs: [
    {
      id: 'fit', name: 'Output surface', title: 'The network’s output surface settles onto the target once there are enough neurons',
      caption: 'Green-ramp surface: the target y(x) = ¼ Σ w*_k relu(θ*_k·x), a ReLU network with m₀ = 4 neurons on 2-D Gaussian inputs. Violet sheet: the student f(x) = (1/m) Σ w_j relu(θ_j·x) during gradient descent on all (w_j, θ_j), starting from tiny weights on a small sphere. With 20 or 100 neurons the sheet reaches the target; with 5 or 10 it gets stuck slightly off.',
      note: 'Fixed sample of 600 inputs, square loss, step 0.4·m per particle (time-rescaled particle gradient flow), 5 000 steps replayed from the check script. The paper’s own instance fails at 5 and succeeds at 10; on this teacher the threshold sits between 10 and 20.',
      legend: [['#b28cff', 'student f(x)'], [HEX.good, 'target y(x)']],
      cam: [3.5, 2.9, 3.9], tgt: [0, .2, 0],
      controls: [{ id: 'm', type: 'seg', label: 'neurons m', value: 100, options: MS }],
      actions: [{ label: 'Replay', fn: ctx => { ctx.k = 0; ctx.acc = 0; } }],
      charts: [{ id: 'L', title: 'training loss (dashed: other m)', xlabel: 'gradient step', ylabel: 'loss', logy: true, ylim: [1e-7, .3], h: 230 }],
      build(ctx) {
        const D = ctx.data, s = ctx.stage; s.clear();
        ctx.T = def({ L: teacher(D), h: H, dom: DOM, fit: 0, N: 90 });
        s.surface(ctx.T);
        ctx.k = 0; ctx.acc = 0; ctx.sheet = null; this.sheet(ctx);
      },
      sheet(ctx) {
        const D = ctx.data, s = ctx.stage, tr = D[`traj${ctx.vals.m}`], P = tr[Math.min(ctx.k, tr.length - 1)];
        if (ctx.sheet) s.remove(ctx.sheet.mesh);
        ctx.sheet = def({ L: student(P), h: H, dom: DOM, fit: 0, N: 44 });
        s.surface(ctx.sheet, { grid: false, opacity: .55, colorFn: () => [.7, .55, 1] });
        const steps = ctx.k * 50, Lk = D[`loss${ctx.vals.m}`];
        const cur = Lk[Math.min(ctx.k * 2, Lk.length - 1)];
        const ch = ctx.charts.L.clear().set({ xlim: [0, 5000] });
        for (const [m] of MS) if (m !== ctx.vals.m) ch.line(D[`loss${m}`].map((v, i) => [i * 25, v]), { color: COLS[m], dash: [3, 3], width: 1, alpha: .6 });
        ch.line(Lk.slice(0, ctx.k * 2 + 1).map((v, i) => [i * 25, v]), { color: COLS[ctx.vals.m], width: 2, label: `m = ${ctx.vals.m}` }).draw();
        ctx.stat(`m = ${ctx.vals.m} neurons · step ${steps} · loss ${cur.toExponential(2)} (loss of f = 0: ${D.L0.toFixed(3)}) · final after 5 000 steps: ${D[`final${ctx.vals.m}`].toExponential(2)}`);
      },
      frame(ctx, dt) {
        const tr = ctx.data[`traj${ctx.vals.m}`]; if (!tr || ctx.k >= tr.length - 1) return;
        ctx.acc += dt * 12; if (ctx.acc < 1) return; ctx.acc = 0; ctx.k++; this.sheet(ctx);
      },
      onControl(ctx) { this.build(ctx); },
    },
    {
      id: 'particles', name: 'Particle paths', mode: '2d', title: 'Neurons as particles: trajectories of |w_j|·θ_j (the paper’s Fig. 2 view)',
      caption: 'Each curve is one neuron, drawn as |w_j|θ_j ∈ ℝ² (coral: w_j > 0, violet: w_j < 0), from a small circle at the origin to its final position (dot). Dashed rays: the directions of the 4 target neurons. With enough particles the mass gathers on every target ray; with few, some ray is missed or a particle parks on a wrong direction.',
      note: 'At the end, the total mass (1/m)Σ|w_j|θ_j on each ray must equal the target neuron’s |w*_k|θ*_k/4 for a perfect fit; particles may split a target between them.',
      hint: '', legend: [[HEX.ball, 'w_j > 0'], [HEX.opt2, 'w_j < 0'], [HEX.good, 'target directions']],
      controls: [{ id: 'm', type: 'seg', label: 'neurons m', value: 10, options: MS }],
      plot: { xlabel: '(|w|θ)₁', ylabel: '(|w|θ)₂', xlim: [-3.2, 3.2], ylim: [-2.4, 2.4], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const D = ctx.data, tr = D[`traj${ctx.vals.m}`], m = tr[0].length, P = ctx.plot.clear();
        D.TH.forEach((t, k) => { const n = Math.hypot(...t); P.line([[0, 0], [4 * t[0] / n, 4 * t[1] / n]], { color: HEX.good, dash: [5, 4], width: 1.2, alpha: D.WS[k] > 0 ? 1 : .55 }); });
        for (let j = 0; j < m; j++) {
          const last = tr[tr.length - 1][j], col = last[2] > 0 ? HEX.ball : HEX.opt2;
          P.line(tr.map(F => [F[j][0], F[j][1]]), { color: col, width: m > 20 ? .8 : 1.4, alpha: .8 });
          P.scatter([[last[0], last[1]]], { color: col, r: m > 20 ? 2 : 3.5 });
        }
        P.draw();
        ctx.stat(`m = ${m} · final loss ${D[`final${ctx.vals.m}`].toExponential(2)} · target weights w* = (${D.WS.map(v => v.toFixed(1)).join(', ')}) (faded rays: negative w*)`);
      },
    },
  ],
};
