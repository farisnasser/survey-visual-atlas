// Chapter 11 (survey Section IV-D). New visual 'normgd' (checked in scripts/check-math/guide_why.py);
// reused paper tabs: damian2023self, wang2022analyzing, chen2023beyond, kalra2025universal (each with its own check script).
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { paperTab } from '../lib.js';

// Arora, Li & Panigrahi (2022), Fig. 1: L(x, y) = (1 + x^2) y^2. Minimizers: the line y = 0, sharpness 2(1 + x^2).
const L = (x, y) => (1 + x * x) * y * y;
const gL = (x, y) => [2 * x * y * y, 2 * (1 + x * x) * y];
const gSqrt = (x, y) => { const r = Math.sqrt(1 + x * x); return [x / r * Math.abs(y), r * Math.sign(y)]; };
const STEP = {
  sqrt: (p) => { const g = gSqrt(p[0], p[1]); return [p[0] - .1 * g[0], p[1] - .1 * g[1]]; },
  ngd: (p) => { const g = gL(p[0], p[1]), n = Math.hypot(g[0], g[1]); if (n === 0) return p; return [p[0] - .1 * g[0] / n, p[1] - .1 * g[1] / n]; },
  gd: (p) => { const g = gL(p[0], p[1]); return [p[0] - .02 * g[0], p[1] - .02 * g[1]]; },
};
const NAME = { sqrt: 'GD on √L (η = 0.1)', ngd: 'normalized GD on L (η = 0.1)', gd: 'plain GD on L (η = 0.02)' };
const sharp = x => 2 * (1 + x * x);
const normgd = {
  title: 'Oscillating across the valley, drifting along it to the flattest minimum',
  caption: 'Arora, Li & Panigrahi’s Fig. 1 example: L = (1 + x²)y². Every point of the line y = 0 is a minimum, with sharpness 2(1 + x²): flattest at x = 0. Plain GD with a small step stops at the nearest minimum. GD on √L and normalized GD bounce across the valley and slide along it to x = 0.',
  note: 'Exact runs of 3000 steps from (3, 0.4). GD on √L and normalized GD use η = 0.1; plain GD uses η = 0.02. Height drawn as √L; the y-axis is stretched so the valley is visible.',
  legend: [[HEX.ball, 'the chosen method'], [HEX.good, 'minimizers y = 0'], [HEX.flat, 'sharpness 2(1 + x_t²)'], [HEX.opt2, '|y_t|']],
  mode: '3d', cam: [2.6, 2.3, 2.9], tgt: [0, .15, 0],
  controls: [{ id: 'm', type: 'seg', label: 'method', value: 'sqrt', options: [['sqrt', 'GD on √L'], ['ngd', 'normalized GD'], ['gd', 'plain GD, small step']] }],
  action: 'Restart',
  charts: [
    { id: 's', title: 'sharpness at the nearest minimizer, 2(1 + x_t²)', xlabel: 'step t', ylabel: 'sharpness', xlim: [0, 3000], ylim: [0, 22], h: 190, legend: false },
    { id: 'y', title: 'distance across the valley |y_t| (log)', xlabel: 'step t', ylabel: '|y_t|', logy: true, xlim: [0, 3000], ylim: [1e-6, 1], h: 190, legend: false },
  ],
  build(ctx) {
    const d = ctx.d = def({ L, g: gL, h: v => Math.sqrt(v), dom: [-3.5, 3.5, -.6, .6], square: true, fit: .9, N: 120 });
    ctx.stage.surface(d);
    ctx.stage.tube(d, Array.from({ length: 71 }, (_, i) => [-3.5 + i * .1, 0]), C.good, .016, .015);
    ctx.stage.label('flattest minimum (x = 0)', d.W(0, 0, .25), '', HEX.good);
    this.drop(ctx);
  },
  onControl(ctx) { this.drop(ctx); },
  drop(ctx) {
    const st = ctx.stage, m = ctx.vals.m;
    st.clearBalls(); ctx.hist = [[0, 3, .4]];
    ctx.ball = st.ball(ctx.d, { start: [3, .4], step: STEP[m], speed: m === 'gd' ? 20 : 8, maxSteps: 3000, r: .05 });
    ctx.ball.onStep = b => ctx.hist.push([b.steps, b.p[0], b.p[1]]);
    this.chart(ctx);
  },
  chart(ctx) {
    const H = ctx.hist, S = ctx.charts.s.clear(), Y = ctx.charts.y.clear();
    S.hline(2, { color: HEX.good, dash: [3, 4], text: 'flattest: 2' });
    S.line(H.map(h => [h[0], sharp(h[1])]), { color: HEX.flat, width: 1.8 });
    Y.line(H.map(h => [h[0], Math.max(1e-6, Math.abs(h[2]))]), { color: HEX.opt2, width: 1 });
    S.draw(); Y.draw();
    const [t, x, y] = H[H.length - 1];
    ctx.stat(`${NAME[ctx.vals.m]} · step ${t} · x = ${x.toFixed(4)}, |y| = ${Math.abs(y).toExponential(1)} · sharpness of the nearest minimizer 2(1 + x²) = ${sharp(x).toFixed(3)} (started at 20)`);
  },
  frame(ctx) { const b = ctx.ball; if (b && b.steps !== ctx.last) { ctx.last = b.steps; this.chart(ctx); } },
};

export default {
  stages: await paperTab('damian2023self', 'stages', { title: 'Self-stabilization on Damian et al.’s toy: four stages that repeat' }),
  orbit: await paperTab('damian2023self', 'orbit', { title: 'Oscillation size against sharpness: a predator–prey cycle' }),
  trace: await paperTab('wang2022analyzing', 'trace', { title: 'Four phases in a real two-layer network, and the output-layer norm that tracks them' }),
  normgd,
  bif: await paperTab('chen2023beyond', 'bif', { title: 'Beyond the edge: stable period-2 orbits, then period doubling' }),
  chaos: await paperTab('kalra2025universal', 'chaos', { title: 'A two-layer linear model on one example: the route to chaos' }),
};
