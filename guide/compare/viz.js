// Chapter 17 (survey Section VI). New visual 'dashboard': the measured runs behind the survey's Fig. 9
// (survey-v3/experiments/run_dashboard.py -> results.json -> scripts/check-math/guide3_numbers.py -> guide/compare/data.json).
import { HEX } from '../../src/lib/atlas3d.js';

const DATA = await fetch('guide/compare/data.json').then(r => r.json());
const NAMES = Object.keys(DATA.opt);
const COL = { 'SGD+momentum': HEX.pos, Adam: HEX.good, Lion: HEX.flat, Adafactor: HEX.muted, Shampoo: HEX.opt3, Muon: HEX.ball, 'Sophia-H': HEX.opt2 };
const FLOOR = 1.2e-7;
const PLOTS = {
  steps: { xlabel: 'gradient evaluations (a Hessian-vector product counts as two)', ylabel: 'training loss (median of 5 seeds; band: min to max)', logx: false, logy: true, xlim: [0, 800], ylim: [1e-7, 5] },
  time: { xlabel: 'optimizer wall-clock, seconds (one CPU thread, log scale)', ylabel: 'training loss', logx: true, logy: true, xlim: [5e-3, 3], ylim: [1e-7, 5] },
  memory: { xlabel: 'parameters (hidden width h = 64 … 1024)', ylabel: 'optimizer state, MiB (measured)', logx: true, logy: true, xlim: [7e3, 1.5e6], ylim: [1e-3, 60] },
  lr: { xlabel: 'learning rate ÷ that optimizer\'s best learning rate', ylabel: 'final training loss (median of 5 seeds)', logx: true, logy: true, xlim: [5e-5, 2e4], ylim: [1e-7, 20] },
};
const dashboard = {
  title: 'The survey\'s Fig. 9, live: seven optimizers on one task, every run measured',
  caption: 'One MLP (64-128-128-10) trained on the scikit-learn digits for 600 updates, 5 seeds, by seven optimizers, each given the same 9-point learning-rate grid. Choose a panel: loss against gradient evaluations (move the learning rate away from each optimizer\'s best), loss against measured wall-clock, measured optimizer memory against model size, or final loss against the learning rate.',
  note: 'Real runs (survey-v3/experiments/run_dashboard.py; full record in the survey\'s Table VIII): fp32, CPU (Intel Core Ultra 9 285H), PyTorch 2.11, batch 64, no schedule, no weight decay. Losses below 10⁻⁷ are drawn at the floor. In the LR panel, an x at the top marks a setting that diverged or made no progress.',
  legend: NAMES.map(n => [COL[n], n]),
  mode: '2d',
  plot: PLOTS.steps,
  controls: [
    { id: 'panel', type: 'seg', label: 'panel', value: 'steps', options: [['steps', '(a) loss vs evaluations'], ['time', '(b) loss vs time'], ['memory', '(c) memory'], ['lr', '(d) learning rate']] },
    { id: 'off', label: 'learning rate in panel (a): grid steps from each best', min: -4, max: 4, step: 1, value: 0, fmt: v => (v > 0 ? '+' : '') + v + ` (×${(10 ** (v / 2)).toPrecision(2)})` },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const pan = ctx.vals.panel, off = ctx.vals.off, P = ctx.plot.clear();
    P.set(Object.assign({ legend: false }, PLOTS[pan]));
    const bits = [];
    for (const n of NAMES) {
      const o = DATA.opt[n], bi = o.lrs.indexOf(DATA.best_lr[n]), c = COL[n];
      if (pan === 'steps') {
        const i = Math.max(0, Math.min(o.lrs.length - 1, bi + off)), cv = o.curves[i];
        const pts = cv.ge.map((x, k) => [x, cv.med[k] == null ? NaN : Math.max(cv.med[k], FLOOR)]);
        const lo = cv.ge.map((x, k) => [x, cv.lo[k] == null ? NaN : Math.max(cv.lo[k], FLOOR)]), hi = cv.ge.map((x, k) => [x, cv.hi[k] == null ? NaN : Math.max(Math.min(cv.hi[k], 5), FLOOR)]);
        P.custom((g, p) => { g.fillStyle = c + '22'; g.beginPath(); let ok = false; lo.forEach(([x, y], k) => { if (!isFinite(y) || !isFinite(hi[k][1])) return; const X = p.sx(x), Y = p.sy(y); if (!ok) { g.moveTo(X, Y); ok = true; } else g.lineTo(X, Y); }); for (let k = hi.length - 1; k >= 0; k--) { const [x, y] = hi[k]; if (isFinite(y) && isFinite(lo[k][1])) g.lineTo(p.sx(x), p.sy(y)); } g.closePath(); g.fill(); });
        P.line(pts, { color: c, width: 1.8 });
        const last = cv.med.filter(v => v != null).pop();
        bits.push(`${n} (lr ${o.lrs[i].toPrecision(2)}): ${cv.med.some(v => v == null) || last == null ? 'diverged' : last.toExponential(1)}`);
      } else if (pan === 'time') {
        const t = o.timing; P.line(t.t.slice(1).map((x, k) => [x, t.loss[k + 1] == null ? NaN : Math.max(t.loss[k + 1], FLOOR)]), { color: c, width: 1.8 });
        bits.push(`${n} ${t.total.toFixed(2)} s`);
      } else if (pan === 'memory') {
        P.line(o.memory.map(m => [m.params, m.state_bytes / 2 ** 20]), { color: c, width: 1.8, dots: 2.5 });
        const m128 = o.memory.find(m => m.h === 128);
        bits.push(`${n} ${(m128.state_bytes / 2 ** 20).toFixed(3)} MiB`);
      } else {
        const rel = o.lrs.map(l => l / DATA.best_lr[n]), med = o.final.map(f => { const v = f.map(x => x == null ? Infinity : x).sort((a, b) => a - b); return v[2]; });
        P.line(rel.map((r, k) => [r, isFinite(med[k]) && med[k] < 2.5 ? Math.max(med[k], FLOOR) : NaN]), { color: c, width: 1.6, dots: 2.5 });
        P.scatter(rel.map((r, k) => [r, med[k]]).filter(([, v]) => !(isFinite(v) && v < 2.5)).map(([r]) => [r, 9]), { color: c, r: 3 });
        const good = med.filter(v => isFinite(v) && v < 10 * Math.min(...med.filter(isFinite))).length;
        bits.push(`${n}: ${good} of 9 grid points within 10× of its best`);
      }
    }
    P.draw();
    const head = { steps: `loss after 600 updates, learning rate ${off === 0 ? 'at each optimizer\'s best' : `${off > 0 ? '+' : ''}${off} grid steps (×${(10 ** (off / 2)).toPrecision(2)}) from each best`}`, time: 'total optimizer time for 600 updates', memory: 'optimizer state at width 128 (26,122 parameters)', lr: 'how forgiving the learning rate is' }[pan];
    ctx.stat(`${head} · ${bits.join(' · ')}`);
  },
};

// ---------- 'memory' (survey Section VI-B, Table II): persistent optimizer state for one m × n weight matrix ----------
const fmtC = v => v >= 1e9 ? (v / 1e9).toPrecision(3) + 'G' : v >= 1e6 ? (v / 1e6).toPrecision(3) + 'M' : v >= 1e3 ? (v / 1e3).toPrecision(3) + 'k' : String(v);
const MEM = [
  ['SGD', HEX.muted, (m, n) => 0],
  ['momentum', HEX.pos, (m, n) => m * n],
  ['Lion', HEX.flat, (m, n) => m * n],
  ['Muon', HEX.ball, (m, n) => m * n],
  ['Adam', HEX.good, (m, n) => 2 * m * n],
  ['Sophia', HEX.opt2, (m, n) => 2 * m * n],
  ['Adafactor', '#b8c0dd', (m, n) => m + n],
  ['K-FAC', '#7fd3c6', (m, n) => m * m + n * n],
  ['Shampoo', HEX.opt3, (m, n) => m * m + n * n],
  ['L-BFGS, k=10', '#d9a3ff', (m, n) => 20 * m * n],
  ['full curvature', HEX.neg, (m, n) => (m * n) ** 2],
];
const memory = {
  title: 'Count the actual tensors: optimizer state for one m × n weight matrix',
  caption: 'Bars: how many numbers each optimizer keeps between steps for a single m × n weight matrix, following the survey\'s Table II (persistent state only; weights, gradients, activations and temporary buffers excluded). The log scale spans from a few thousand numbers to a full curvature matrix that no accelerator could hold. Change the shape: the structured methods (K-FAC, Shampoo) are always far below the full matrix, but never below Adam\'s two moment arrays, since m² + n² ≥ 2mn.',
  note: 'Counts of scalars, not bytes. SGD keeps nothing (drawn as an empty slot). Momentum, Lion and Muon keep one momentum array; Adam and Sophia keep two arrays the size of the weights; Adafactor (lean version) keeps row and column sums; K-FAC and Shampoo keep an m × m and an n × n matrix; L-BFGS keeps k = 10 pairs (s, y) of length mn; full curvature is the (mn) × (mn) matrix of dense Newton on this layer.',
  legend: [...MEM.map(r => [r[1], r[0]]), ['rgba(232,236,248,.6)', 'dashed line: mn, the weights themselves']],
  mode: '2d',
  plot: { xlabel: 'optimizers, in the order of the legend', ylabel: 'numbers kept between steps (log scale)', logy: true, ylim: [10, 1e17], legend: false, xfmt: () => '' },
  controls: [
    { id: 'lm', label: 'rows m', min: 4, max: 14, step: 1, value: 12, digits: 0, fmt: v => String(2 ** v) },
    { id: 'ln', label: 'columns n', min: 4, max: 14, step: 1, value: 10, digits: 0, fmt: v => String(2 ** v) },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const m = 2 ** ctx.vals.lm, n = 2 ** ctx.vals.ln, P = ctx.plot.clear(), vals = MEM.map(r => r[2](m, n));
    P.bars(MEM.map((r, i) => ({ y: Math.max(vals[i], 1e-9), color: r[1] })), { width: .72 });
    MEM.forEach((r, i) => {
      P.text(i, Math.max(vals[i], 10) * 1.9, vals[i] ? fmtC(vals[i]) : '0', { color: '#e8ecf8', align: 'center', base: 'bottom', font: '10px "JetBrains Mono", monospace' });
    });
    P.hline(m * n, { color: 'rgba(232,236,248,.45)', dash: [4, 4] });
    P.draw();
    const sh = (m * m + n * n) / (2 * m * n), af = (m + n) / (2 * m * n);
    ctx.stat(`W is ${m} × ${n} (${(m * n).toLocaleString('en')} weights) · Adam ${(2 * m * n).toLocaleString('en')} · Adafactor ${(m + n).toLocaleString('en')} (${(af * 100).toPrecision(2)}% of Adam) · K-FAC / Shampoo ${(m * m + n * n).toLocaleString('en')} (${sh.toFixed(2)}× Adam${m === n ? ', equal only because m = n' : ''}) · full curvature ${(m * n) ** 2 > 1e15 ? ((m * n) ** 2).toExponential(2) : ((m * n) ** 2).toLocaleString('en')} numbers`);
  },
};

export default { dashboard, memory };
