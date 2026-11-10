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

export default { dashboard };
