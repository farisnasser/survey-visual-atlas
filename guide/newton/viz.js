// Chapter 10 (survey Sections IV-A to IV-D, Appendix D-A). New visual 'newton1d' (the survey's L(x) = x^4 - x^2 of
// Section IV-C, damped as in Eq. 33); numbers in scripts/check-math/guide3_numbers.py. Reused: crash-course 'newton'
// (course_escape.py) and nesterov2006cubic 'model'.
import { HEX } from '../../src/lib/atlas3d.js';
import esc from '../../course/escape/viz.js';
import { paperTab } from '../lib.js';

const f = x => x ** 4 - x ** 2, g = x => 4 * x ** 3 - 2 * x, h = x => 12 * x * x - 2;
const crit = [[-Math.SQRT1_2, 'the minimum −1/√2'], [0, 'the local MAXIMUM 0'], [Math.SQRT1_2, 'the minimum +1/√2']];
const newton1d = {
  title: 'Newton on L(x) = x⁴ − x²: the step goes to the stationary point of the model, wherever that is',
  caption: 'At x₀ the quadratic model L(x₀) + gΔ + ½(h + λ)Δ² (dashed) has curvature h = L″(x₀) plus the damping λ. Newton\'s step jumps to the model\'s stationary point. Where h < 0 the model is an upside-down parabola: unbounded below, and its stationary point is a maximum, so the undamped step goes uphill toward the local maximum at 0. Damping with λ > −h makes the model a bowl again.',
  note: 'Exact derivatives: g = 4x³ − 2x, h = 12x² − 2. Dots: eight further damped Newton steps x ← x − g/(h + λ). Minima at ±1/√2 ≈ ±0.707, local maximum at 0, h < 0 for |x| < 0.408.',
  legend: [['#e8ecf8', 'L(x) = x⁴ − x²'], [HEX.opt2, 'model at x₀ (curvature h + λ)'], [HEX.ball, 'first step and further steps'], [HEX.good, 'minima']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'L(x)', xlim: [-1.5, 1.5], ylim: [-.6, 1.4], legend: false },
  controls: [
    { id: 'x0', label: 'starting point x₀', min: -1.4, max: 1.4, step: .01, value: .3 },
    { id: 'lam', label: 'damping λ (Eq. 33)', min: 0, max: 4, step: .05, value: 0 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const x0 = ctx.vals.x0, lam = ctx.vals.lam, P = ctx.plot.clear(), hl = h(x0) + lam;
    P.fn(f, { color: '#e8ecf8', width: 2 });
    P.fn(x => f(x0) + g(x0) * (x - x0) + .5 * hl * (x - x0) ** 2, { color: HEX.opt2, width: 1.6, dash: [5, 4] });
    P.scatter([[-Math.SQRT1_2, f(Math.SQRT1_2)], [Math.SQRT1_2, f(Math.SQRT1_2)]], { color: HEX.good, r: 5 });
    const xs = [x0];
    for (let k = 0; k < 9; k++) { const x = xs[xs.length - 1], d = h(x) + lam; if (Math.abs(d) < 1e-9) break; const n = x - g(x) / d; if (!isFinite(n) || Math.abs(n) > 1e3) break; xs.push(n); }
    P.line(xs.slice(0, 2).map(x => [x, f(x)]), { color: HEX.ball, width: 2.2 });
    P.scatter(xs.slice(1).filter(x => Math.abs(x) < 1.5).map(x => [x, f(x)]), { color: HEX.ball, r: 3.2 });
    P.scatter([[x0, f(x0)]], { color: '#ffffff', r: 5 });
    P.draw();
    const x1 = xs[1], last = xs[xs.length - 1], dL = f(x1) - f(x0), desc = g(x0) * (x1 - x0) < 0;
    const near = crit.reduce((a, c) => Math.abs(last - c[0]) < Math.abs(last - a[0]) ? c : a);
    ctx.stat(`x₀ = ${x0.toFixed(2)} · h = L″(x₀) = ${h(x0).toFixed(3)}, h + λ = ${hl.toFixed(3)} · step Δ = −g/(h + λ) = ${(x1 - x0).toFixed(4)} → x₁ = ${x1.toFixed(4)} · ${desc ? 'a descent direction' : 'NOT a descent direction (g·Δ > 0)'}, L changes by ${dL >= 0 ? '+' : ''}${dL.toFixed(4)} · after ${xs.length - 1} steps x = ${last.toFixed(4)}, ${Math.abs(last - near[0]) < 1e-3 ? 'at ' + near[1] : 'not yet at a critical point'}`);
  },
};

export default { newton1d, newton: esc.newton, model: await paperTab('nesterov2006cubic', 'model') };
