// Chapter 12 (survey Sections IV-H to IV-J, Appendix C-D). New visual 'natgrad': maximum-likelihood fit of a Gaussian
// N(mu, sigma^2), GD vs natural gradient (Eq. 35) in two parameterizations; numbers in scripts/check-math/guide3_numbers.py.
import { HEX } from '../../src/lib/atlas3d.js';
import { rampCss } from '../../src/lib/charts.js';

const MBAR = 1, SBAR = .5, LSTAR = Math.log(SBAR) + .5;
const NLL = (mu, s) => Math.log(s) + (SBAR * SBAR + (mu - MBAR) ** 2) / (2 * s * s);           // average negative log-likelihood (+ const)
const dmu = (mu, s) => (mu - MBAR) / (s * s), dsig = (mu, s) => 1 / s - (SBAR * SBAR + (mu - MBAR) ** 2) / s ** 3;
const START = [-1.5, 2.5], T = 300, TOL = 1e-8;
/** one step of each method; state p = [mu, sigma] in the plane, whatever the internal coordinates */
const STEP = {
  gds: (p, e) => [p[0] - e * dmu(...p), p[1] - e * dsig(...p)],                                   // GD in (mu, sigma)
  gdl: (p, e) => { const l = Math.log(p[1]) - e * p[1] * dsig(...p); return [p[0] - e * dmu(...p), Math.exp(l)]; },        // GD in (mu, log sigma)
  ngs: (p, e) => [p[0] - e * p[1] ** 2 * dmu(...p), p[1] - e * (p[1] ** 2 / 2) * dsig(...p)],                              // F = diag(1/s^2, 2/s^2)
  ngl: (p, e) => { const l = Math.log(p[1]) - e * .5 * p[1] * dsig(...p); return [p[0] - e * p[1] ** 2 * dmu(...p), Math.exp(l)]; }, // F = diag(1/s^2, 2)
};
const META = { gds: ['GD in (μ, σ)', HEX.opt2], gdl: ['GD in (μ, log σ)', HEX.opt3], ngs: ['natural gradient in (μ, σ)', HEX.ball], ngl: ['natural gradient in (μ, log σ)', HEX.flat] };
function run(k, eta) {
  const P = [START.slice()]; let hit = null;
  for (let t = 1; t <= T; t++) {
    const q = STEP[k](P[P.length - 1], eta);
    if (!isFinite(q[0]) || !isFinite(q[1]) || q[1] <= 0) { P.push([NaN, NaN]); break; }
    P.push(q); if (hit === null && NLL(...q) - LSTAR < TOL) hit = t;
  }
  return { P, hit };
}
const GRID = (() => { const nx = 70, ny = 60, z = []; for (let j = 0; j < ny; j++) { const s = .05 + 2.95 * (j + .5) / ny, row = []; for (let i = 0; i < nx; i++) { const mu = -2 + 4.5 * (i + .5) / nx; row.push(Math.log10(NLL(mu, s) - LSTAR + 1e-4)); } z.push(row); } return z; })();

const natgrad = {
  title: 'Natural gradient follows the predictions, not the coordinates',
  caption: 'Fit a Gaussian N(μ, σ²) to data with mean 1 and spread 0.5 by minimizing the average negative log-likelihood, from μ = −1.5, σ = 2.5. The same model can be written with parameters (μ, σ) or (μ, log σ). Plain gradient descent (violet, white) follows a different path in each. Natural gradient −ηF⁻¹g (coral, amber) uses the Fisher matrix of whichever parameters it is given, and the two paths almost coincide.',
  note: 'Exact gradients and Fisher matrices: F = diag(1/σ², 2/σ²) in (μ, σ) and diag(1/σ², 2) in (μ, log σ). Both paths are drawn in the (μ, σ) plane. Background: log of the loss gap. A GD run that makes σ ≤ 0 is stopped.',
  legend: Object.values(META).map(([n, c]) => [c, n]).concat([[HEX.good, 'optimum (1, 0.5)']]),
  mode: '2d',
  plot: { xlabel: 'mean μ', ylabel: 'standard deviation σ', xlim: [-2, 2.5], ylim: [.05, 3], legend: false, grid: false },
  controls: [{ id: 'eta', label: 'step size η (same for all four)', min: .02, max: 1, step: .01, value: .15 }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const eta = ctx.vals.eta, P = ctx.plot.clear();
    P.heat(GRID, [-2, 2.5, .05, 3], { cell: z => rampCss(Math.min(1, Math.max(0, (z + 4) / 5.2)) * .75) });
    const R = {};
    for (const k of ['gds', 'gdl', 'ngs', 'ngl']) { R[k] = run(k, eta); P.line(R[k].P, { color: META[k][1], width: k === 'ngl' ? 1.4 : 2, dash: k === 'ngl' ? [5, 4] : null, dots: 1.6 }); }
    P.scatter([[MBAR, SBAR]], { color: HEX.good, r: 6 }); P.scatter([START], { color: '#ffffff', r: 4 });
    P.draw();
    const gap = (a, b, n = 40) => { let m = 0; for (let t = 0; t <= n && t < a.length && t < b.length; t++) { if (!isFinite(a[t][0]) || !isFinite(b[t][0])) return Infinity; m = Math.max(m, Math.hypot(a[t][0] - b[t][0], a[t][1] - b[t][1])); } return m; };
    const say = k => R[k].hit ? `${R[k].hit} steps` : (R[k].P.some(p => !isFinite(p[0])) ? 'stopped (σ ≤ 0)' : `> ${T}`);
    ctx.stat(`η = ${eta.toFixed(2)} · steps to a loss gap below 10⁻⁸: GD (μ, σ) ${say('gds')}, GD (μ, log σ) ${say('gdl')}, natural (μ, σ) ${say('ngs')}, natural (μ, log σ) ${say('ngl')} · largest distance between the two paths over the first 40 steps: GD ${gap(R.gds.P, R.gdl.P).toFixed(3)}, natural ${gap(R.ngs.P, R.ngl.P).toFixed(3)}`);
  },
};

export default { natgrad };
