// Chapter 6 (survey Sections II-G, II-H). New visual 'metric': Eq. (12) and its argmin form drawn literally. Numbers in
// scripts/check-math/guide3_numbers.py.
import { HEX } from '../../src/lib/atlas3d.js';

const TH = 30 * Math.PI / 180, L1 = 6, L2 = .6;
const V1 = [Math.cos(TH), Math.sin(TH)], V2 = [-Math.sin(TH), Math.cos(TH)];
const H = [[L1 * V1[0] ** 2 + L2 * V2[0] ** 2, L1 * V1[0] * V1[1] + L2 * V2[0] * V2[1]], [0, 0]];
H[1] = [H[0][1], L1 * V1[1] ** 2 + L2 * V2[1] ** 2];
const mv = (M, x) => [M[0][0] * x[0] + M[0][1] * x[1], M[1][0] * x[0] + M[1][1] * x[1]];
const Lf = p => .5 * (p[0] * mv(H, p)[0] + p[1] * mv(H, p)[1]);
const inv2 = M => { const d = M[0][0] * M[1][1] - M[0][1] * M[1][0]; return [[M[1][1] / d, -M[0][1] / d], [-M[1][0] / d, M[0][0] / d]]; };
function metricOf(kind, g) {
  if (kind === 'I') return [[1, 0], [0, 1]];
  if (kind === 'D') return [[H[0][0], 0], [0, H[1][1]]];
  if (kind === 'H') return H;
  return [[Math.abs(g[0]) + 1e-12, 0], [0, Math.abs(g[1]) + 1e-12]];      // diag(|g|): Adam's first step, a sign step
}
const NAMES = { I: 'M = I (gradient descent)', D: 'M = diag(H) (diagonal curvature)', H: 'M = H (Newton)', S: 'M = diag(|g|) (Adam\'s first step: a sign step)' };
/** ellipse {Δ : ΔᵀMΔ = r²} around p */
function ellipse(M, p, r) {
  const a = M[0][0], b = M[0][1], c = M[1][1], m = (a + c) / 2, d = Math.hypot((a - c) / 2, b), l1 = m + d, l2 = m - d;
  let v = Math.abs(b) > 1e-12 ? [l1 - c, b] : (a >= c ? [1, 0] : [0, 1]); const n = Math.hypot(...v); v = [v[0] / n, v[1] / n];
  const u = [-v[1], v[0]];
  return Array.from({ length: 121 }, (_, i) => { const t = 2 * Math.PI * i / 120, x = r / Math.sqrt(l1) * Math.cos(t), y = r / Math.sqrt(l2) * Math.sin(t); return [p[0] + x * v[0] + y * u[0], p[1] + x * v[1] + y * u[1]]; });
}
const levels = c => Array.from({ length: 121 }, (_, i) => { const t = 2 * Math.PI * i / 120, x = Math.sqrt(2 * c / L1) * Math.cos(t), y = Math.sqrt(2 * c / L2) * Math.sin(t); return [x * V1[0] + y * V2[0], x * V1[1] + y * V2[1]]; });
const COL = { I: HEX.opt2, D: HEX.ball, H: HEX.opt3, S: HEX.flat };

const metric = {
  title: 'One formula, four optimizers: the step that minimizes g·Δ + ΔᵀMΔ/(2η)',
  caption: 'At the white point, every method sees the same gradient g. Each choice of metric M draws a different ellipse ΔᵀMΔ = r² of "equally long" steps, and the step Δ = −ηM⁻¹g of Eq. (12) is the point of that ellipse where the linear model g·Δ is lowest. The dashed line through the tip is the level line of the linear model: it just touches the ellipse there.',
  note: 'L = ½θᵀHθ with curvatures 6 and 0.6 along axes turned by 30°; grey curves are its level sets, green is the minimum. r is set to η√(gᵀM⁻¹g), so the ellipse passes through the step.',
  legend: [[HEX.opt2, 'M = I'], [HEX.ball, 'M = diag(H)'], [HEX.opt3, 'M = H'], [HEX.flat, 'M = diag(|g|)'], [HEX.good, 'minimum']],
  mode: '2d',
  plot: { xlabel: 'θ₁', ylabel: 'θ₂', xlim: [-2.2, 2.2], ylim: [-2.2, 2.2], equal: true, legend: false, grid: false },
  controls: [
    { id: 'M', type: 'seg', label: 'metric M', value: 'I', options: [['I', 'I'], ['D', 'diag(H)'], ['H', 'H'], ['S', 'diag(|g|)']] },
    { id: 'eta', label: 'step size η', min: .02, max: 1.5, step: .01, value: .25 },
    { id: 'px', label: 'point θ₁', min: -1.8, max: 1.8, step: .01, value: -1.2 },
    { id: 'py', label: 'point θ₂', min: -1.8, max: 1.8, step: .01, value: 1.1 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const p = [ctx.vals.px, ctx.vals.py], g = mv(H, p), kind = ctx.vals.M, eta = ctx.vals.eta;
    const M = metricOf(kind, g), Mi = inv2(M), s = mv(Mi, g), D = [-eta * s[0], -eta * s[1]];
    const r = eta * Math.sqrt(g[0] * s[0] + g[1] * s[1]);
    const P = ctx.plot.clear();
    for (const c of [.05, .2, .5, 1, 2, 3.5, 5.5]) P.line(levels(c), { color: 'rgba(160,170,210,.28)', width: 1 });
    P.line(ellipse(M, p, r), { color: COL[kind], width: 1.6, dash: [5, 3] });
    const tip = [p[0] + D[0], p[1] + D[1]], gn = Math.hypot(...g), t = [-g[1] / gn, g[0] / gn];
    P.line([[tip[0] - 3 * t[0], tip[1] - 3 * t[1]], [tip[0] + 3 * t[0], tip[1] + 3 * t[1]]], { color: 'rgba(255,255,255,.35)', dash: [2, 4], width: 1 });
    P.arrows([[p[0], p[1], -.35 * g[0] / gn, -.35 * g[1] / gn]], { color: 'rgba(255,255,255,.6)' });
    P.arrows([[p[0], p[1], D[0], D[1]]], { color: COL[kind], width: 2.2 });
    P.scatter([[0, 0]], { color: HEX.good, r: 5 }); P.scatter([p], { color: '#ffffff', r: 4.5 });
    P.draw();
    const cosang = -(g[0] * D[0] + g[1] * D[1]) / (gn * Math.hypot(...D)), ang = Math.acos(Math.max(-1, Math.min(1, cosang))) * 180 / Math.PI;
    ctx.stat(`${NAMES[kind]} · g = (${g[0].toFixed(2)}, ${g[1].toFixed(2)}) · step Δ = −ηM⁻¹g = (${D[0].toFixed(3)}, ${D[1].toFixed(3)}), ${ang.toFixed(1)}° away from −g · loss ${Lf(p).toFixed(4)} → ${Lf(tip).toFixed(4)} · linear-model decrease g·Δ = ${(g[0] * D[0] + g[1] * D[1]).toFixed(4)}`);
  },
};

export default { metric };
