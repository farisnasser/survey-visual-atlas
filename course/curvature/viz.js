// Stop 4 visuals. Numbers checked in scripts/check-math/course_curvature.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours, heat } from '../../src/learn/plot2d.js';

const sgn3 = v => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);

// ---------- the second derivative and the 1-D quadratic model ----------
const F = x => .1 * x * x + .5 * Math.sin(3 * x), dF = x => .2 * x + 1.5 * Math.cos(3 * x), d2F = x => .2 - 4.5 * Math.sin(3 * x);
const bend = {
  title: 'The second derivative: which way the curve bends',
  caption: 'The curve is blue where f″ > 0 (it bends up, like a cup) and red where f″ < 0 (it bends down, like a cap). At the white point, the dashed tangent line uses f′ only; the violet parabola also uses f″ and hugs the curve much longer.',
  note: 'f(x) = 0.1x² + 0.5 sin 3x, the loss from stop 1; f″(x) = 0.2 − 4.5 sin 3x ranges from −4.3 to 4.7.',
  legend: [[HEX.pos, 'f″ > 0: bends up'], [HEX.neg, 'f″ < 0: bends down'], ['#e8ecf8', 'tangent line'], [HEX.opt2, 'quadratic model']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'f(x)', xlim: [-3, 3], ylim: [-1, 1.6], legend: false },
  controls: [
    { id: 'x', label: 'point x', min: -2.8, max: 2.8, step: .01, value: -.5 },
    { id: 'h', label: 'look-ahead step h', min: .05, max: 1, step: .01, value: .3 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { x, h } = ctx.vals, fx = F(x), g = dF(x), c = d2F(x), P = ctx.plot.clear();
    const up = [], dn = [];
    for (let t = -3; t <= 3.0001; t += .005) { const v = F(t), pos = d2F(t) >= 0; up.push([t, pos ? v : NaN]); dn.push([t, pos ? NaN : v]); }
    P.line(up, { color: HEX.pos, width: 2.4 }); P.line(dn, { color: HEX.neg, width: 2.4 });
    P.fn(t => fx + g * (t - x), { color: 'rgba(232,236,248,.75)', width: 1.4, dash: [6, 4] });
    P.fn(t => fx + g * (t - x) + .5 * c * (t - x) ** 2, { color: HEX.opt2, width: 1.8 });
    P.vline(x + h, { color: 'rgba(255,200,87,.45)', dash: [3, 4] });
    P.scatter([[x + h, F(x + h)]], { color: HEX.flat, r: 4 });
    P.scatter([[x, fx]], { color: '#ffffff', r: 5 });
    P.draw();
    const lin = fx + g * h, quad = lin + .5 * c * h * h;
    ctx.stat(`f′(x) = ${sgn3(g)} · f″(x) = ${sgn3(c)} → ${Math.abs(c) < 5e-4 ? 'no bend (inflection)' : c > 0 ? 'bends up' : 'bends down'} · at x + h: tangent error ${Math.abs(F(x + h) - lin).toFixed(3)}, parabola error ${Math.abs(F(x + h) - quad).toFixed(3)}`);
  },
};

// ---------- a quadratic bowl / saddle / dome with its eigen-directions ----------
const rotH = (l1, l2, th) => { const c = Math.cos(th), s = Math.sin(th); return [l1 * c * c + l2 * s * s, (l1 - l2) * c * s, l1 * s * s + l2 * c * c]; };
const shape = (l1, l2) => {
  const z = v => Math.abs(v) < 1e-9, p = v => v > 1e-9, n = v => v < -1e-9;
  if (z(l1) && z(l2)) return 'flat plane';
  if (p(l1) && p(l2)) return 'bowl (both curvatures up)';
  if (n(l1) && n(l2)) return 'dome (both curvatures down)';
  if ((p(l1) && n(l2)) || (n(l1) && p(l2))) return 'saddle (up one way, down the other)';
  return (p(l1) || p(l2)) ? 'trough (up one way, flat the other)' : 'ridge (down one way, flat the other)';
};
const quad = {
  title: 'Two numbers describe the bend: the eigenvalues',
  caption: 'The surface is ½ΔᵀHΔ for a 2×2 symmetric Hessian H. Its eigen-directions (tubes: blue = bends up, red = bends down, amber = flat) are perpendicular, and the curvature along any other direction u (coral) is a blend of the two eigenvalues.',
  note: 'Heights drawn to scale (×0.1). Defaults are the worked example H = [[4, 2], [2, 4]]: eigenvalues 6 and 2 along the diagonals.',
  legend: [[HEX.pos, 'λ > 0'], [HEX.neg, 'λ < 0'], [HEX.flat, 'λ = 0'], [HEX.ball, 'direction u']],
  mode: '3d', cam: [3.5, 3.0, 4.5], tgt: [0, 0, 0],
  controls: [
    { id: 'l1', label: 'eigenvalue λ₁', min: -6, max: 6, step: .1, value: 6, digits: 1 },
    { id: 'l2', label: 'eigenvalue λ₂', min: -6, max: 6, step: .1, value: 2, digits: 1 },
    { id: 'th', label: 'rotation of v₁ (degrees)', min: 0, max: 180, step: 1, value: 45, digits: 0 },
    { id: 'phi', label: 'direction u (degrees)', min: 0, max: 180, step: 1, value: 0, digits: 0 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { l1, l2, th, phi } = ctx.vals, [a, b, e] = rotH(l1, l2, th * Math.PI / 180), st = ctx.stage;
    st.clear();
    const d = def({ L: (x, y) => .5 * (a * x * x + 2 * b * x * y + e * y * y), g: (x, y) => [a * x + b * y, b * x + e * y], dom: [-1.3, 1.3, -1.3, 1.3], h: v => v, fit: null, hScale: .1 });
    d.hess = () => [a, b, e];
    st.surface(d, { opacity: .9 });
    st.eigenLines(d, 0, 0, { len: 1.25, eps: .05, fmtL: l => (Math.abs(l) < .05 ? '0' : l.toFixed(1)) });
    const p = phi * Math.PI / 180, u = [Math.cos(p), Math.sin(p)], pts = [];
    for (let t = -1.25; t <= 1.2501; t += .025) pts.push([t * u[0], t * u[1]]);
    st.tube(d, pts, C.ball, .014, .03);
    st.dot(d.W(0, 0, .03), C.white, .05);
    const cu = a * u[0] * u[0] + 2 * b * u[0] * u[1] + e * u[1] * u[1];
    ctx.stat(`H = [[${a.toFixed(2)}, ${b.toFixed(2)}], [${b.toFixed(2)}, ${e.toFixed(2)}]] · λ₁ = ${l1.toFixed(1)}, λ₂ = ${l2.toFixed(1)} → ${shape(l1, l2)} · curvature along u: uᵀHu = ${cu.toFixed(2)}`);
  },
};

// ---------- GD on a tilted quadratic, one eigen-direction at a time ----------
const TH = Math.PI / 6, V1 = [Math.cos(TH), Math.sin(TH)], V2 = [-Math.sin(TH), Math.cos(TH)], X0 = [-2.2, -.6];
const DOMZ = [-2.6, 2.6, -1.7, 1.7], CDZ = [-4.8, 4.8, -1.8, 1.8];   // CDZ: drawn area (equal aspect widens the view)
const zigzag = {
  title: 'Gradient descent, one eigen-direction at a time',
  caption: 'f = ½(μ u₁² + L u₂²), where u₁, u₂ are the coordinates along the tilted eigen-directions (μ = 1 flat, L = κ sharp). In those coordinates every step multiplies u₁ by 1 − ημ and u₂ by 1 − ηL, exactly as on stop 3\'s parabola.',
  note: 'Chart: dots are the run, lines are (1 − ηλ)ᵗ times the start. They coincide because the function is exactly quadratic.',
  legend: [[HEX.ball, 'GD path'], [HEX.pos, 'eigen-directions'], [HEX.good, 'minimum']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'y', xlim: [DOMZ[0], DOMZ[1]], ylim: [DOMZ[2], DOMZ[3]], equal: true, legend: false, grid: false },
  charts: [{ id: 'u', title: 'size of each eigen-coordinate', xlabel: 'step t', ylabel: '|uᵢ|', logy: true, xlim: [0, 80], ylim: [1e-8, 30], h: 200 }],
  controls: [
    { id: 'eta', label: 'step size η', min: .01, max: .25, step: .001, value: .18, digits: 3 },
    { id: 'k', label: 'condition number κ = L/μ', min: 1, max: 30, step: .5, value: 10, digits: 1 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { eta, k } = ctx.vals, Hm = [1 * V1[0] * V1[0] + k * V2[0] * V2[0], 1 * V1[0] * V1[1] + k * V2[0] * V2[1], 1 * V1[1] * V1[1] + k * V2[1] * V2[1]];
    const f = (x, y) => .5 * (Hm[0] * x * x + 2 * Hm[1] * x * y + Hm[2] * y * y), P = ctx.plot.clear();
    heat(P, (x, y) => Math.log1p(f(x, y)), CDZ, { alpha: .4, n: 90 });
    drawContours(P, contours(f, CDZ, [.02, .1, .3, .7, 1.4, 2.6, 4.5, 7.5, 12, 19], 140));
    for (const [v, lab, at] of [[V1, 'u₁ (μ = 1)', 3], [V2, `u₂ (L = ${k.toFixed(1)})`, 1.45]]) {
      P.line([[-3 * v[0], -3 * v[1]], [3 * v[0], 3 * v[1]]], { color: 'rgba(124,196,255,.55)', width: 1.2, dash: [5, 4] });
      P.text(at * v[0], at * v[1], lab, { color: HEX.pos, dx: 6, dy: -8 });
    }
    const path = [X0.slice()], u1 = [], u2 = [];
    let x = X0.slice(), hit = null, out = false;
    const c1 = V1[0] * X0[0] + V1[1] * X0[1], c2 = V2[0] * X0[0] + V2[1] * X0[1];
    for (let t = 0; t <= 400; t++) {
      if (t <= 80) { u1.push([t, Math.abs(V1[0] * x[0] + V1[1] * x[1])]); u2.push([t, Math.abs(V2[0] * x[0] + V2[1] * x[1])]); }
      if (t === 400) break;
      const g = [Hm[0] * x[0] + Hm[1] * x[1], Hm[1] * x[0] + Hm[2] * x[1]];
      x = [x[0] - eta * g[0], x[1] - eta * g[1]];
      if (Math.max(Math.abs(x[0]), Math.abs(x[1])) > 50) { out = true; break; }
      if (t < 120) path.push(x.slice());
      if (hit === null && f(...x) < 1e-6) hit = t + 1;
    }
    P.line(path, { color: HEX.ball, width: 1.6, alpha: .9 });
    P.scatter(path, { color: HEX.ball, r: 2.4 });
    P.scatter([[0, 0]], { color: HEX.good, r: 5 });
    P.scatter([X0], { color: '#ffffff', r: 4.5 });
    P.draw();
    const q1 = 1 - eta, q2 = 1 - eta * k, Cc = ctx.charts.u.clear();
    Cc.fn(t => Math.abs(c1) * Math.abs(q1) ** t, { color: HEX.opt2, width: 1.4, label: `|1 − ημ|ᵗ|u₁(0)|` });
    Cc.fn(t => Math.abs(c2) * Math.abs(q2) ** t, { color: HEX.pos, width: 1.4, label: `|1 − ηL|ᵗ|u₂(0)|` });
    Cc.scatter(u1, { color: HEX.opt2, r: 2.2 }); Cc.scatter(u2, { color: HEX.pos, r: 2.2 });
    Cc.draw();
    ctx.stat(`factor along u₁: 1 − ημ = ${q1.toFixed(3)} · along u₂: 1 − ηL = ${q2.toFixed(3)} · stable needs η < 2/L = ${(2 / k).toFixed(3)} · ${out ? 'diverged' : hit ? `f < 10⁻⁶ after ${hit} steps` : 'f ≥ 10⁻⁶ after 400 steps'}`);
  },
};

// ---------- steps needed vs condition number ----------
function stepsNeeded(k, eta, tol = 1e-6, T = 200000) {
  let u1 = 1, u2 = 1; const q1 = 1 - eta, q2 = 1 - eta * k;
  for (let t = 1; t <= T; t++) { u1 *= q1; u2 *= q2; if (Math.abs(u1) <= tol && Math.abs(u2) <= tol) return t; }
  return null;
}
const KS = [1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000];
const SN1 = KS.map(k => [k, stepsNeeded(k, 1 / k)]), SN2 = KS.map(k => [k, stepsNeeded(k, 2 / (k + 1))]);
const kappa = {
  title: 'How many steps a narrow valley costs',
  caption: 'Gradient descent on ½(μu₁² + Lu₂²) from u = (1, 1), until both coordinates are below 10⁻⁶. The count grows in proportion to κ: ten times narrower means ten times slower.',
  note: 'Dots: measured runs. Dashed: κ ln(10⁶) and (κ/2) ln(10⁶). Both axes logarithmic.',
  legend: [[HEX.ball, 'η = 1/L'], [HEX.opt2, 'best fixed η = 2/(L + μ)']],
  mode: '2d',
  plot: { xlabel: 'condition number κ', ylabel: 'steps to shrink by 10⁶', logx: true, logy: true, xlim: [1, 1000], ylim: [1, 3e4], legend: false },
  controls: [{ id: 'lk', label: 'condition number κ', min: 0, max: 3, step: .01, value: 1, fmt: v => (10 ** v).toFixed(10 ** v < 10 ? 1 : 0) }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const k = 10 ** ctx.vals.lk, P = ctx.plot.clear(), ln = Math.log(1e6);
    P.fn(x => x * ln, { color: 'rgba(255,122,89,.55)', width: 1.2, dash: [5, 4] });
    P.fn(x => x * ln / 2, { color: 'rgba(178,140,255,.55)', width: 1.2, dash: [5, 4] });
    P.line(SN1, { color: HEX.ball, width: 1.6, dots: 3.2 });
    P.line(SN2, { color: HEX.opt2, width: 1.6, dots: 3.2 });
    P.vline(k, { color: 'rgba(255,200,87,.6)', dash: [3, 4] });
    const a = stepsNeeded(k, 1 / k), b = stepsNeeded(k, 2 / (k + 1));
    P.scatter([[k, a], [k, b]], { color: '#ffffff', r: 4.5 });
    P.draw();
    ctx.stat(`κ = ${k.toFixed(k < 10 ? 2 : 0)} · η = 1/L: factor 1 − 1/κ = ${(1 - 1 / k).toFixed(4)}, ${a.toLocaleString('en')} steps · η = 2/(L + μ): factor (κ − 1)/(κ + 1) = ${((k - 1) / (k + 1)).toFixed(4)}, ${b.toLocaleString('en')} steps`);
  },
};

// ---------- Hessian-vector products from two gradients ----------
const gp = (x, y) => [.6 * x + .6 * Math.cos(1.5 * x) * Math.cos(y), .4 * y - .4 * Math.sin(1.5 * x) * Math.sin(y)];
const Hp = (x, y) => [.6 - .9 * Math.sin(1.5 * x) * Math.cos(y), -.6 * Math.cos(1.5 * x) * Math.sin(y), .4 - .4 * Math.sin(1.5 * x) * Math.cos(y)];
const PX = [1.1, .9], VV = [Math.SQRT1_2, Math.SQRT1_2];
const HM = Hp(...PX), HV = [HM[0] * VV[0] + HM[1] * VV[1], HM[1] * VV[0] + HM[2] * VV[1]];
const fdErr = (h, central) => {
  const a = gp(PX[0] + h * VV[0], PX[1] + h * VV[1]), b = central ? gp(PX[0] - h * VV[0], PX[1] - h * VV[1]) : gp(...PX), s = central ? 2 * h : h;
  return [(a[0] - b[0]) / s, (a[1] - b[1]) / s];
};
const HS = []; for (let e = -12; e <= 1e-9; e += .125) HS.push(10 ** e);
const ERRF = HS.map(h => { const q = fdErr(h, false); return [h, Math.max(Math.hypot(q[0] - HV[0], q[1] - HV[1]), 1e-17)]; });
const ERRC = HS.map(h => { const q = fdErr(h, true); return [h, Math.max(Math.hypot(q[0] - HV[0], q[1] - HV[1]), 1e-17)]; });
const hvp = {
  title: 'Curvature from two gradients',
  caption: 'H v without the matrix: the change of the gradient over a tiny step h along v, divided by h. Too large an h and the curve of f spoils the estimate; too small and rounding errors in the subtraction take over.',
  note: 'Stop 2\'s surface 0.3x² + 0.2y² + 0.4 sin 1.5x cos y at (1.1, 0.9), v = (1, 1)/√2. Error = distance to the exact H v.',
  legend: [[HEX.ball, 'forward: (∇f(x + hv) − ∇f(x))/h'], [HEX.opt2, 'central: (∇f(x + hv) − ∇f(x − hv))/(2h)']],
  mode: '2d',
  plot: { xlabel: 'step h', ylabel: 'error of the estimate', logx: true, logy: true, xlim: [1e-12, 1], ylim: [1e-13, 1], legend: false },
  controls: [{ id: 'lh', label: 'step h', min: -12, max: 0, step: .125, value: -4, fmt: v => '10^' + v.toFixed(v % 1 ? 2 : 0) }],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const h = 10 ** ctx.vals.lh, P = ctx.plot.clear();
    P.line(ERRF, { color: HEX.ball, width: 2 });
    P.line(ERRC, { color: HEX.opt2, width: 2 });
    P.vline(h, { color: 'rgba(255,200,87,.6)', dash: [3, 4] });
    const q = fdErr(h, false), e = Math.hypot(q[0] - HV[0], q[1] - HV[1]);
    P.scatter([[h, Math.max(e, 1e-17)]], { color: '#ffffff', r: 4.5 });
    P.draw();
    ctx.stat(`exact H v = (${HV[0].toFixed(4)}, ${HV[1].toFixed(4)}) · forward estimate at h = ${h.toExponential(0)}: (${q[0].toFixed(6)}, ${q[1].toFixed(6)}) · error ${e.toExponential(1)}`);
  },
};

export default { bend, quad, zigzag, kappa, hvp };
