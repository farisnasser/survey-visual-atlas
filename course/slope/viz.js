// Stop 2 visuals. Numbers checked in scripts/check-math/course_slope.py.
import { def, C, HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours, heat, arrow, draggable } from '../../src/learn/plot2d.js';

const THREE = window.THREE;
const f3 = v => (Math.abs(v) < 5e-4 ? 0 : v).toFixed(3);

// ---------- secant -> tangent ----------
const F = t => .1 * t * t + .5 * Math.sin(3 * t), dF = t => .2 * t + 1.5 * Math.cos(3 * t);
const secant = {
  title: 'From a secant to the tangent',
  caption: 'The coral line passes through x and x + h. Shrink h and watch its slope settle onto the derivative f′(x), the slope of the violet tangent.',
  note: 'f(x) = 0.1x² + 0.5 sin 3x, the loss from stop 1; f′(x) = 0.2x + 1.5 cos 3x.',
  legend: [[HEX.ball, 'secant through x and x + h'], [HEX.opt2, 'tangent at x']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'f(x)', xlim: [-3, 3], ylim: [-.8, 1.4], legend: false },
  controls: [
    { id: 'x', label: 'point x', min: -2.6, max: 2.6, step: .01, value: -1.2 },
    { id: 'k', label: 'step h', min: -3, max: 0, step: .01, value: -.1, fmt: v => (10 ** v).toPrecision(2) },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const x = ctx.vals.x, h = 10 ** ctx.vals.k, s = (F(x + h) - F(x)) / h, d = dF(x), P = ctx.plot.clear();
    P.fn(F, { color: '#8ea0d0', width: 2 });
    P.fn(t => F(x) + d * (t - x), { color: HEX.opt2, width: 1.6, dash: [6, 4] });
    P.fn(t => F(x) + s * (t - x), { color: HEX.ball, width: 2 });
    P.scatter([[x, F(x)]], { color: '#e8ecf8', r: 5 });
    P.scatter([[x + h, F(x + h)]], { color: HEX.ball, r: 5 });
    P.draw();
    ctx.stat(`h = ${h.toPrecision(2)} · secant slope (f(x+h) − f(x))/h = ${s.toFixed(4)} · derivative f′(x) = ${d.toFixed(4)} · gap ${Math.abs(s - d).toExponential(1)}`);
  },
};

// ---------- gradient compass on a contour map ----------
const FUN = {
  bowl: { f: (x, y) => x * x + 3 * y * y, g: (x, y) => [2 * x, 6 * y], name: 'f(x, y) = x² + 3y²', levels: [.25, .75, 1.5, 2.5, 3.75, 5.25, 7, 9] },
  saddle: { f: (x, y) => x * x - y * y, g: (x, y) => [2 * x, -2 * y], name: 'f(x, y) = x² − y²', levels: [-3, -2, -1.2, -.5, 0, .5, 1.2, 2, 3] },
};
const DOM = [-2, 2, -1.6, 1.6], CD = [-4.5, 4.5, -2.8, 2.8];   // CD: drawn area (equal aspect may widen the view)
const compass = {
  title: 'The gradient is the steepest way up',
  caption: 'Drag the white point. The coral arrow is ∇f; the ring shows, for every direction u, how fast f changes (coral = up, blue = down). Turn u with the slider.',
  note: 'The highlighted curve is the contour through the point: ∇f always meets it at a right angle.',
  legend: [[HEX.ball, '∇f(x)'], [HEX.opt2, 'direction u'], [HEX.flat, 'contour through x']],
  mode: '2d',
  plot: { xlabel: 'x', ylabel: 'y', xlim: [DOM[0], DOM[1]], ylim: [DOM[2], DOM[3]], equal: true, legend: false, grid: false },
  controls: [
    { id: 'fn', type: 'seg', label: 'function', value: 'bowl', options: [['bowl', 'x² + 3y²'], ['saddle', 'x² − y²']] },
    { id: 'phi', label: 'direction u (degrees)', min: 0, max: 360, step: 1, value: 200, digits: 0 },
  ],
  build(ctx) {
    ctx.p = [1.1, .55];
    draggable(ctx, q => { ctx.p = [Math.max(DOM[0], Math.min(DOM[1], q[0])), Math.max(DOM[2], Math.min(DOM[3], q[1]))]; this.draw(ctx); });
    this.prep(ctx); this.draw(ctx);
  },
  prep(ctx) { const F2 = FUN[ctx.vals.fn]; ctx.sets = contours(F2.f, CD, F2.levels, 130); },
  onControl(ctx, id) { if (id === 'fn') this.prep(ctx); this.draw(ctx); },
  draw(ctx) {
    const F2 = FUN[ctx.vals.fn], [x, y] = ctx.p, [gx, gy] = F2.g(x, y), gn = Math.hypot(gx, gy), P = ctx.plot.clear();
    const a = ctx.vals.phi * Math.PI / 180, u = [Math.cos(a), Math.sin(a)], du = gx * u[0] + gy * u[1];
    heat(P, F2.f, CD, { alpha: .42, n: 90 });
    drawContours(P, ctx.sets);
    drawContours(P, contours(F2.f, CD, [F2.f(x, y)], 160), { color: HEX.flat, width: 2.2 });
    const R = .42;
    P.custom((g, p) => {   // ring: colour = directional derivative in that direction
      for (let k = 0; k < 72; k++) {
        const t0 = k / 72 * 2 * Math.PI, t1 = (k + 1) / 72 * 2 * Math.PI, d = (gx * Math.cos(t0 + Math.PI / 72) + gy * Math.sin(t0 + Math.PI / 72)) / (gn || 1);
        g.strokeStyle = d >= 0 ? `rgba(255,122,89,${.15 + .85 * d})` : `rgba(124,196,255,${.15 + .85 * -d})`; g.lineWidth = 7;
        g.beginPath(); g.moveTo(p.sx(x + R * Math.cos(t0)), p.sy(y + R * Math.sin(t0))); g.lineTo(p.sx(x + R * Math.cos(t1)), p.sy(y + R * Math.sin(t1))); g.stroke();
      }
    });
    const s = gn > 1e-9 ? .75 / Math.max(gn, 1.6) : 0;
    arrow(P, x, y, gx * s, gy * s, { color: HEX.ball });
    arrow(P, x, y, u[0] * .62, u[1] * .62, { color: HEX.opt2, width: 2 });
    P.scatter([[x, y]], { color: '#ffffff', r: 5 });
    P.draw();
    const phi = gn > 1e-9 ? Math.acos(Math.max(-1, Math.min(1, du / gn))) * 180 / Math.PI : NaN;
    ctx.stat(`x = (${x.toFixed(2)}, ${y.toFixed(2)}) · ∇f = (${f3(gx)}, ${f3(gy)}), ‖∇f‖ = ${gn.toFixed(3)} · angle(u, ∇f) = ${isNaN(phi) ? '—' : phi.toFixed(0) + '°'} · D_u f = ∇f·u = ${f3(du)}`);
  },
};

// ---------- tangent plane on a curved surface ----------
const fp = (x, y) => .3 * x * x + .2 * y * y + .4 * Math.sin(1.5 * x) * Math.cos(y);
const gp = (x, y) => [.6 * x + .6 * Math.cos(1.5 * x) * Math.cos(y), .4 * y - .4 * Math.sin(1.5 * x) * Math.sin(y)];
const plane = {
  title: 'The tangent plane is the first-order Taylor approximation',
  caption: 'The violet plane f(x) + ∇f(x)·Δ touches the surface at the white point. Near the point they agree; farther out the surface curves away. Move the point and grow the patch.',
  note: 'f(x, y) = 0.3x² + 0.2y² + 0.4 sin(1.5x) cos y; heights drawn to scale (×0.55).',
  legend: [[HEX.opt2, 'tangent plane'], [HEX.ball, 'gradient direction']],
  mode: '3d', cam: [2.9, 2.6, 3.6], tgt: [0, .35, 0],
  controls: [
    { id: 'x', label: 'point x', min: -1.6, max: 1.6, step: .01, value: 1.1 },
    { id: 'y', label: 'point y', min: -1.6, max: 1.6, step: .01, value: .9 },
    { id: 'r', label: 'patch half-width r', min: .1, max: 1.2, step: .01, value: .8 },
  ],
  build(ctx) {
    ctx.d = def({ L: fp, g: gp, dom: [-2, 2, -2, 2], h: v => v, fit: null, hScale: .55 });
    ctx.stage.surface(ctx.d, { opacity: .78 });
    this.patch(ctx);
  },
  onControl(ctx) { this.patch(ctx); },
  patch(ctx) {
    const d = ctx.d, st = ctx.stage, { x, y, r } = ctx.vals, f0 = fp(x, y), [gx, gy] = gp(x, y);
    (ctx.tmp || []).forEach(o => st.remove(o)); ctx.tmp = [];
    const Pl = (u, v) => f0 + gx * (u - x) + gy * (v - y);
    const Wp = (u, v, lift = 0) => new THREE.Vector3((u - d.cx) * d.sx, Pl(u, v) * d.hScale + d.hOff + lift, -(v - d.cy) * d.sy);
    const n = 16, pos = [], idx = [];
    for (let j = 0; j <= n; j++) for (let i = 0; i <= n; i++) { const v = Wp(x - r + 2 * r * i / n, y - r + 2 * r * j / n); pos.push(v.x, v.y, v.z); }
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const a = j * (n + 1) + i; idx.push(a, a + n + 1, a + 1, a + 1, a + n + 1, a + n + 2); }
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
    ctx.tmp.push(st.add(new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: C.opt2, transparent: true, opacity: .55, side: THREE.DoubleSide, depthWrite: false }))));
    const edge = [[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]].map(([a, b]) => Wp(x + a * r, y + b * r));
    ctx.tmp.push(st.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(edge), new THREE.LineBasicMaterial({ color: C.opt2 }))));
    ctx.tmp.push(st.dot(d.W(x, y, .02), C.white, .055));
    const gn = Math.hypot(gx, gy), ux = gn ? gx / gn : 1, uy = gn ? gy / gn : 0;
    ctx.tmp.push(st.segment(Wp(x, y, .03), Wp(x + ux * r, y + uy * r, .03), C.ball));
    ctx.tmp.push(st.dot(Wp(x + ux * r, y + uy * r, .03), C.ball, .04));
    const true_ = fp(x + ux * r, y + uy * r), lin = Pl(x + ux * r, y + uy * r);
    ctx.stat(`f(x) = ${f0.toFixed(3)}, ∇f(x) = (${f3(gx)}, ${f3(gy)}) · a step of length r = ${r.toFixed(2)} uphill: plane predicts ${lin.toFixed(3)}, surface has ${true_.toFixed(3)} (error ${Math.abs(true_ - lin).toFixed(3)})`);
  },
};

// ---------- the Jacobian: a curved map looks like a matrix up close ----------
const G = (x, y) => [x + .5 * Math.sin(y), y + .5 * Math.sin(x)];
const JG = (x, y) => [[1, .5 * Math.cos(y)], [.5 * Math.cos(x), 1]];
const jacobian = {
  title: 'Up close, a curved map is a matrix',
  caption: 'The blue grid is the plane after the map g bends it. The coral shape is the exact image of a small square at the point; the violet dashed shape is what the Jacobian J predicts. Shrink the square: they merge.',
  note: 'g(x, y) = (x + 0.5 sin y, y + 0.5 sin x). det J = 1 − 0.25 cos x cos y is the local area scale.',
  legend: [[HEX.pos, 'image of the grid'], [HEX.ball, 'exact image of the square'], [HEX.opt2, 'J × square']],
  mode: '2d',
  plot: { xlabel: 'g₁', ylabel: 'g₂', xlim: [-3, 3], ylim: [-2.6, 2.6], equal: true, legend: false, grid: false },
  controls: [
    { id: 'x', label: 'point x', min: -2, max: 2, step: .01, value: .9 },
    { id: 'y', label: 'point y', min: -2, max: 2, step: .01, value: -.6 },
    { id: 's', label: 'square size', min: .05, max: 1.4, step: .01, value: .9 },
  ],
  build(ctx) { this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { x, y, s } = ctx.vals, P = ctx.plot.clear(), h = s / 2;
    for (let k = -3; k <= 3; k += .5) {
      const a = [], b = [];
      for (let t = -3.2; t <= 3.2 + 1e-9; t += .05) { a.push(G(k, t)); b.push(G(t, k)); }
      P.line(a, { color: 'rgba(124,196,255,.38)', width: 1 }); P.line(b, { color: 'rgba(124,196,255,.38)', width: 1 });
    }
    const sq = []; const add = (u, v) => sq.push(G(u, v));
    for (let t = 0; t <= 1; t += .02) add(x - h + 2 * h * t, y - h);
    for (let t = 0; t <= 1; t += .02) add(x + h, y - h + 2 * h * t);
    for (let t = 0; t <= 1; t += .02) add(x + h - 2 * h * t, y + h);
    for (let t = 0; t <= 1; t += .02) add(x - h, y + h - 2 * h * t);
    P.line(sq, { color: HEX.ball, width: 2.4 });
    const J = JG(x, y), g0 = G(x, y), lin = (a, b) => [g0[0] + J[0][0] * a + J[0][1] * b, g0[1] + J[1][0] * a + J[1][1] * b];
    P.line([lin(-h, -h), lin(h, -h), lin(h, h), lin(-h, h), lin(-h, -h)], { color: HEX.opt2, width: 2, dash: [6, 4] });
    P.scatter([g0], { color: '#ffffff', r: 4.5 });
    P.draw();
    const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    let err = 0; for (const [a, b] of [[-h, -h], [h, -h], [h, h], [-h, h]]) { const e = G(x + a, y + b), l = lin(a, b); err = Math.max(err, Math.hypot(e[0] - l[0], e[1] - l[1])); }
    ctx.stat(`J = [[${J[0][0].toFixed(2)}, ${J[0][1].toFixed(3)}], [${J[1][0].toFixed(3)}, ${J[1][1].toFixed(2)}]] · det J = ${det.toFixed(3)} · largest corner gap ${err.toFixed(4)} (shrinks like size²)`);
  },
};

export default { secant, compass, plane, jacobian };
