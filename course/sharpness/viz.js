// Stop 10 visuals. Numbers checked in scripts/check-math/course_sharpness.py.
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { contours, drawContours, draggable } from '../../src/learn/plot2d.js';

const eig2max = (a, b, c) => (a + c) / 2 + Math.hypot((a - c) / 2, b);

// ---------- two equally deep wells, one sharp and one flat ----------
const SX = -.9, SY = .35, FX = .8, FY = -.2, SS = .22, SF = .75, AS = 1, AF = 1.045, Q = .06;
const wL = (x, y) => Q * (x * x + y * y) - AS * Math.exp(-((x - SX) ** 2 + (y - SY) ** 2) / (2 * SS * SS)) - AF * Math.exp(-((x - FX) ** 2 + (y - FY) ** 2) / (2 * SF * SF));
const wG = (x, y) => {
  const es = AS * Math.exp(-((x - SX) ** 2 + (y - SY) ** 2) / (2 * SS * SS)), ef = AF * Math.exp(-((x - FX) ** 2 + (y - FY) ** 2) / (2 * SF * SF));
  return [2 * Q * x + es * (x - SX) / (SS * SS) + ef * (x - FX) / (SF * SF), 2 * Q * y + es * (y - SY) / (SS * SS) + ef * (y - FY) / (SF * SF)];
};
const MS = [-0.885506, 0.344972], MF = [0.751359, -0.187840], S_SHARP = 20.845, S_FLAT = 1.974;   // Newton-refined in the check script
const near = (p, m) => Math.hypot(p[0] - m[0], p[1] - m[1]) < 1e-3;
const STARTS = []; for (let j = 0; j < 17; j++) for (let i = 0; i < 21; i++) STARTS.push([-2 + 4 * i / 20, -1.5 + 3 * j / 16]);
function fates(eta) {
  const out = [];
  for (const s of STARTS) {
    let x = s[0], y = s[1];
    for (let t = 0; t < 4000; t++) { const g = wG(x, y); x -= eta * g[0]; y -= eta * g[1]; if (!isFinite(x) || Math.abs(x) > 1e6) break; }
    out.push(near([x, y], MS) ? 0 : near([x, y], MF) ? 1 : 2);
  }
  return out;
}
const wells = {
  title: 'A big step cannot stay in a sharp minimum',
  caption: 'Two minima of the same depth: a sharp one (sharpness 20.8, left) and a flat one (1.97, right). The dots are 357 starting points coloured by where gradient descent ends from each. Raise η past 2/20.8 ≈ 0.096 and the sharp well loses every one of them.',
  note: 'f(x, y) = 0.06(x² + y²) minus two Gaussian wells of widths 0.22 and 0.75. Each start runs 4000 GD steps.',
  legend: [[HEX.opt2, 'ends in the sharp well'], [HEX.good, 'ends in the flat well'], ['#f2f4ff', 'never settles'], [HEX.ball, 'one GD run']],
  mode: '3d', cam: [0, 3.6, 2.75], tgt: [0, .95, .1],
  controls: [{ id: 'eta', label: 'step size η', min: .02, max: 1.1, step: .005, value: .05, digits: 3, live: false }],
  action: 'Drop the ball',
  build(ctx) {
    const d = ctx.d = def({ L: wL, g: wG, h: v => v, dom: [-2, 2, -1.5, 1.5], N: 150, fit: 1.6 });
    ctx.stage.surface(d);
    // the ball lives on a much larger box (no clamping of its GD steps); it is drawn with the surface's scales
    ctx.bd = Object.assign(def({ L: wL, g: wG, h: v => v, dom: [-50, 50, -50, 50] }), { cx: d.cx, cy: d.cy, sx: d.sx, sy: d.sy, hScale: d.hScale, hOff: d.hOff, fit: 0 });
    ctx.stage.label('sharp', d.W(MS[0], MS[1], .55), '', HEX.opt2);
    ctx.stage.label('flat', d.W(MF[0], MF[1], .45), '', HEX.good);
    this.drop(ctx);
  },
  drop(ctx) {
    const eta = ctx.vals.eta, st = ctx.stage;
    st.clearBalls();
    if (ctx.dots) st.remove(ctx.dots);
    ctx.f = fates(eta);
    const col = [[.70, .55, 1], [.37, .88, .69], [.95, .96, 1]];
    ctx.dots = st.points(STARTS.map(s => { const v = ctx.d.W(s[0], s[1], .03); return [v.x, v.y, v.z]; }), ctx.f.map(k => col[k]), .1);
    ctx.ball = st.ball(ctx.bd, { start: [-1.25, .75], step: opt.gd(eta), speed: eta < .2 ? 6 : 2, maxSteps: 4000, stop: b => Math.hypot(...wG(...b.p)) < 1e-7 || Math.max(Math.abs(b.p[0]), Math.abs(b.p[1])) >= 50 });
    this.show(ctx);
  },
  show(ctx) {
    const eta = ctx.vals.eta, n = [0, 0, 0]; ctx.f.forEach(k => n[k]++);
    const b = ctx.ball, out = b && (Math.abs(b.p[0]) > 2 || Math.abs(b.p[1]) > 1.5), where = !b ? '' : near(b.p, MS) ? 'settled in the sharp well' : near(b.p, MF) ? 'settled in the flat well' : b.done ? 'did not settle' : out ? 'outside the drawn area' : `f = ${wL(...b.p).toFixed(3)}`;
    ctx.stat(`η = ${eta.toFixed(3)} · 2/η = ${(2 / eta).toFixed(2)} vs sharpness ${S_SHARP.toFixed(1)} (sharp) and ${S_FLAT.toFixed(2)} (flat) · of 357 starts: ${n[0]} sharp, ${n[1]} flat, ${n[2]} never settle · ball: step ${b ? b.steps : 0}, ${where}`);
  },
  frame(ctx) { if (ctx.ball && ctx.ball.steps !== ctx.last) { ctx.last = ctx.ball.steps; this.show(ctx); } },
  onControl(ctx) { this.drop(ctx); },
};

// ---------- a valley of equivalent minima: f = 1/2 (uv - 1)^2 ----------
const uvL = (u, v) => .5 * (u * v - 1) ** 2, uvG = (u, v) => [(u * v - 1) * v, (u * v - 1) * u];
const uvS = (u, v) => eig2max(v * v, 2 * u * v - 1, u * u);
const HYP = []; for (let u = 1 / 2.2; u <= 3.0001; u += .01) HYP.push([u, 1 / u]);
const family = {
  title: 'A whole valley of minima, and the step size picks one',
  caption: 'Every point of the green curve uv = 1 has loss 0, but its sharpness u² + v² grows from 2 at (1, 1) to large values along the arms. GD starts at (2.5, 0.05), next to the sharp end. Small steps stop at the nearest point of the curve; larger steps bounce off the sharp part and settle where the valley is flatter.',
  note: 'f(u, v) = ½(uv − 1)². Height drawn on a log scale. The chart shows the largest Hessian eigenvalue at each iterate.',
  legend: [[HEX.good, 'minima uv = 1'], [HEX.ball, 'gradient descent'], [HEX.flat, '2/η']],
  mode: '3d', cam: [.7, 4.5, 2.9], tgt: [.1, .2, -.1],
  controls: [{ id: 'eta', label: 'step size η', min: .02, max: .46, step: .005, value: .05, digits: 3 }],
  action: 'Drop the ball',
  charts: [{ id: 's', title: 'sharpness λ_max at x_t', xlabel: 'step t', ylabel: 'λ_max', xlim: [0, 120], ylim: [0, 10], h: 190, legend: false }],
  build(ctx) {
    const d = ctx.d = def({ L: uvL, g: uvG, dom: [-1, 3, -1, 2.2], N: 130 });
    ctx.stage.surface(d);
    ctx.stage.tube(d, HYP, C.good, .02, .03);
    ctx.stage.marker(d, 1, 1, C.good, .1);
    ctx.stage.label('(1, 1): sharpness 2', d.W(1, 1, .35), '', HEX.good);
    ctx.stage.label('u →', d.W(2.6, -1, .15), '', HEX.muted);
    ctx.stage.label('v →', d.W(-1, 1.8, .15), '', HEX.muted);
    this.drop(ctx);
  },
  drop(ctx) {
    ctx.stage.clearBalls(); ctx.hist = [uvS(2.5, .05)];
    ctx.ball = ctx.stage.ball(ctx.d, { start: [2.5, .05], step: opt.gd(ctx.vals.eta), speed: .5, maxSteps: 4000, stop: b => uvL(...b.p) < 1e-14 });
    ctx.ball.onStep = b => { if (ctx.hist.length < 400) ctx.hist.push(uvS(...b.p)); };
    this.chart(ctx);
  },
  chart(ctx) {
    const P = ctx.charts.s.clear(), eta = ctx.vals.eta;
    P.hline(2 / eta, { color: HEX.flat, dash: [5, 4], text: `2/η = ${(2 / eta).toFixed(2)}` });
    P.line(ctx.hist.map((s, t) => [t, s]), { color: HEX.ball, width: 2, dots: 1.6 });
    P.draw();
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    if (b.steps !== ctx.last) { ctx.last = b.steps; this.chart(ctx); }
    const eta = ctx.vals.eta, [u, v] = b.p, S = u * u + v * v;
    ctx.stat(b.done && uvL(u, v) < 1e-14
      ? `η = ${eta.toFixed(3)} · settled at (${u.toFixed(3)}, ${v.toFixed(3)}) after ${b.steps} steps · its sharpness u² + v² = ${S.toFixed(2)} < 2/η = ${(2 / eta).toFixed(2)}`
      : `η = ${eta.toFixed(3)} · step ${b.steps} · loss ${uvL(u, v).toExponential(2)} · sharpness here ${uvS(u, v).toFixed(2)} · 2/η = ${(2 / eta).toFixed(2)}`);
  },
  onControl(ctx) { this.drop(ctx); },
};

// ---------- edge of stability: the Cohen-page MLP run ----------
const ETAS = ['0.1', '0.3', '0.36', '0.42'], ECOL = { '0.1': HEX.pos, '0.3': HEX.ball, '0.36': HEX.opt2, '0.42': HEX.good };
const eos = {
  title: 'Edge of stability in a real network',
  caption: 'Full-batch GD on a small tanh network, with the sharpness measured every 10 steps. With η = 0.1 the sharpness rises but stays far below 2/η = 20. With the larger steps it rises until it reaches 2/η (dashed) and then stays there, while the loss keeps falling with small bumps.',
  note: 'tanh MLP 10–48–48–1, 128 synthetic points, squared loss; sharpness by power iteration. Data from the Atlas page of Cohen et al. 2021 (our replication, not the paper\'s runs).',
  legend: [[HEX.pos, 'η = 0.1'], [HEX.ball, 'η = 0.3'], [HEX.opt2, 'η = 0.36'], [HEX.good, 'η = 0.42']],
  mode: '2d',
  plot: { xlabel: 'step t', ylabel: 'sharpness λ_max', xlim: [0, 3000], ylim: [3, 9], legend: false },
  controls: [{ id: 'eta', label: 'highlight η', type: 'seg', value: '0.3', options: ETAS.map(e => [e, e]) }],
  charts: [{ id: 'loss', title: 'training loss', xlabel: 'step t', ylabel: 'loss', logy: true, xlim: [0, 3000], ylim: [1e-7, 1], h: 180, legend: false }],
  build(ctx) {
    ctx.stat('loading the recorded run…');
    fetch('papers/cohen2021gradient/data.json').then(r => r.json()).then(D => { ctx.D = D; this.draw(ctx); }).catch(err => console.error('[course] eos data', err));
  },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const D = ctx.D; if (!D) return;
    const sel = ctx.vals.eta, P = ctx.plot.clear(), Lp = ctx.charts.loss.clear();
    for (const e of ETAS) {
      const R = D.curves[e], on = e === sel;
      P.line(R.map(r => [r[0], r[2]]), { color: ECOL[e], width: on ? 2.4 : 1.1, alpha: on ? 1 : .3 });
      Lp.line(R.map(r => [r[0], r[1]]), { color: ECOL[e], width: on ? 2 : 1, alpha: on ? 1 : .3 });
    }
    const e = +sel;
    if (2 / e < 9) P.hline(2 / e, { color: ECOL[sel], dash: [6, 4], text: `2/η = ${(2 / e).toFixed(2)}` });
    else P.text(1500, 8.6, `2/η = 20 is far above this chart`, { color: ECOL[sel], align: 'center' });
    P.draw(); Lp.draw();
    const R = D.curves[sel], late = R.slice(-100).reduce((a, r) => a + r[2], 0) / 100, peak = Math.max(...R.map(r => r[2]));
    let ups = 0; for (let i = 1; i < R.length; i++) if (R[i][1] > R[i - 1][1]) ups++;
    ctx.stat(`η = ${sel} · 2/η = ${(2 / e).toFixed(2)} · sharpness: start ${R[0][2].toFixed(2)}, peak ${peak.toFixed(2)}, average of the last 1000 steps ${late.toFixed(2)} · loss rose in ${ups} of 300 ten-step windows, final loss ${R[R.length - 1][1].toExponential(1)}`);
  },
};

// ---------- self-stabilization: Damian et al.'s toy (App. B), eta = 1 ----------
const ETA_D = 1;
const stabilize = {
  title: 'Why the oscillation does not blow up',
  caption: 'A trough whose curvature across (x) grows as we go downhill along y: S = 2/η + y. GD slides down, the trough sharpens past 2/η, the ball starts bouncing across it, and the bouncing itself pushes it back up the slope to where S < 2/η. The cycle repeats.',
  note: 'f(x, y) = (2/η + y) x²/2 − α y with η = 1 (Damian et al. 2023, App. B, restricted to two coordinates). The downhill tilt α is tiny, so the surface looks level along y; the white line is S = 2/η.',
  legend: [[HEX.ball, 'gradient descent'], [HEX.white, 'S = 2/η'], [HEX.flat, 'S_t − 2/η'], [HEX.opt2, '|x_t|']],
  mode: '3d', cam: [3.4, 2.6, 3.3], tgt: [0, .3, 0],
  controls: [{ id: 'al', label: 'sharpening α', min: .001, max: .003, step: .0001, value: .002, digits: 4, live: false }],
  action: 'Replay',
  charts: [{ id: 'c', title: 'S_t − 2/η and |x_t|', xlabel: 'step t', ylabel: '', xlim: [0, 600], ylim: [-.25, .32], h: 190, legend: false }],
  build(ctx) {
    this.make(ctx);
  },
  make(ctx) {
    const al = ctx.vals.al, st = ctx.stage;
    st.clear();
    const d = ctx.d = def({ L: (x, y) => (2 / ETA_D + y) * x * x / 2 - al * y, g: (x, y) => [(2 / ETA_D + y) * x, x * x / 2 - al], h: v => v, dom: [-.45, .45, -.35, .3], square: true, fit: 1.4, N: 90 });
    st.surface(d);
    st.line(d, Array.from({ length: 41 }, (_, i) => [-.45 + .9 * i / 40, 0]), C.white, .9, .02);
    st.label('S = 2/η', d.W(.45, 0, .25), '', HEX.white);
    st.label('downhill: y grows, S grows →', d.W(.25, .24, .12), '', HEX.muted);
    ctx.hist = [[.02, -.05]];
    ctx.ball = st.ball(d, { start: [.02, -.05], step: opt.gd(ETA_D), speed: .6, maxSteps: 600, r: .045 });
    ctx.ball.onStep = b => ctx.hist.push(b.p.slice());
    this.chart(ctx);
  },
  drop(ctx) { this.make(ctx); },
  onControl(ctx) { this.make(ctx); },
  chart(ctx) {
    const P = ctx.charts.c.clear(), dl = Math.sqrt(2 * ctx.vals.al);
    P.hline(0, { color: 'rgba(255,255,255,.5)' });
    P.hline(dl, { color: HEX.opt2, dash: [3, 4], text: `δ = √(2α) = ${dl.toFixed(3)}` });
    P.line(ctx.hist.map((p, t) => [t, p[1]]), { color: HEX.flat, width: 2 });
    P.line(ctx.hist.map((p, t) => [t, Math.abs(p[0])]), { color: HEX.opt2, width: 1.4 });
    P.draw();
  },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    if (b.steps !== ctx.last) { ctx.last = b.steps; this.chart(ctx); }
    const [x, y] = b.p, dl = Math.sqrt(2 * ctx.vals.al), big = Math.abs(x) > dl;
    const stage = y < 0 ? (big ? '4 · back below the edge: |x| shrinks' : '1 · progressive sharpening: S rises toward 2/η') : (big ? '3 · self-stabilization: the bouncing lowers S' : '2 · past the edge: the bouncing grows');
    ctx.stat(`step ${b.steps} · S_t = 2/η + y = ${(2 + y).toFixed(3)} · |x_t| = ${Math.abs(x).toFixed(3)} · stage ${stage}`);
  },
};

// ---------- SAM on the valley of minima ----------
const SAM_ETA = .3, LV = [.005, .02, .06, .15, .3, .6, 1.2, 2.4, 5];
const samEdge = rho => rho < 1e-9 ? 2 / SAM_ETA : (-1 + Math.sqrt(1 + 8 * rho / SAM_ETA)) / (2 * rho);
function runUV(p0, mode, rho, T = 4000) {
  let [u, v] = p0; const path = [[u, v]], S = [];
  for (let t = 0; t < T; t++) {
    let [gu, gv] = uvG(u, v), qu = u, qv = v;
    if (mode === 'un') { qu = u + rho * gu; qv = v + rho * gv; }
    else if (mode === 'norm') { const n = Math.hypot(gu, gv); if (n > 1e-12) { qu = u + rho * gu / n; qv = v + rho * gv / n; } }
    [gu, gv] = uvG(qu, qv); u -= SAM_ETA * gu; v -= SAM_ETA * gv;
    if (!isFinite(u) || Math.abs(u) > 1e6) return { path, diverged: true };
    if (t < 600 || t % 10 === 0) path.push([u, v]);
    S.push(u * u + v * v);
  }
  return { path, end: [u, v], S, late: S.slice(-200).reduce((a, s) => a + s, 0) / 200 };
}
const sam = {
  title: 'SAM: descend from the worst nearby point',
  caption: 'Same valley as above, seen from above. GD (coral) and SAM (violet) start at the white dot with η = 0.3; drag the dot to move it. Unnormalized SAM settles below its own, lower edge ηλ(1 + ρλ) = 2. Normalized SAM keeps a fixed radius ρ, never settles, and drifts along the valley toward the flattest point (1, 1).',
  note: 'f(u, v) = ½(uv − 1)², equal axes; green: the minima uv = 1, with their sharpness u² + v² marked. 4000 steps per path.',
  legend: [[HEX.ball, 'gradient descent'], [HEX.opt2, 'SAM'], [HEX.good, 'minima uv = 1']],
  mode: '2d',
  plot: { xlabel: 'u', ylabel: 'v', xlim: [-.3, 3], ylim: [-.3, 2.4], equal: true, legend: false },
  controls: [
    { id: 'rho', label: 'SAM radius ρ', min: 0, max: .25, step: .005, value: .1, digits: 3 },
    { id: 'mode', label: 'SAM version', type: 'seg', value: 'un', options: [['un', 'unnormalized'], ['norm', 'normalized']] },
  ],
  build(ctx) {
    ctx.p0 = [2.2, .1];
    ctx.cs = contours(uvL, [-.3, 3, -.3, 2.4], LV, 110);
    draggable(ctx, p => { ctx.p0 = [Math.min(3, Math.max(-.3, p[0])), Math.min(2.4, Math.max(-.3, p[1]))]; this.draw(ctx); });
    this.draw(ctx);
  },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const { rho, mode } = ctx.vals, P = ctx.plot.clear();
    drawContours(P, ctx.cs);
    P.line(HYP.filter(p => p[1] <= 2.4), { color: HEX.good, width: 2.4 });
    for (const a of [1, 1.5, 2, 2.5]) { P.scatter([[a, 1 / a]], { color: HEX.good, r: 3 }); P.text(a, 1 / a, (a * a + 1 / (a * a)).toFixed(2), { color: HEX.good, dx: 6, dy: -9 }); }
    const g = runUV(ctx.p0, 'gd', 0), s = runUV(ctx.p0, mode, rho);
    P.line(g.path, { color: HEX.ball, width: 1.3, alpha: .85 });
    P.line(s.path, { color: HEX.opt2, width: 1.3, alpha: .85 });
    if (g.end) P.scatter([g.end], { color: HEX.ball, r: 5 });
    if (s.end) P.scatter([s.end], { color: HEX.opt2, r: 5 });
    P.scatter([ctx.p0], { color: '#ffffff', r: 5.5 });
    P.draw();
    const fmt = r => r.diverged ? 'diverged' : `${r.S[r.S.length - 1].toFixed(2)}`;
    const samTxt = s.diverged ? 'diverged' : mode === 'un' ? `${fmt(s)} (edge ηλ(1 + ρλ) = 2 at λ = ${samEdge(rho).toFixed(2)})` : `about ${s.late.toFixed(2)} on average over the last 200 steps, still moving (|uv − 1| = ${Math.abs(s.end[0] * s.end[1] - 1).toFixed(3)})`;
    ctx.stat(`start (${ctx.p0[0].toFixed(2)}, ${ctx.p0[1].toFixed(2)}) · η = 0.3, 2/η = 6.67 · final sharpness: GD ${fmt(g)} · ${mode === 'un' ? 'unnormalized' : 'normalized'} SAM, ρ = ${rho.toFixed(3)}: ${samTxt}`);
  },
};

export default { wells, family, eos, stabilize, sam };
