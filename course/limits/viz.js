// Stop 12 visuals. Numbers checked in scripts/check-math/course_limits.py (same seeded RNG as src/lib/math.js).
import { def, opt, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

// ---------- Carmon et al.'s chain f_T (Eq. 10) ----------
// erf to ~1e-14 (Taylor series below 3, continued fraction for erfc above), so the paths match the check script's math.erf
function erf(x) {
  const a = Math.abs(x), s = Math.sign(x);
  if (a < 3) { let term = a, sum = a; for (let n = 1; n < 120; n++) { term *= -a * a / n; const t = term / (2 * n + 1); sum += t; if (Math.abs(t) < 1e-17 * Math.abs(sum)) break; } return s * 2 / Math.sqrt(Math.PI) * sum; }
  let f = a; for (let n = 80; n >= 1; n--) f = a + (n / 2) / f;
  return s * (1 - Math.exp(-a * a) / Math.sqrt(Math.PI) / f);
}
const Psi = t => t <= .5 ? 0 : Math.exp(1 - 1 / (2 * t - 1) ** 2);
const dPsi = t => t <= .5 ? 0 : Psi(t) * 4 / (2 * t - 1) ** 3;
const Phi = t => Math.sqrt(Math.E) * Math.sqrt(2 * Math.PI) * .5 * (1 + erf(t / Math.SQRT2));
const dPhi = t => Math.sqrt(Math.E) * Math.exp(-t * t / 2);
function gT(x) {
  const T = x.length, G = new Array(T).fill(0);
  G[0] = -Psi(1) * dPhi(x[0]);
  for (let i = 1; i < T; i++) {
    G[i] += -Psi(-x[i - 1]) * dPhi(-x[i]) - Psi(x[i - 1]) * dPhi(x[i]);
    G[i - 1] += -dPsi(-x[i - 1]) * Phi(-x[i]) - dPsi(x[i - 1]) * Phi(x[i]);
  }
  return G;
}
const nrm = v => Math.hypot(...v);
const f2 = (a, b) => -Psi(1) * Phi(a) + Psi(-a) * Phi(-b) - Psi(a) * Phi(b);
const g2 = (a, b) => gT([a, b]);

const chain2 = {
  title: 'A function that hides its next coordinate',
  caption: 'Carmon et al.\'s hard function in two coordinates. Gradient descent starts at 0. As long as x₁ ≤ ½ (left of the amber line), the gradient has no x₂ component at all, so x₂ stays exactly 0: the algorithm cannot even see that x₂ matters. Only after x₁ passes ½ does x₂ start to move, and the gradient stays above 1 until both coordinates have moved far enough.',
  note: 'f̄₂(x) = −Ψ(1)Φ(x₁) + Ψ(−x₁)Φ(−x₂) − Ψ(x₁)Φ(x₂) (Eq. 10 with T = 2). The ball stops when ‖∇f̄₂‖ ≤ 1.',
  legend: [[HEX.ball, 'gradient descent from 0'], [HEX.flat, 'x₁ = ½']],
  mode: '3d', cam: [-1.2, 3.9, 3.6], tgt: [.1, .75, 0],
  controls: [{ id: 'eta', label: 'step size η', min: .01, max: .1, step: .005, value: .05, digits: 3 }],
  action: 'Drop the ball',
  build(ctx) {
    const d = ctx.d = def({ L: f2, g: g2, h: v => v, dom: [-.6, 3, -.6, 3], N: 120, fit: 1.7 });
    ctx.stage.surface(d);
    ctx.stage.line(d, segPts(.5, -.6, .5, 3), C.flat, .95, .03);
    ctx.stage.label('x₁ →', d.W(2.7, -.6, .2), '', HEX.muted); ctx.stage.label('x₂ →', d.W(-.6, 2.6, .2), '', HEX.muted);
    this.drop(ctx);
  },
  drop(ctx) {
    ctx.stage.clearBalls(); ctx.first = null;
    ctx.ball = ctx.stage.ball(ctx.d, { start: [0, 0], step: opt.gd(ctx.vals.eta), speed: .25, maxSteps: 3000, stop: b => nrm(g2(...b.p)) <= 1 });
    ctx.ball.onStep = b => { if (ctx.first === null && b.p[1] !== 0) ctx.first = { k: b.steps, x1: ctx.prev }; ctx.prev = b.p[0]; };
    ctx.prev = 0;
  },
  onControl(ctx) { this.drop(ctx); },
  frame(ctx) {
    const b = ctx.ball; if (!b) return;
    const g = nrm(g2(...b.p)), f = ctx.first;
    ctx.stat(`η = ${ctx.vals.eta.toFixed(3)} · step ${b.steps} · x = (${b.p[0].toFixed(3)}, ${b.p[1].toFixed(3)}) · ‖∇f̄₂‖ = ${g.toFixed(3)}${g <= 1 ? ' ≤ 1: target reached' : ' > 1'} · ${f ? `x₂ first moved at step ${f.k}, when x₁ = ${f.x1.toFixed(3)} > ½` : 'x₂ still exactly 0'}`);
  },
};

// ---------- the chain in T coordinates: exact GD vs the probability-p oracle (Arjevani et al. Eq. 17) ----------
const prog = (x, a) => { for (let i = x.length - 1; i >= 0; i--) if (Math.abs(x[i]) > a) return i + 1; return 0; };
function chainRun(T, p, eta = .05, kmax = 20000) {
  const r = rng(1000 * T + Math.round(100 * p)); let x = new Array(T).fill(0), disc = null;
  const hist = [x.slice()], found = [0];
  for (let k = 0; k < kmax; k++) {
    if (disc === null && prog(x, 0) >= T) disc = k;
    if (disc !== null && nrm(gT(x)) <= 1) return { disc, hit: k, hist, found };
    const G = gT(x);
    if (p < 1) { const z = r() < p ? 1 : 0, j = prog(x, .25); for (let i = j; i < T; i++) G[i] *= z / p; }
    x = x.map((v, i) => v - eta * G[i]);
    hist.push(x.slice()); found.push(prog(x, 0));
  }
  return { disc, hit: null, hist, found };
}
const PS = [1, .2, .05], PCOL = { 1: HEX.ball, .2: HEX.opt2, .05: HEX.pos };
const chain = {
  title: 'One coordinate per query: why the bound grows with T',
  caption: 'The same construction with T coordinates. Each bar is one |xᵢ|; a coordinate stays exactly 0 until its left neighbour has grown past ½. With exact gradients (p = 1) GD uncovers them one after another. With the noisy oracle of Arjevani et al., the next coordinate\'s gradient is revealed only with probability p (and scaled by 1/p, so the oracle stays unbiased), and the queries grow like T/p.',
  note: 'GD / SGD with step 0.05 from 0. Chart: coordinates uncovered vs queries for the three oracles at this T. One seeded run per (T, p).',
  legend: [[HEX.ball, 'p = 1 (exact)'], [HEX.opt2, 'p = 0.2'], [HEX.pos, 'p = 0.05'], [HEX.flat, '½']],
  mode: '2d',
  plot: { xlabel: 'coordinate i', ylabel: '|xᵢ|', xlim: [.4, 16.6], ylim: [0, 3.2], legend: false },
  controls: [
    { id: 'T', label: 'number of coordinates T', min: 2, max: 16, step: 1, value: 8, digits: 0 },
    { id: 'p', label: 'oracle', type: 'seg', value: 1, options: [[1, 'exact'], [.2, 'p = 0.2'], [.05, 'p = 0.05']] },
  ],
  actions: [{ label: 'Replay', fn: ctx => { ctx.k = 0; } }],
  charts: [{ id: 'q', title: 'coordinates uncovered', xlabel: 'queries', ylabel: 'uncovered', xlim: [0, 600], ylim: [0, 16.5], h: 180, legend: false }],
  build(ctx) { this.run(ctx); },
  onControl(ctx) { this.run(ctx); },
  run(ctx) { const T = ctx.vals.T; ctx.runs = Object.fromEntries(PS.map(p => [p, chainRun(T, p)])); ctx.k = 0; ctx.acc = 0; this.draw(ctx); },
  frame(ctx, dt) {
    const R = ctx.runs[ctx.vals.p]; if (!R || ctx.k >= R.hist.length - 1) return;
    ctx.acc += dt * 40; const n = Math.floor(ctx.acc); if (!n) return; ctx.acc -= n;
    ctx.k = Math.min(R.hist.length - 1, ctx.k + n); this.draw(ctx);
  },
  draw(ctx) {
    const T = ctx.vals.T, p = ctx.vals.p, R = ctx.runs[p], x = R.hist[ctx.k], P = ctx.plot.clear();
    P.set({ xlim: [.4, T + .6] });
    P.hline(.5, { color: HEX.flat, dash: [5, 4] });
    P.custom((g, pl) => {
      for (let i = 0; i < T; i++) {
        const X0 = pl.sx(i + 1 - .35), X1 = pl.sx(i + 1 + .35), Y0 = pl.sy(0), Y1 = pl.sy(Math.min(3.2, Math.abs(x[i])));
        g.fillStyle = x[i] === 0 ? 'rgba(142,153,189,.25)' : PCOL[p]; g.fillRect(X0, Y1, X1 - X0, Y0 - Y1);
        if (x[i] === 0) { g.fillStyle = 'rgba(142,153,189,.7)'; g.fillRect(X0, Y0 - 2, X1 - X0, 2); }
      }
    });
    P.draw();
    const Q = ctx.charts.q.clear(), xmax = Math.max(...PS.map(q => ctx.runs[q].hist.length)) + 10;
    Q.set({ xlim: [0, xmax], ylim: [0, T + .5] });
    for (const q of PS) { const r = ctx.runs[q], upto = q === p ? ctx.k + 1 : r.found.length; Q.line(r.found.slice(0, upto).map((v, k) => [k, v]), { color: PCOL[q], width: q === p ? 2.4 : 1.2, alpha: q === p ? 1 : .45, step: true }); }
    Q.draw();
    const s = q => `${ctx.runs[q].hit}`;
    ctx.stat(`T = ${T} · ${p === 1 ? 'exact gradients' : `oracle with p = ${p}`} · query ${ctx.k}: ${prog(x, 0)} of ${T} coordinates uncovered, ‖∇f̄_T‖ = ${nrm(gT(x)).toFixed(2)} · queries until ‖∇f̄_T‖ ≤ 1: exact ${s(1)}, p = 0.2: ${s(.2)}, p = 0.05: ${s(.05)}`);
  },
};

// ---------- a quartic whose Hessian at 0 is always zero ----------
const qf = q => (x, y) => x ** 4 + 2 * q * x * x * y * y + y ** 4;
const qg = q => (x, y) => [4 * x ** 3 + 4 * q * x * y * y, 4 * y ** 3 + 4 * q * x * x * y];
const quartic = {
  title: 'Is the origin a local minimum? The Hessian cannot tell',
  caption: 'f(u) = u₁⁴ + 2q u₁²u₂² + u₂⁴ has zero gradient and zero Hessian at the origin for every q, so first- and second-order tests say nothing. The answer depends on whether f is negative along some direction: along the diagonals it is (1 + q)/2 · r⁴. For q ≥ −1 the origin is a local minimum; for q < −1 the diagonals go downhill and the ball escapes.',
  note: 'f(u) = vᵀQv with v = (u₁², u₂²), Q = [[1, q], [q, 1]]: the origin is a local minimum iff Q is copositive. Deciding copositivity is NP-hard in high dimension (Murty & Kabadi 1987). GD step 0.05 from (0.5, 0.45).',
  legend: [[HEX.neg, 'diagonal: f < 0'], [HEX.pos, 'diagonal: f ≥ 0'], [HEX.ball, 'gradient descent']],
  mode: '3d', cam: [0, 4.4, 4.0], tgt: [0, .4, -.2],
  controls: [{ id: 'q', label: 'coupling q', min: -2, max: 1.5, step: .05, value: -1.3, digits: 2 }],
  action: 'Drop the ball',
  build(ctx) {
    const q = ctx.vals.q, st = ctx.stage; st.clear();
    const d = ctx.d = def({ L: qf(q), g: qg(q), h: v => v, dom: [-1, 1, -1, 1], N: 100, fit: 1.6 });
    st.surface(d);
    const col = q < -1 ? C.neg : C.pos;
    st.tube(d, segPts(-1, -1, 1, 1), col, .016); st.tube(d, segPts(-1, 1, 1, -1), col, .016);
    st.marker(d, 0, 0, C.white, .07);
    st.label('origin', d.W(0, 0, .25), '', HEX.white);
    st.ball(d, { start: [.5, .45], step: opt.gd(.05), speed: .5, maxSteps: 2000, stop: b => Math.max(Math.abs(b.p[0]), Math.abs(b.p[1])) >= 1 });
    ctx.ball = st.balls[0];
  },
  drop(ctx) { this.build(ctx); },
  onControl(ctx) { this.build(ctx); },
  frame(ctx) {
    const q = ctx.vals.q, b = ctx.ball; if (!b) return;
    const m = Math.min(1, (1 + q) / 2), psd = Math.abs(q) <= 1, cop = q >= -1;
    const out = Math.max(Math.abs(b.p[0]), Math.abs(b.p[1])) >= 1;
    ctx.stat(`q = ${q.toFixed(2)} · ∇f(0) = 0, ∇²f(0) = 0 · min of f on the unit circle = ${m.toFixed(3)} · Q PSD: ${psd ? 'yes' : 'no'}, copositive: ${cop ? 'yes' : 'no'} → origin ${cop ? 'is' : 'is not'} a local minimum · ball: ${out ? `left the square after ${b.steps} steps` : `step ${b.steps}, distance to 0 = ${Math.hypot(...b.p).toFixed(3)}`}`);
  },
};

// ---------- the complexity ladder ----------
const LAD = [
  ['exact gradients', 0, 'up', 'GD, strongly convex or PL', 'f − f* ≤ ε', 'O(κ log(1/ε))', 'stops 4 and 11', null],
  ['exact gradients', 0, 'up', 'heavy ball / Nesterov, quadratic', 'f − f* ≤ ε', 'O(√κ log(1/ε))', 'stop 7', null],
  ['exact gradients', 2, 'up', 'GD', 'ε-FOSP', 'O(ε⁻²)', 'Nesterov 2004 (stop 3)', 'nesterov2004introductory'],
  ['exact gradients', 2, 'low', 'any first-order method', 'ε-FOSP', 'Ω(ε⁻²)', 'Carmon et al. 2020', 'carmon2020lower'],
  ['exact gradients', 2, 'up', 'perturbed GD', 'ε-SOSP', 'Õ(ε⁻²)', 'Jin et al. 2017, Thm 3', 'jin2017escape'],
  ['exact gradients, ρ-Hessian-Lipschitz', 1.75, 'up', 'perturbed accelerated GD', 'ε-SOSP', 'Õ(ε^−7/4)', 'Jin, Netrapalli & Jordan 2018, Thm 3', 'jin2018accelerated'],
  ['exact Hessians', 1.5, 'up', 'cubic regularization', 'ε-SOSP', 'O(ε^−3/2)', 'Nesterov & Polyak 2006', 'nesterov2006cubic'],
  ['exact Hessians', 1.5, 'low', 'any second-order method', 'ε-FOSP', 'Ω(ε^−3/2)', 'Carmon et al. 2020', 'carmon2020lower'],
  ['stochastic gradients', 4, 'up', 'SGD', 'ε-FOSP', 'O(ε⁻⁴)', 'Ghadimi & Lan 2013, Cor. 2.2', 'ghadimi2013stochastic'],
  ['stochastic gradients', 4, 'low', 'any stochastic first-order method', 'ε-FOSP', 'Ω(ε⁻⁴)', 'Arjevani et al. 2023, Thm 1', 'arjevani2023lower'],
  ['stochastic gradients', 3.5, 'up', 'SGD with anchored averaging', 'ε-SOSP-type', 'Õ(ε^−3.5)', 'Fang, Lin & Zhang 2019', 'fang2019sharp'],
  ['stochastic gradients, mean-squared smooth', 3, 'up', 'SPIDER', 'ε-FOSP', 'O(ε⁻³)', 'Fang et al. 2018, Thm 1', 'fang2018spider'],
  ['stochastic gradients, mean-squared smooth', 3, 'low', 'any such method', 'ε-FOSP', 'Ω(ε⁻³)', 'Arjevani et al. 2023, Thm 2', 'arjevani2023lower'],
  ['stochastic gradients + Hessian-vector products', 3.5, 'up', 'stochastic cubic regularization', 'ε-SOSP', 'Õ(ε^−3.5)', 'Tripuraneni et al. 2018', 'tripuraneni2018stochastic'],
  ['stochastic gradients + Hessian-vector products', 3.25, 'up', 'Natasha2', '(ε, ε^¼) local min', 'Õ(ε^−3.25)', 'Allen-Zhu 2018, Cor. 4.6', 'allenzhu2018natasha'],
  ['stochastic gradients + Hessian-vector products', 3, 'up', 'SGD + HVP-RVR', 'ε-FOSP', 'O(ε⁻³)', 'Arjevani et al. 2020', 'arjevani2020second'],
  ['stochastic gradients + Hessian-vector products', 3, 'low', 'any stochastic second-order method', 'ε-FOSP', 'Ω(ε⁻³)', 'Arjevani et al. 2020', 'arjevani2020second'],
];
const NL = LAD.length;
// vertical slots from the top: one per group header, one per row
const SLOT = []; { let y = 0, last = null; LAD.forEach(r => { if (r[0] !== last) { last = r[0]; y += .9; } SLOT.push(y); y += 1; }); SLOT.total = y; }
const ladder = {
  title: 'The complexity ladder',
  caption: 'The main rates of the course as bars of length a in ε^−a (oracle calls to reach the target, constants and log factors dropped). Coral bars are upper bounds (a method that achieves it), violet outlined bars are lower bounds (no method of that class can do better). Where a coral and a violet bar of the same setting have the same length, the method is optimal.',
  note: 'Bars at a = 0 are logarithmic rates (κ = condition number). Slide to highlight a row; the readout gives the paper, target and oracle, and the count at ε = 10^−k.',
  legend: [[HEX.ball, 'upper bound (a method)'], [HEX.opt2, 'lower bound (all methods)']],
  mode: '2d',
  plot: { xlabel: 'exponent a in ε^−a', ylabel: '', xlim: [-.1, 4.6], ylim: [-SLOT.total + .3, .2], legend: false, grid: true, yfmt: () => '', pad: [12, 14, 40, 14] },
  controls: [
    { id: 'i', label: 'highlight row', min: 1, max: NL, step: 1, value: 3, digits: 0 },
    { id: 'k', label: 'ε = 10^−k, k', min: 1, max: 6, step: 1, value: 3, digits: 0 },
  ],
  build(ctx) { ctx.plot.c.parentElement.style.height = '600px'; this.draw(ctx); },
  onControl(ctx) { this.draw(ctx); },
  draw(ctx) {
    const sel = ctx.vals.i - 1, k = ctx.vals.k, P = ctx.plot.clear();
    P.custom((g, pl) => {
      let lastG = null;
      LAD.forEach((r, i) => {
        const Y = pl.sy(-SLOT[i]), h = Math.max(6, (pl.sy(0) - pl.sy(1)) * .6), on = i === sel;
        if (r[0] !== lastG) { lastG = r[0]; g.fillStyle = 'rgba(142,153,189,.9)'; g.font = '600 10.5px "JetBrains Mono", monospace'; g.textAlign = 'left'; g.textBaseline = 'middle'; g.fillText(r[0], pl.sx(-.08), pl.sy(-SLOT[i] + .85)); }
        const X0 = pl.sx(0), X1 = pl.sx(Math.max(r[1], .06)), col = r[2] === 'up' ? HEX.ball : HEX.opt2;
        g.globalAlpha = on ? 1 : .55;
        if (r[2] === 'up') { g.fillStyle = col; g.fillRect(X0, Y - h / 2, X1 - X0, h); }
        else { g.strokeStyle = col; g.lineWidth = 2; g.strokeRect(X0, Y - h / 2, X1 - X0, h); }
        g.globalAlpha = 1;
        g.fillStyle = on ? '#ffffff' : 'rgba(232,236,248,.8)'; g.font = `${on ? '600 ' : ''}11px "JetBrains Mono", monospace`; g.textBaseline = 'middle';
        const lab = pl.box[2] < 520 ? r[5] : `${r[5]} ${r[3]}`, right = r[1] < 2.6;
        g.textAlign = right ? 'left' : 'right'; g.fillText(lab, right ? X1 + 6 : X1 - 6, Y);
        if (on) { g.strokeStyle = '#ffffff'; g.lineWidth = 1; g.strokeRect(X0 - 2, Y - h / 2 - 2, X1 - X0 + 4, h + 4); }
      });
    });
    P.draw();
    const r = LAD[sel], cnt = r[1] === 0 ? null : 10 ** (k * r[1]);
    const cntTxt = cnt === null ? `logarithmic: ${r[5].includes('√') ? 'about √κ' : 'about κ'} × ${(k * Math.LN10).toFixed(1)} for ε = 10^−${k}` : `ε^−${r[1]} = ${cnt >= 1e4 ? cnt.toExponential(1).replace('e+', ' × 10^') : Math.round(cnt).toLocaleString('en')} at ε = 10^−${k}`;
    ctx.stat(`row ${sel + 1}: ${r[2] === 'up' ? 'upper bound' : 'lower bound'} · ${r[3]} · oracle: ${r[0]} · target: ${r[4]} · ${r[5]} · ${r[6]} · ${cntTxt}`);
  },
};

export default { chain2, chain, quartic, ladder };
