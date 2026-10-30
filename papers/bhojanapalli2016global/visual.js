// Bhojanapalli, Neyshabur, Srebro 2016 — no spurious local minima in factorized low-rank matrix sensing under RIP.
// n = 2, r = 1 so the factor u ∈ R² can be drawn. Offline sweep in scripts/check-math/bhojanapalli2016global.py (data.json).
import { def, opt, C, HEX } from '../../src/lib/atlas3d.js';
import { rng } from '../../src/lib/math.js';

const US = [1, .5], XS = [US[0] * US[0], US[0] * US[1], US[1] * US[1]];   // X* = u* u*ᵀ as (x11, x12, x22)
/** m symmetric Gaussian sensing matrices A = (G + Gᵀ)/2 stored as (a11, a12, a22); m = 0 means fully observed. */
function sensing(m, seed) {
  const R = rng(seed), A = Array.from({ length: m }, () => { const g = [R.normal(), R.normal(), R.normal(), R.normal()]; return [g[0], (g[1] + g[2]) / 2, g[3]]; });
  const ip = (a, x) => a[0] * x[0] + 2 * a[1] * x[1] + a[2] * x[2], y = A.map(a => ip(a, XS));
  if (!m) {
    const L = (p, q) => (p * p - XS[0]) ** 2 + 2 * (p * q - XS[1]) ** 2 + (q * q - XS[2]) ** 2;
    const g = (p, q) => [4 * (p * p - XS[0]) * p + 4 * (p * q - XS[1]) * q, 4 * (p * q - XS[1]) * p + 4 * (q * q - XS[2]) * q];
    return { L, g };
  }
  const L = (p, q) => { let s = 0; for (let i = 0; i < m; i++) s += (ip(A[i], [p * p, p * q, q * q]) - y[i]) ** 2; return s / m; };
  const g = (p, q) => { let gx = 0, gy = 0; for (let i = 0; i < m; i++) { const a = A[i], r = ip(a, [p * p, p * q, q * q]) - y[i]; gx += r * (a[0] * p + a[1] * q); gy += r * (a[1] * p + a[2] * q); } return [4 * gx / m, 4 * gy / m]; };
  return { L, g };
}
const recErr = p => Math.hypot(p[0] * p[0] - XS[0], Math.SQRT2 * (p[0] * p[1] - XS[1]), p[1] * p[1] - XS[2]);

export default {
  tabs: [
    {
      id: 'sensing', name: 'Sensing landscape', title: 'Too few measurements create bad valleys; enough of them leave only saddles and the truth',
      caption: 'Recover X* = u*u*ᵀ (rank 1, 2×2) from m random linear measurements yᵢ = ⟨Aᵢ, X*⟩ by gradient descent on the factor u. The surface is the real loss f(u). Twelve balls start at random. With m = 2 there are zero-loss points that are not X*. With m = 3–6 some draws have genuine spurious local minima (positive loss, positive-definite Hessian), e.g. draw 25 at m = 3 or draw 4 at m = 6. With m = 20 or fully observed, every ball ends at ±u* in every draw we tried.',
      note: 'f(u) = (1/m)Σ(uᵀAᵢu − yᵢ)², Aᵢ = (G + Gᵀ)/2 with G Gaussian, u* = (1, 0.5). The theorem needs (4r, 1/10)-RIP, far more measurements than shown here: it is sufficient, not necessary.',
      legend: [[HEX.ball, 'GD from random starts'], [HEX.good, '±u* (global)'], [HEX.neg, 'λ < 0 at u = 0']],
      action: 'Release balls', cam: [0, 4.6, 3.6], tgt: [0, .3, 0],
      controls: [{ id: 'm', type: 'seg', label: 'measurements m', value: 3, options: [[2, '2'], [3, '3'], [4, '4'], [6, '6'], [20, '20'], [0, 'all (fully observed)']] },
        { id: 'seed', label: 'measurement draw', min: 1, max: 40, step: 1, value: 25, live: false, fmt: v => String(+v) }],
      build(ctx) {
        const { L, g } = sensing(+ctx.vals.m, 1000 * (+ctx.vals.m) + ctx.vals.seed);
        ctx.d = def({ L, g, h: v => Math.log1p(v * 3), dom: [-1.6, 1.6, -1.6, 1.6] });
        const s = ctx.stage; s.surface(ctx.d); s.marker(ctx.d, ...US, C.good); s.marker(ctx.d, -US[0], -US[1], C.good); s.eigenLines(ctx.d, 0, 0, { len: .45 });
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage, R = rng(77); s.clearBalls(); ctx.bs = [];
        for (let k = 0; k < 12; k++) ctx.bs.push(s.ball(ctx.d, { start: [1.4 * (2 * R() - 1), 1.4 * (2 * R() - 1)], step: opt.gd(.02), speed: 6, maxSteps: 4000, r: .05, stop: b => Math.hypot(...ctx.d.g(...b.p)) < 1e-7 }));
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      frame(ctx) {
        if (!ctx.bs) return;
        let good = 0, wrong = 0, spur = 0, moving = 0;
        for (const b of ctx.bs) { if (!b.done) { moving++; continue; } const l = b.loss(), e = recErr(b.p); if (e < 1e-3) good++; else if (l < 1e-6) wrong++; else spur++; }
        ctx.stat(`m = ${+ctx.vals.m || 'all'} · ended at ±u*: ${good} · zero loss but UUᵀ ≠ X*: ${wrong} · stuck with loss > 0 (spurious minimum): ${spur}${moving ? ` · still moving ${moving}` : ''}`);
      },
    },
    {
      id: 'rip', name: 'RIP', mode: '2d', title: 'How many measurements the theorem asks for',
      caption: 'Restricted isometry constant δ of m Gaussian measurements on 2×2 symmetric matrices (here every matrix has rank ≤ 4r = 4, so RIP is a condition on all of them). The theorem’s conditions δ < 1/10 (noisy) and δ < 1/5 (noiseless) need hundreds of measurements, while spurious minima already become rare at m = 6 and were not found from m = 10 on.',
      note: 'Median over 200 draws; δ = max |eig(ΦᵀΦ/m) − 1| in an orthonormal basis of symmetric matrices. Side chart: GD endpoints that are not X* among 800 runs per m (40 draws × 20 starts, offline).',
      hint: '', legend: [[HEX.ball, 'median RIP constant δ'], [HEX.neg, 'runs ending in a spurious minimum'], [HEX.flat, 'runs ending at a wrong zero-loss point']],
      plot: { xlabel: 'measurements m', ylabel: 'δ', logx: true, xlim: [1.5, 2500], ylim: [0, 2] },
      charts: [{ id: 'sp', title: 'GD endpoints that are not X* (of 800)', xlabel: 'measurements m', ylabel: 'runs', logx: true, xlim: [.8, 60], ylim: [0, 800], h: 220, legend: false }],
      build(ctx) {
        const D = ctx.data, P = ctx.plot.clear();
        P.hline(.2, { color: HEX.good, dash: [5, 4], text: 'δ < 1/5 (noiseless)' }).hline(.1, { color: HEX.pos, dash: [5, 4], text: 'δ < 1/10' }).line(D.rip, { color: HEX.ball, width: 2.4, dots: 4 }).draw();
        ctx.charts.sp.clear().line(D.rows.map(r => [r[0], r[1]]), { color: HEX.neg, width: 2, dots: 4 }).line(D.rows.map(r => [r[0], r[2]]), { color: HEX.flat, width: 2, dots: 4 }).draw();
        const first = D.rip.find(r => r[1] < .2);
        ctx.stat(`median δ < 1/5 first at m = ${first ? first[0] : '>2000'} · offline, draws with a spurious minimum: ${D.rows.filter(r => r[0] >= 3).map(r => `m = ${r[0]}: ${r[4]}/40`).join(', ')}`);
      },
    },
  ],
};
