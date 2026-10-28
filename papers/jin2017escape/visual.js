// Jin et al. 2017 — perturbed gradient descent (Algorithm 2). Checked in scripts/check-math/jin2017escape.py
import { def, opt, C, HEX, segPts, circlePts } from '../../src/lib/atlas3d.js';
import { rng, median } from './util.js';
import { rampCss } from '../../src/lib/charts.js';

const LR = .2;
const F = def({ L: (x, y) => .5 * x * x + Math.cos(y) - 1, g: (x, y) => [x, -Math.sin(y)], hess: (x, y) => [1, 0, -Math.cos(y)], h: v => (v + 2.05) * .55, dom: [-2, 2, -4, 4], size: 3.6 });

/** Algorithm 2 as a ball step: GD, a uniform kick in B(r) when |grad| <= g_thres (at most once per t_thres steps),
 *  and termination (return the pre-kick point) if f did not drop by f_thres within t_thres steps. */
const pgd = (r, gT, fT, tT, R) => (p, b, d) => {
  b.t = (b.t ?? -1) + 1; b.tn ??= -tT - 1; b.kicks ??= 0;
  let q = p;
  if (Math.hypot(...d.g(...q)) <= gT && b.t - b.tn > tT) {
    b.xt = q.slice(); b.tn = b.t; b.kicks++;
    const a = R() * 2 * Math.PI, rad = r * Math.sqrt(R());
    q = [q[0] + rad * Math.cos(a), q[1] + rad * Math.sin(a)];
    b.kickAt = b.xt; b.onKick?.(b);
  }
  if (b.xt && b.t - b.tn === tT && d.L(...q) - d.L(...b.xt) > -fT) { b.finished = true; return b.xt; }
  const [gx, gy] = d.g(...q); return [q[0] - LR * gx, q[1] - LR * gy];
};

export default {
  tabs: [
    {
      id: 'kick', name: 'Kick', title: 'A tiny random kick unsticks gradient descent',
      caption: 'Both balls start on the saddle’s stable set. Plain GD (white) converges to the saddle and stops. PGD (coral) adds one uniform kick when the gradient gets small, then GD carries it to a minimum.',
      note: 'f = x²/2 + cos y − 1 (gradient and Hessian 1-Lipschitz). Kick radius drawn as a ring; g_thres = 10⁻³, t_thres = 30, f_thres = 0.05.',
      legend: [[HEX.opt3, 'plain GD'], [HEX.ball, 'perturbed GD'], [HEX.good, 'minima f = −2'], [HEX.neg, 'λ = −1'], [HEX.pos, 'λ = +1']],
      action: 'Run again', cam: [3.6, 3.2, 3.6], tgt: [0, .5, 0],
      controls: [{ id: 'r', label: 'kick radius r', min: .005, max: .3, step: .005, value: .05, digits: 3 }],
      build(ctx) {
        const s = ctx.stage; s.surface(F);
        s.eigenLines(F, 0, 0, { len: .8 }); s.marker(F, 0, 0, C.white);
        for (const y of [Math.PI, -Math.PI]) { s.tube(F, segPts(-2, y, 2, y), C.good, .012); }
        s.label('f = −2', F.W(1.4, Math.PI, .3), '', HEX.good);
        this.drop(ctx);
      },
      onControl(ctx) { this.drop(ctx); },
      drop(ctx) {
        const s = ctx.stage, R = rng(Date.now() % 1e6); s.clearBalls(); (ctx.rings || []).forEach(m => s.remove(m)); ctx.rings = [];
        ctx.gd = s.ball(F, { start: [1.6, 0], step: opt.gd(LR), color: C.opt3, speed: .6, maxSteps: 600 });
        ctx.pg = s.ball(F, { start: [1.6, 0], step: pgd(ctx.vals.r, 1e-3, .05, 30, R), speed: .6, maxSteps: 900, stop: b => b.finished });
        ctx.pg.onKick = b => ctx.rings.push(s.tube(F, circlePts(b.xt[0], b.xt[1], Math.max(ctx.vals.r, .03)), C.flat, .01, .03));
      },
      frame(ctx) {
        const a = ctx.gd, b = ctx.pg; if (!a || !b) return;
        const atSaddle = b.finished && Math.abs(b.p[1]) < .5;
        ctx.stat(`GD: f = ${a.loss().toFixed(3)}, |∇f| = ${Math.hypot(...F.g(...a.p)).toExponential(1)} · PGD: f = ${b.loss().toFixed(3)}, kicks ${b.kicks || 0}`
          + (b.finished ? (atSaddle ? ' · stopped at the saddle: this r is too small for t_thres (outside the theorem’s parameter choice)' : ' · returned an ε-SOSP (a minimum here)') : ''));
      },
    },
    {
      id: 'pancake', name: 'Thin pancake', title: 'Failed kicks fill a thin, bent pancake',
      caption: 'Kicks sampled in a ball around a 3D saddle. Amber points are the ones that do not escape within t_thres steps. They form a thin curved slab, so a random kick fails with small probability.',
      note: 'f = (x² + z²)/2 + cos(y − x²/2) − 1, a strict saddle at 0 with λ = (1, −1, 1). "Stuck" = f drops by < 0.05 in t_thres GD steps (η = 0.2).',
      legend: [[HEX.flat, 'stuck kicks'], ['#3c5d8a', 'kicks that escape'], [HEX.neg, 'negative-curvature direction y']],
      cam: [3.2, 2.2, 3.4], tgt: [0, 0, 0],
      controls: [{ id: 'T', label: 't_thres', min: 8, max: 40, step: 1, value: 14, live: false, fmt: v => (+v).toFixed(0) }],
      build(ctx) { this.compute(ctx); },
      onControl(ctx) { ctx.stage.clear(); this.compute(ctx); },
      compute(ctx) {
        const s = ctx.stage, R = rng(11), T = ctx.vals.T, rad = .6, sc = 1.9 / rad, pos = [], col = [];
        let stuck = 0, n = 0;
        while (n < 9000) {
          const x = R.uniform(-rad, rad), y = R.uniform(-rad, rad), z = R.uniform(-rad, rad);
          if (x * x + y * y + z * z > rad * rad) continue; n++;
          let a = x, b = y, c = z;
          for (let k = 0; k < T; k++) { const u = b - .5 * a * a, sn = Math.sin(u); [a, b, c] = [a - LR * (a + a * sn), b + LR * sn, c - LR * c]; }
          const dec = -(.5 * (a * a + c * c) + Math.cos(b - .5 * a * a) - 1), st = dec < .05;
          if (st) stuck++;
          pos.push([x * sc, y * sc, -z * sc]);
          col.push(st ? [1, .78, .34] : [.24, .36, .54]);
        }
        s.points(pos, col, .045);
        const THREE = window.THREE;
        s.segment(new THREE.Vector3(0, -2.3, 0), new THREE.Vector3(0, 2.3, 0), C.neg);
        s.segment(new THREE.Vector3(-2.3, 0, 0), new THREE.Vector3(2.3, 0, 0), C.pos, .6);
        s.segment(new THREE.Vector3(0, 0, -2.3), new THREE.Vector3(0, 0, 2.3), C.pos, .6);
        s.label('y (λ = −1)', new THREE.Vector3(0, 2.45, 0), '', HEX.neg); s.label('x', new THREE.Vector3(2.45, 0, 0), '', HEX.pos); s.label('z', new THREE.Vector3(0, 0, -2.45), '', HEX.pos);
        ctx.stat(`${stuck} of ${n} kicks stuck (${(100 * stuck / n).toFixed(1)}%) with t_thres = ${T}`);
      },
    },
    {
      id: 'band', name: 'Narrow band', mode: '2d', title: 'In 2D the stuck region is a narrow band',
      caption: 'Colour = how much f drops in t_thres GD steps after a kick to that point. The amber band (drop < f_thres) hugs the stable direction and shrinks fast as t_thres grows.',
      note: 'f = x²/2 + γ(cos y − 1): saddle curvature −γ. Paper Fig. 2 shows the same band under gradient flow.',
      hint: '', legend: [[HEX.flat, 'stuck (drop < 0.05γ)']],
      controls: [{ id: 'T', label: 't_thres', min: 4, max: 40, step: 1, value: 12, fmt: v => (+v).toFixed(0) }, { id: 'gm', label: 'curvature γ', min: .2, max: 1, step: .05, value: 1 }],
      plot: { xlim: [-.5, .5], ylim: [-.5, .5], xlabel: 'kick x', ylabel: 'kick y (unstable direction)', legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const T = ctx.vals.T, gm = ctx.vals.gm, n = 110, z = [];
        let st = 0, tot = 0;
        for (let j = 0; j < n; j++) {
          const row = [];
          for (let i = 0; i < n; i++) {
            const x0 = -.5 + (i + .5) / n, y0 = -.5 + (j + .5) / n; if (x0 * x0 + y0 * y0 > .25) { row.push(null); continue; }
            let x = x0, y = y0; for (let k = 0; k < T; k++) { x -= LR * x; y += LR * gm * Math.sin(y); }
            const dec = -(.5 * x * x + gm * (Math.cos(y) - 1)); row.push(dec); tot++; if (dec < .05 * gm) st++;
          }
          z.push(row);
        }
        const mx = Math.max(...z.flat().filter(v => v !== null));
        ctx.plot.clear().heat(z, [-.5, .5, -.5, .5], { cell: v => v === null ? 'rgba(0,0,0,0)' : v < .05 * gm ? '#ffc857' : rampCss(Math.pow(Math.max(v, 0) / mx, .5)) }).draw();
        ctx.stat(`stuck fraction of the kick disc: ${(100 * st / tot).toFixed(2)}%`);
      },
    },
    {
      id: 'dimension', name: 'Dimension', mode: '2d', title: 'Nearly dimension-free',
      caption: 'Iterations for PGD to escape from exactly the saddle after one kick, as the number of parameters d grows (extra directions are stable). The count grows like log d, not like d.',
      note: 'Median over 40 random kicks of radius 0.05 per d. Theorem 3’s bound is O(log⁴ d / ε²).',
      hint: '', legend: [],
      plot: { logx: true, xlabel: 'dimension d', ylabel: 'iterations to escape', xlim: [1.5, 15000], ylim: [0, 70] },
      build(ctx) {
        const R = rng(5), pts = [];
        for (const d of [2, 5, 10, 30, 100, 300, 1000, 3000, 10000]) {
          const its = [];
          for (let s = 0; s < 40; s++) {
            const x = new Float64Array(d); let nr = 0; for (let i = 0; i < d; i++) { x[i] = R.normal(); nr += x[i] * x[i]; }
            const sc = .05 * Math.pow(R(), 1 / d) / Math.sqrt(nr); for (let i = 0; i < d; i++) x[i] *= sc;
            let t = 0; while (Math.cos(x[1]) - 1 >= -1 && t < 5000) { for (let i = 0; i < d; i++) x[i] = i === 1 ? x[i] + LR * Math.sin(x[i]) : x[i] - LR * x[i]; t++; }
            its.push(t);
          }
          pts.push([d, median(its)]);
        }
        const a = pts[0][1], b = (pts[pts.length - 1][1] - a) / Math.log(10000 / 2);
        ctx.plot.clear().fn(d => a + b * Math.log(d / 2), { color: HEX.muted, dash: [5, 5], label: 'a + b·log d' }).fn(d => a * d / 2, { color: HEX.neg, dash: [2, 4], label: '∝ d (for contrast)' })
          .line(pts, { color: HEX.ball, width: 2, dots: 3.5, label: 'PGD, median of 40 kicks' }).draw();
        ctx.stat(`d = 2: ${pts[0][1]} iterations · d = 10,000: ${pts[pts.length - 1][1]} iterations`);
      },
    },
  ],
};
