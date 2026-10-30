// He, Zhang, Ren, Sun 2016 — residual learning: y = F(x) + x. In a deep plain chain the gradient at a small initialization
// vanishes; with identity shortcuts it does not. Toy chain and reported Table 2: scripts/check-math/he2016deep.py
import { def, C, HEX, statLabels } from '../../src/lib/atlas3d.js';

const Y = 2, Z0 = [.02, .03];
function chain(K, res, ox) {
  const f = res ? (a, b) => (1 + a) ** K * (1 + b) ** K : (a, b) => a ** K * b ** K;
  const g = res ? (a, b) => { const r = f(a, b) - Y; return [r * K * (1 + a) ** (K - 1) * (1 + b) ** K, r * K * (1 + a) ** K * (1 + b) ** (K - 1)]; }
    : (a, b) => { const r = f(a, b) - Y; return [r * K * a ** (K - 1) * b ** K, r * K * a ** K * b ** (K - 1)]; };
  return def({ L: (a, b) => .5 * (f(a, b) - Y) ** 2, g, dom: [-.3, 1.7, -.3, 1.7], h: v => Math.log1p(Math.log1p(v)), size: 2.9, ox, N: 90 });   // double log: display only
}

export default {
  tabs: [
    {
      id: 'chain', name: 'Plain vs residual', title: 'Shortcuts keep the gradient alive in a deep chain',
      caption: 'The same deep linear chain of 2K scalar layers (weights a and b, alternating), trained by GD to map 1 to 2 from a small initialization. Left, plain layers h → wh: the output is aᴷbᴷ, the loss is flat around the start and the gradient there vanishes like K·aᴷ⁻¹bᴷ, so for 20 layers GD does not move. Right, residual layers h → h + wh: the output is (1 + a)ᴷ(1 + b)ᴷ, the start is already close to the identity map, the gradient is order one, and GD converges.',
      note: 'A two-parameter toy (tied weights, linear, one example), not a CNN. Step sizes: the best of a small grid for each chain and depth (check script). Display height is log(1 + log(1 + loss)).',
      legend: [[HEX.opt2, 'GD, plain chain'], [HEX.ball, 'GD, residual chain'], [HEX.white, 'start (0.02, 0.03)']],
      action: 'Restart', cam: [0, 5.4, 6.4], tgt: [0, .5, 0],
      controls: [{ id: 'K', type: 'seg', label: 'depth 2K', value: 10, options: [[1, '2'], [3, '6'], [10, '20']] }],
      charts: [{ id: 'L', title: 'loss (log)', xlabel: 'step', ylabel: 'loss', logy: true, ylim: [1e-12, 10], xlim: [0, 2000], h: 220, legend: false }],
      build(ctx) {
        const K = +ctx.vals.K, s = ctx.stage; ctx.dp = chain(K, false, -1.65); ctx.dr = chain(K, true, 1.65);
        s.surface(ctx.dp); s.surface(ctx.dr); s.marker(ctx.dp, ...Z0, C.white, .06); s.marker(ctx.dr, ...Z0, C.white, .06);
        ctx.labs = statLabels(s, [['plain: aᴷbᴷ', ctx.dp.W(.7, 1.7, 1.2)], ['residual: (1 + a)ᴷ(1 + b)ᴷ', ctx.dr.W(.7, 1.7, 1.2)]]);
        this.drop(ctx);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, row = ctx.data.rows.find(r => r[0] === +ctx.vals.K); s.clearBalls(); ctx.H = { p: [], r: [] };
        const gd = (d, eta) => p => { const g = d.g(...p); return [p[0] - eta * g[0], p[1] - eta * g[1]]; };
        ctx.bp = s.ball(ctx.dp, { start: Z0, speed: 6, maxSteps: 2000, color: C.opt2, step: gd(ctx.dp, row[4]) }); ctx.bp.onStep = b => ctx.H.p.push([b.steps, ctx.dp.L(...b.p)]);
        ctx.br = s.ball(ctx.dr, { start: Z0, speed: 6, maxSteps: 2000, step: gd(ctx.dr, row[6]) }); ctx.br.onStep = b => ctx.H.r.push([b.steps, ctx.dr.L(...b.p)]);
        ctx.row = row;
      },
      frame(ctx) {
        ctx.charts.L.clear().line(ctx.H.p.map(h => [h[0], Math.max(h[1], 1e-12)]), { color: HEX.opt2, width: 1.6 }).line(ctx.H.r.map(h => [h[0], Math.max(h[1], 1e-12)]), { color: HEX.ball, width: 1.8 }).draw();
        const row = ctx.row, lp = ctx.dp.L(...ctx.bp.p), lr = ctx.dr.L(...ctx.br.p);
        ctx.labs[0].stat(`|∇L| at start ${row[1].toExponential(1)} · loss ${lp.toExponential(1)}`); ctx.labs[1].stat(`|∇L| at start ${row[2].toFixed(2)} · loss ${lr.toExponential(1)}`);
        ctx.stat(`depth ${2 * row[0]} · step ${ctx.br.steps} · plain (η ${row[4]}): loss ${lp.toExponential(2)} · residual (η ${row[6]}): loss ${lr.toExponential(2)}`);
      },
    },
    {
      id: 'imagenet', name: 'Degradation', mode: '2d', title: 'Deeper plain nets train worse; deeper ResNets train better',
      caption: 'Reported top-1 error on ImageNet (Table 2, 10-crop): going from 18 to 34 layers makes the plain network worse (27.94% → 28.54%), the degradation problem, which the paper shows is a training difficulty since training error rises too (Fig. 4). With identity shortcuts and no extra parameters the 34-layer ResNet improves to 25.03%.',
      note: 'Reported in the paper, not reproduced. The survey cites ResNets for smoother loss landscapes (via Li et al. 2018’s visualizations).',
      hint: '', legend: [[HEX.opt2, 'plain'], [HEX.ball, 'ResNet']],
      plot: { xlabel: '', ylabel: 'top-1 error (%)', ylim: [24, 29.5], legend: false, xfmt: () => '' },
      build(ctx) {
        const T = ctx.data.table2, P = ctx.plot.clear();
        const items = [{ y: T.plain[0], color: HEX.opt2 }, { y: T.resnet[0], color: HEX.ball }, { y: T.plain[1], color: HEX.opt2 }, { y: T.resnet[1], color: HEX.ball }];
        P.bars(items);
        ['plain-18', 'ResNet-18', 'plain-34', 'ResNet-34'].forEach((l, i) => P.text(i, items[i].y + .15, `${l}: ${items[i].y}`, { color: HEX.white, align: 'center' }));
        P.draw();
        ctx.stat(`18 → 34 layers: plain ${T.plain[0]}% → ${T.plain[1]}% (worse), ResNet ${T.resnet[0]}% → ${T.resnet[1]}% (better)`);
      },
    },
  ],
};
