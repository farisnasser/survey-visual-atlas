// Kawaguchi 2016 — port of reference/saddles.html onto the shared library. Checked by scripts/check-math/kawaguchi2016deep.py
import { def, opt, C, HEX, logH, hyperbola, segPts } from '../../src/lib/atlas3d.js';

const strict = o => def({ L: (x, y) => (1 - x * y) ** 2, g: (x, y) => { const r = 1 - x * y; return [-2 * y * r, -2 * x * r]; }, h: logH, fit: null, ...o });
const bad = o => def({ L: (x, y) => (1 - x * y * y) ** 2, g: (x, y) => { const r = 1 - x * y * y; return [-2 * y * y * r, -4 * x * y * r]; }, h: logH, fit: null, ...o });
const bumpy = o => def({
  L: (x, y) => .22 * (x * x + y * y) + .3 * (1 - Math.cos(4 * x) * Math.cos(4 * y)),
  g: (x, y) => [.44 * x + 1.2 * Math.sin(4 * x) * Math.cos(4 * y), .44 * y + 1.2 * Math.cos(4 * x) * Math.sin(4 * y)], h: L => L * .9, fit: null, ...o,
});
const halt = b => b.loss() < 1e-5 || Math.hypot(...b.d.g(...b.p)) < 3e-4;   // same stop rule as reference/saddles.html
const stat = b => b ? `step ${b.steps} · loss ${b.loss().toFixed(3)}` : '';
const statLabels = (ctx, items) => { ctx.sl = items.map(([html, v]) => ctx.stage.label(`${html}<span class="stat"></span>`, v, 'big')); };
const frame = ctx => (ctx.sl || []).forEach((L, i) => { const s = L.e.querySelector('.stat'); if (s) s.textContent = stat(ctx.stage.balls[i]); });

export default {
  tabs: [
    {
      id: 'valleys', name: 'Valleys', title: 'No poor local minima',
      caption: 'Left: a landscape with shallow pits. Right: a linear network’s loss. Every valley on the right sits at the bottom.',
      legend: [[HEX.ball, 'gradient descent'], [HEX.good, 'global minima']], action: 'Drop balls',
      cam: [0, 4.4, 6.2], tgt: [0, .3, 0],
      build(ctx) {
        const s = ctx.stage, A = s.surface(bumpy({ ox: -2 })), B = s.surface(strict({ ox: 2 }));
        s.tube(B, hyperbola(1, 1.6, 1), C.good); s.tube(B, hyperbola(1, 1.6, -1), C.good);
        s.marker(A, 0, 0, C.good);
        ctx.A = A; ctx.B = B;
        statLabels(ctx, [['Poor local minima', A.W(0, -1.95, 0).setY(0)], ['This paper', B.W(0, -1.95, 0).setY(0)]]);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        const st = [(Math.random() * 2 - 1) * 1.35, (Math.random() * 2 - 1) * 1.35];
        s.ball(ctx.A, { start: st, step: opt.gd(.02), speed: 2, stop: halt });
        s.ball(ctx.B, { start: st, step: opt.gd(.01), speed: 2, stop: halt });
      },
      frame,
    },
    {
      id: 'good', name: 'Good saddle', title: 'Escapable saddle · 1 hidden layer',
      caption: 'The surface curves down along one direction (negative Hessian eigenvalue). A nudge and the ball slides off.',
      note: 'Scalar weights stand in for the weight matrices.',
      legend: [[HEX.neg, 'curves down · λ = −2'], [HEX.pos, 'curves up · λ = +2']], action: 'Nudge ball',
      cam: [2.8, 2.7, 3.4], tgt: [0, .45, 0],
      build(ctx) {
        const s = ctx.stage, S = ctx.S = s.surface(strict());
        s.tube(S, segPts(-1.1, -1.1, 1.1, 1.1), C.neg); s.tube(S, segPts(-1.1, 1.1, 1.1, -1.1), C.pos);
        s.marker(S, 0, 0, C.white);
        s.label('λ = −2', S.W(.85, .85, .25), '', HEX.neg); s.label('λ = +2', S.W(.85, -.85, .2), '', HEX.pos);
        statLabels(ctx, [['', S.W(0, 0, 0).setY(1.25)]]);
        this.drop(ctx);
      },
      drop(ctx) { ctx.stage.clearBalls(); ctx.stage.ball(ctx.S, { start: [.05, .04], step: opt.gd(.02), stop: halt }); },
      frame,
    },
    {
      id: 'bad', name: 'Bad saddle', title: 'Bad saddle · 2+ hidden layers',
      caption: 'No direction curves down: the Hessian has no negative eigenvalue. The top is a flat plateau, so the ball creeps before it finds the way out.',
      note: 'Slice w₃ = w₂ of a three-weight chain (1 − w₁w₂w₃)².',
      legend: [[HEX.flat, 'flat · λ = 0'], [HEX.good, 'global minima']], action: 'Nudge ball',
      cam: [2.8, 2.7, 3.4], tgt: [0, .45, 0],
      build(ctx) {
        const s = ctx.stage, S = ctx.S = s.surface(bad());
        s.tube(S, segPts(-.75, 0, .75, 0), C.flat); s.tube(S, segPts(0, -.75, 0, .75), C.flat);
        const v1 = [], v2 = []; for (let y = .79; y <= 1.6; y += .01) { v1.push([1 / (y * y), y]); v2.push([1 / (y * y), -y]); }
        s.tube(S, v1, C.good); s.tube(S, v2, C.good);
        s.marker(S, 0, 0, C.white);
        s.label('λ = 0, 0', S.W(0, .8, .25), '', HEX.flat);
        statLabels(ctx, [['', S.W(0, 0, 0).setY(1.25)]]);
        this.drop(ctx);
      },
      drop(ctx) { ctx.stage.clearBalls(); ctx.stage.ball(ctx.S, { start: [.05, .04], step: opt.gd(.02), stop: halt }); },
      frame,
    },
    {
      id: 'race', name: 'Race', title: 'Same nudge, same step size',
      caption: 'Both balls start the same distance from the saddle. Depth turns a quick slide into a long crawl.',
      legend: [[HEX.neg, 'escapable'], [HEX.flat, 'bad (flat)']], action: 'Restart race',
      cam: [0, 4.4, 6.2], tgt: [0, .35, 0],
      build(ctx) {
        const s = ctx.stage, A = ctx.A = s.surface(strict({ ox: -2 })), B = ctx.B = s.surface(bad({ ox: 2 }));
        s.tube(A, segPts(-.9, -.9, .9, .9), C.neg); s.tube(B, segPts(-.7, 0, .7, 0), C.flat); s.tube(B, segPts(0, -.7, 0, .7), C.flat);
        s.marker(A, 0, 0, C.white); s.marker(B, 0, 0, C.white);
        statLabels(ctx, [['1 hidden layer', A.W(0, -1.95, 0).setY(0)], ['2 hidden layers', B.W(0, -1.95, 0).setY(0)]]);
        this.drop(ctx);
      },
      drop(ctx) { const s = ctx.stage; s.clearBalls(); s.ball(ctx.A, { start: [.05, .04], step: opt.gd(.02), stop: halt }); s.ball(ctx.B, { start: [.05, .04], step: opt.gd(.02), stop: halt }); },
      frame,
    },
  ],
};
