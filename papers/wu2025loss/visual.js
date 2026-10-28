// Wu, Simsek, Ged 2025 — ReLU kinks, escape neurons, and the Sec. 4.1.1 / Fig. 3 experiment replicated offline
// (scripts/check-math/wu2025loss.py -> data.json: loss curve + 501 snapshots of all 50 neurons).
import { def, C, HEX, segPts } from '../../src/lib/atlas3d.js';
import { ramp } from '../../src/lib/atlas3d.js';

const one = def({
  L: (w, a) => .5 * (a * Math.max(w, 0) - 1) ** 2,
  g: (w, a) => { const r = a * Math.max(w, 0) - 1; return [w > 0 ? r * a : 0, r * Math.max(w, 0)]; },
  h: L => L * 1.6, dom: [-1.6, 1.6, -1.6, 1.6],
});
one.checkDom = [.05, 1.6, -1.6, 1.6];
const snapAt = (D, e) => D.snaps[Math.max(0, Math.min(D.snaps.length - 1, Math.round(e / 1000)))];
const play = { label: 'Play training', fn: ctx => { ctx.playing = !ctx.playing; if (ctx.vals.e >= 499000) ctx.set('e', 0); } };
function advance(ctx, dt, draw) { if (!ctx.playing) return; const e = Math.min(500000, ctx.vals.e + dt * 60000); ctx.set('e', e); if (e >= 500000) ctx.playing = false; draw(); }

export default {
  tabs: [
    {
      id: 'kink', name: 'Kink & escape', title: 'At a ReLU kink, escape needs two weights to move together',
      caption: 'One neuron, one data point: ŷ = a·ReLU(w). At the origin no single weight lowers the loss, but moving w and a together does (second order). For w < 0 the neuron is dead and the loss is flat.',
      note: 'Teaching example from our notes (x = (1, 0), y = 1). Directional stationarity, not gradients, decides what is a minimum here (Def. 3.5, Thm. 4.2).',
      legend: [[HEX.ball, 'small living neuron'], [HEX.opt2, 'dead neuron'], [HEX.good, 'a·w = 1 (zero loss)'], [HEX.flat, 'dead region w < 0']],
      action: 'Release balls', cam: [-2.6, 3.4, 4.2], tgt: [0, .4, 0],
      build(ctx) {
        const s = ctx.stage;
        s.surface(one, { colorFn: (w, a, t) => w < 0 ? [.42, .35, .2].map((v, i) => v * .5 + ramp(Math.pow(t, .6))[i] * .5) : ramp(Math.pow(t, .6)) });
        const v = []; for (let w = .63; w <= 1.6; w += .01) v.push([w, 1 / w]);
        s.tube(one, v, C.good); s.tube(one, segPts(0, -1.6, 0, 1.6), C.white, .01);
        s.marker(one, 0, 0, C.white);
        s.label('kink w = 0', one.W(0, 1.35, .25), '', HEX.white); s.label('dead · ∇L = 0', one.W(-1, -.9, .3), '', HEX.flat);
        this.drop(ctx);
      },
      drop(ctx) {
        const s = ctx.stage; s.clearBalls();
        const gd = lr => (p, b, d) => { const [gw, ga] = d.g(p[0], p[1]); return [p[0] - lr * gw, p[1] - lr * ga]; };
        ctx.live = s.ball(one, { start: [.06, .05], step: gd(.05), speed: 1, maxSteps: 3000, stop: b => b.loss() < 1e-8 });
        ctx.dead = s.ball(one, { start: [-.5, .7], step: gd(.05), color: C.opt2, maxSteps: 3000 });
      },
      frame(ctx) { const a = ctx.live, b = ctx.dead; if (a) ctx.stat(`living: step ${a.steps}, L = ${a.loss().toFixed(4)}, w² − a² = ${(a.p[0] ** 2 - a.p[1] ** 2).toFixed(4)} · dead: L = ${b.loss().toFixed(3)} (never moves)`); },
    },
    {
      id: 'plateaus', name: 'Plateaus', mode: '2d', title: 'Saddle to saddle to a local minimum',
      caption: 'The paper’s experiment: 50 ReLU neurons, 5 points, tiny initialization, plain GD. The loss sits on plateaus near stationary points and drops when small living neurons grow. The last plateau is a local minimum.',
      note: 'Replicated with the paper’s setup and our own random draw: escapes near epochs 25k and 65k (paper: 42k and 152k). Right: ‖wᵢ‖ = |hᵢ| for every neuron, as Fact 4.5 predicts.',
      hint: 'scrub or play through training',
      controls: [{ id: 'e', label: 'epoch', min: 0, max: 500000, step: 1000, value: 0, fmt: v => `${Math.round(v / 1000)}k` }],
      actions: [play],
      plot: { xlabel: 'epoch', ylabel: 'training loss', xlim: [0, 500000], ylim: [.066, .098], xfmt: v => `${v / 1000}k` },
      charts: [{ id: 'bal', title: 'balance: ‖wᵢ‖ vs |hᵢ| (all 50 neurons)', xlabel: '‖wᵢ‖', ylabel: '|hᵢ|', logx: true, logy: true, xlim: [1e-6, .5], ylim: [1e-6, .5], legend: false, h: 260 }],
      build(ctx) { ctx.playing = false; this.draw(ctx); },
      onControl(ctx) { this.draw(ctx); },
      frame(ctx, dt) { advance(ctx, dt, () => this.draw(ctx)); },
      draw(ctx) {
        const D = ctx.data, e = ctx.vals.e, P = ctx.plot.clear();
        P.line(D.loss, { color: HEX.pos, width: 2 }).vline(e, { color: HEX.ball, width: 1.6 });
        P.hline(.0953, { color: HEX.muted, dash: [3, 5], text: 'origin saddle' }).hline(.0697, { color: HEX.good, dash: [3, 5], text: 'local minimum 0.0697' });
        P.draw();
        const S = snapAt(D, e), pts = S[2].map((n, i) => [Math.max(n, 1e-6), Math.max(S[3][i], 1e-6)]);
        ctx.charts.bal.clear().fn(x => x, { color: HEX.muted, dash: [4, 4] }).scatter(pts, { color: HEX.ball, r: 3 }).draw();
        const L = D.loss[Math.min(D.loss.length - 1, Math.round(e / 250))][1];
        ctx.stat(`epoch ${Math.round(e / 1000)}k · loss ${L.toFixed(5)} · largest neuron ‖w‖ = ${Math.max(...S[2]).toExponential(2)}`);
      },
    },
    {
      id: 'neurons', name: 'Neuron directions', mode: '2d', title: 'Neurons cluster, then grow',
      caption: 'Every hidden neuron’s input weight wᵢ ∈ R² as a dot (angle = direction, radius = log size). Grey: the dead wedge where wᵢ·xₖ ≤ 0 for all inputs. Faint lines: kinks wᵢ ⊥ xₖ. Living neurons gather at a few angles, then grow during each escape.',
      note: 'Same run as Plateaus. Dot colour = |hᵢ| (output weight). Group-1 neurons sit at 2.53 rad, on the kink ⊥ x₅, as in the paper.',
      hint: 'scrub or play through training', legend: [],
      controls: [{ id: 'e', label: 'epoch', min: 0, max: 500000, step: 1000, value: 0, fmt: v => `${Math.round(v / 1000)}k` }],
      actions: [play],
      plot: { xlim: [-1.25, 1.25], ylim: [-1.15, 1.15], legend: false, grid: false, xfmt: () => '', yfmt: () => '' },
      build(ctx) { ctx.playing = false; this.draw(ctx); },
      onControl(ctx) { this.draw(ctx); },
      frame(ctx, dt) { advance(ctx, dt, () => this.draw(ctx)); },
      draw(ctx) {
        const D = ctx.data, S = snapAt(D, ctx.vals.e), P = ctx.plot.clear(), rad = n => Math.max(.04, (Math.log10(Math.max(n, 1e-7)) + 6.5) / 6);
        P.custom((g, p) => {
          const o = [p.sx(0), p.sy(0)], R = r => p.sx(r) - p.sx(0);
          g.fillStyle = 'rgba(142,153,189,.12)'; g.beginPath(); g.moveTo(...o); g.arc(o[0], o[1], R(1.1), -D.dead_wedge[1], -D.dead_wedge[0]); g.closePath(); g.fill();
          g.strokeStyle = 'rgba(142,153,189,.28)'; g.setLineDash([3, 4]);
          for (const x of D.x) { const t = Math.atan2(-x, 1); g.beginPath(); g.moveTo(o[0] - R(1.1) * Math.cos(t), o[1] + R(1.1) * Math.sin(t)); g.lineTo(o[0] + R(1.1) * Math.cos(t), o[1] - R(1.1) * Math.sin(t)); g.stroke(); }
          g.setLineDash([]); g.strokeStyle = 'rgba(142,153,189,.35)'; g.fillStyle = 'rgba(142,153,189,.8)'; g.font = '10px JetBrains Mono'; g.textAlign = 'left';
          for (const [v, l] of [[1e-5, '10⁻⁵'], [1e-3, '10⁻³'], [1e-1, '10⁻¹']]) { g.beginPath(); g.arc(o[0], o[1], R(rad(v)), 0, 7); g.stroke(); g.fillText(l, o[0] + R(rad(v)) + 3, o[1] - 3); }
          const hm = Math.max(...S[3]);
          S[1].forEach((ang, i) => {
            const rr = R(rad(S[2][i])), t = Math.min(1, S[3][i] / Math.max(hm, 1e-12));
            g.fillStyle = `rgba(255,${Math.round(122 + 100 * (1 - t))},${Math.round(89 + 140 * (1 - t))},${.35 + .65 * t})`;
            g.beginPath(); g.arc(o[0] + rr * Math.cos(ang), o[1] - rr * Math.sin(ang), 5.5, 0, 7); g.fill();
          });
        });
        P.draw();
        ctx.stat(`epoch ${Math.round(ctx.vals.e / 1000)}k · radius = log₁₀‖wᵢ‖ · dead wedge ${D.dead_wedge[0]}–${D.dead_wedge[1]} rad`);
      },
    },
  ],
};
