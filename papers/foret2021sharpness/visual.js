// Foret, Kleiner, Mobahi, Neyshabur 2021 — sharpness-aware minimization: minimize the worst loss in a ρ-ball (Eq. 1),
// approximated by one normalized ascent step (Eqs. 2-3). Toy, basins and the stall point: scripts/check-math/foret2021sharpness.py
import { def, C, HEX, statLabels } from '../../src/lib/atlas3d.js';

const CS = [-.55, .35], SS = .16, CF = [.9, -.45], SF = .55, B = .04, AF = 1.035, LR = .01, START = [-1.36, -.32];
const ws = (x, y) => Math.exp(-((x - CS[0]) ** 2 + (y - CS[1]) ** 2) / (2 * SS * SS)), wf = (x, y) => AF * Math.exp(-((x - CF[0]) ** 2 + (y - CF[1]) ** 2) / (2 * SF * SF));
const L = (x, y) => 1 + B * (x * x + y * y) - ws(x, y) - wf(x, y);
const G = (x, y) => { const s = ws(x, y), f = wf(x, y); return [2 * B * x + s * (x - CS[0]) / SS ** 2 + f * (x - CF[0]) / SF ** 2, 2 * B * y + s * (y - CS[1]) / SS ** 2 + f * (y - CF[1]) / SF ** 2]; };
const DIRS = Array.from({ length: 48 }, (_, k) => [Math.cos(2 * Math.PI * k / 48), Math.sin(2 * Math.PI * k / 48)]);
/** Worst point of the ρ-ball, approximated by its centre and 48 points of the boundary circle. */
function worst(x, y, rho) { let bx = x, by = y, bv = L(x, y); for (const [c, s] of DIRS) { const v = L(x + rho * c, y + rho * s); if (v > bv) { bv = v; bx = x + rho * c; by = y + rho * s; } } return [bx, by, bv]; }
const DOM = [-1.6, 1.6, -1.6, 1.6], H = v => Math.log1p(v * 3);
const steps = {
  gd: () => p => { const [gx, gy] = G(p[0], p[1]); return [p[0] - LR * gx, p[1] - LR * gy]; },
  sam: rho => p => { const [gx, gy] = G(p[0], p[1]), n = Math.hypot(gx, gy) || 1, [hx, hy] = G(p[0] + rho * gx / n, p[1] + rho * gy / n); return [p[0] - LR * hx, p[1] - LR * hy]; },
  exact: rho => p => { const [wx, wy] = worst(p[0], p[1], rho), [gx, gy] = G(wx, wy); return [p[0] - LR * gx, p[1] - LR * gy]; },
};
const FATE_COL = { 1: 'rgba(95,224,176,.75)', 2: 'rgba(255,93,115,.75)', 0: 'rgba(142,153,189,.35)' };

export default {
  tabs: [
    {
      id: 'objective', name: 'SAM objective', title: 'Minimizing the worst nearby loss lifts the sharp well',
      caption: 'Left: a loss with a narrow (sharp) and a wide (flat) well of equal depth. Right: the SAM objective, the worst loss inside a ball of radius ρ around each point (Eq. 1). The sharp well rises; the flat one barely moves. From the same start GD (coral) drops into the sharp well, descent on the exact SAM objective (violet) reaches the flat well, and the practical one-step SAM update (white, Eqs. 2–3) stalls between them.',
      note: 'Toy loss on a plane; the ρ-ball maximum uses its centre plus 48 boundary points. Step 0.01 for all three. Not the paper’s networks.',
      legend: [[HEX.ball, 'GD on L'], [HEX.opt3, 'SAM step (Eqs. 2–3) on L'], [HEX.opt2, 'GD on exact SAM objective'], [HEX.good, 'flat min'], [HEX.neg, 'sharp min']],
      action: 'Restart', cam: [0, 6.2, 3.6], tgt: [0, .9, -.2],
      controls: [{ id: 'rho', label: 'radius ρ', min: .05, max: .4, step: .01, value: .3, live: false }],
      build(ctx) {
        const s = ctx.stage, rho = ctx.vals.rho;
        ctx.dl = def({ L, g: G, dom: DOM, h: H, size: 3.1, ox: -1.75, N: 90 });
        ctx.dr = def({ L: (x, y) => worst(x, y, rho)[2], dom: DOM, h: H, size: 3.1, ox: 1.75, N: 80 });
        s.surface(ctx.dl); s.surface(ctx.dr);
        for (const d of [ctx.dl, ctx.dr]) { s.marker(d, ...ctx.data.flat, C.good, .1); s.marker(d, ...ctx.data.sharp, C.neg, .07); }
        ctx.labs = statLabels(s, [['loss L', ctx.dl.W(0, 1.6, .9)], [`SAM objective, ρ = ${rho.toFixed(2)}`, ctx.dr.W(0, 1.6, .9)]]);
        this.drop(ctx);
      },
      onControl(ctx) { ctx.stage.clear(); this.build(ctx); },
      drop(ctx) {
        const s = ctx.stage, rho = ctx.vals.rho; s.clearBalls();
        ctx.bg = s.ball(ctx.dl, { start: START, speed: 4, maxSteps: 3000, step: steps.gd() });
        ctx.bs = s.ball(ctx.dl, { start: START, speed: 4, maxSteps: 3000, step: steps.sam(rho), color: C.opt3 });
        ctx.be = s.ball(ctx.dr, { start: START, speed: 4, maxSteps: 3000, step: steps.exact(rho), color: C.opt2 });
      },
      frame(ctx) {
        const rho = ctx.vals.rho, sh = ctx.data.sharp, fl = ctx.data.flat, at = b => Math.hypot(b.p[0] - fl[0], b.p[1] - fl[1]) < .35 ? 'flat well' : Math.hypot(b.p[0] - sh[0], b.p[1] - sh[1]) < .35 ? 'sharp well' : 'between wells';
        ctx.labs[0].stat(`L(sharp) = ${L(...sh).toFixed(3)} · L(flat) = ${L(...fl).toFixed(3)}`);
        ctx.labs[1].stat(`sharp ${worst(...sh, rho)[2].toFixed(3)} · flat ${worst(...fl, rho)[2].toFixed(3)}`);
        ctx.stat(`step ${ctx.bg.steps} · GD: ${at(ctx.bg)} · exact SAM objective: ${at(ctx.be)} · one-step SAM: ${at(ctx.bs)} (L = ${L(...ctx.bs.p).toFixed(3)}) · λ_max: sharp ${ctx.data.lamS}, flat ${ctx.data.lamF}`);
      },
    },
    {
      id: 'basins', name: 'Where starts end', mode: '2d', title: 'Which well does each starting point reach?',
      caption: 'Every start on a 41 × 41 grid, run for 2500 steps. GD sends about a quarter of the plane into the sharp well (red). Descent on the exact SAM objective shrinks that region as ρ grows. The practical one-step SAM update also sends fewer starts to the sharp well, but many end up jittering between the wells (grey) instead of reaching the flat one.',
      note: 'Same toy and step size as the first tab; precomputed in the check script. Green: flat well, red: sharp well, grey: neither.',
      hint: '', legend: [[HEX.good, 'ends in flat well'], [HEX.neg, 'ends in sharp well'], [HEX.muted, 'neither']],
      controls: [{ id: 'm', type: 'seg', label: 'method', value: 'sam', options: [['gd', 'GD'], ['sam', 'SAM step'], ['exact', 'exact SAM']] }, { id: 'rho', type: 'seg', label: 'radius ρ', value: .4, options: [[.1, '0.1'], [.2, '0.2'], [.4, '0.4']] }],
      plot: { xlabel: 'start x', ylabel: 'start y', xlim: [-1.64, 1.64], ylim: [-1.64, 1.64], legend: false },
      build(ctx) { this.draw(ctx); }, onControl(ctx) { this.draw(ctx); },
      draw(ctx) {
        const m = ctx.vals.m, key = m === 'gd' ? 'gd-0' : `${m}-${ctx.vals.rho}`, s = ctx.data.maps[key], n = 41, z = [];
        for (let j = 0; j < n; j++) { const row = []; for (let i = 0; i < n; i++) row.push(+s[j * n + i]); z.push(row); }
        ctx.plot.clear().heat(z, [-1.64, 1.64, -1.64, 1.64], { cell: v => FATE_COL[v] })
          .scatter([ctx.data.flat], { color: HEX.white, r: 5 }).scatter([ctx.data.sharp], { color: HEX.white, r: 3.5 })
          .text(...ctx.data.flat, '  flat', { color: HEX.white }).text(...ctx.data.sharp, '  sharp', { color: HEX.white }).draw();
        const f = ctx.data.frac[key];
        ctx.stat(`${m === 'gd' ? 'GD' : m === 'sam' ? 'one-step SAM' : 'exact SAM objective'}${m === 'gd' ? '' : `, ρ = ${ctx.vals.rho}`}: flat ${(100 * f[0]).toFixed(0)}% · sharp ${(100 * f[1]).toFixed(0)}% · neither ${(100 * f[2]).toFixed(0)}% (GD: sharp ${(100 * ctx.data.frac['gd-0'][1]).toFixed(0)}%)`);
      },
    },
  ],
};
