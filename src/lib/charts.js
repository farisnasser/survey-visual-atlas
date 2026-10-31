// charts — small canvas plotting layer shared by the paper visuals (loss curves, spectra, bifurcation diagrams,
// heat maps, vector fields). One Plot = one canvas; call draw() after changing layers.

const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MONO = '"JetBrains Mono", ui-monospace, monospace';

function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map(s => s * mag).find(s => span / s <= n + .5) || mag * 10;
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
const tickLabel = v => {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 1e4 || a < 1e-2) return v.toExponential(0).replace('e+', 'e');
  return String(+v.toPrecision(3));
};

export class Plot {
  constructor(canvas, o = {}) {
    this.c = canvas; this.ctx = canvas.getContext('2d');
    this.o = Object.assign({ xlabel: '', ylabel: '', title: '', logx: false, logy: false, xlim: null, ylim: null, pad: [30, 14, 34, 50], legend: true, grid: true, equal: false }, o);
    this.layers = [];
    this.ro = new ResizeObserver(() => this.draw()); this.ro.observe(canvas);
  }
  set(o) { Object.assign(this.o, o); return this; }
  clear() { this.layers = []; return this; }
  /** points: [[x,y],...]; o: {color, width, dash, label, alpha, dots, step} */
  line(points, o = {}) { const L = { type: 'line', points, ...o }; this.layers.push(L); return this; }
  scatter(points, o = {}) { const L = { type: 'scatter', points, ...o }; this.layers.push(L); return this; }
  hline(y, o = {}) { this.layers.push({ type: 'hline', y, ...o }); return this; }
  vline(x, o = {}) { this.layers.push({ type: 'vline', x, ...o }); return this; }
  fn(f, o = {}) { const L = { type: 'fn', f, ...o }; this.layers.push(L); return this; }
  bars(items, o = {}) { const L = { type: 'bars', items, ...o }; this.layers.push(L); return this; }   // items: [{x, y, color, label}]
  hist(values, o = {}) { const L = { type: 'hist', values, ...o }; this.layers.push(L); return this; }
  /** heat map: z[j][i] over [x0,x1]x[y0,y1]; colorFn(t)->[r,g,b] 0..255 or css string per cell via cellColor(z) */
  heat(z, dom, o = {}) { const L = { type: 'heat', z, dom, ...o }; this.layers.push(L); return this; }
  arrows(items, o = {}) { const L = { type: 'arrows', items, ...o }; this.layers.push(L); return this; } // [x,y,dx,dy]
  band(x0, x1, o = {}) { this.layers.push({ type: 'band', x0, x1, ...o }); return this; }
  text(x, y, s, o = {}) { this.layers.push({ type: 'text', x, y, s, ...o }); return this; }
  custom(fn) { this.layers.push({ type: 'custom', fn }); return this; }

  _range() {
    let xs = [], ys = [];
    for (const L of this.layers) {
      if (L.type === 'line' || L.type === 'scatter') for (const p of L.points) { if (isFinite(p[0]) && isFinite(p[1])) { xs.push(p[0]); ys.push(p[1]); } }
      if (L.type === 'bars') L.items.forEach((b, i) => { xs.push(i - .5, i + .5); ys.push(0, b.y); });
      if (L.type === 'heat') { xs.push(L.dom[0], L.dom[1]); ys.push(L.dom[2], L.dom[3]); }
      if (L.type === 'hline' && !L.noRange) ys.push(L.y);
    }
    if (this.o.logy) ys = ys.filter(v => v > 0);
    if (this.o.logx) xs = xs.filter(v => v > 0);
    let [x0, x1] = this.o.xlim || [Math.min(...xs), Math.max(...xs)];
    let [y0, y1] = this.o.ylim || [Math.min(...ys), Math.max(...ys)];
    if (!isFinite(x0)) [x0, x1] = [0, 1];
    if (!isFinite(y0)) [y0, y1] = [0, 1];
    if (x0 === x1) { x0 -= 1; x1 += 1; }
    if (y0 === y1) { y0 -= this.o.logy ? y0 / 2 : 1; y1 += this.o.logy ? y1 : 1; }
    if (!this.o.ylim && !this.o.logy) { const m = (y1 - y0) * .06; y0 -= m; y1 += m; }
    return [x0, x1, y0, y1];
  }
  draw() {
    const c = this.c, dpr = Math.min(2, devicePixelRatio), w = c.clientWidth, h = c.clientHeight;
    if (!w || !h) return;
    if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = this.ctx; g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    const [pt, pr, pb, pl] = this.o.pad, W = w - pl - pr, H = h - pt - pb;
    let [x0, x1, y0, y1] = this._range();
    if (this.o.equal && !this.o.logx && !this.o.logy) {   // same data units per pixel on both axes (contours, gradients, phase planes)
      const kx = (x1 - x0) / W, ky = (y1 - y0) / H;
      if (kx > ky) { const m = (kx * H - (y1 - y0)) / 2; y0 -= m; y1 += m; } else { const m = (ky * W - (x1 - x0)) / 2; x0 -= m; x1 += m; }
    }
    const lx = v => this.o.logx ? Math.log10(v) : v, ly = v => this.o.logy ? Math.log10(v) : v;
    const X0 = lx(x0), X1 = lx(x1), Y0 = ly(y0), Y1 = ly(y1);
    const sx = v => pl + (lx(v) - X0) / (X1 - X0) * W, sy = v => pt + H - (ly(v) - Y0) / (Y1 - Y0) * H;
    this.sx = sx; this.sy = sy; this.box = [pl, pt, W, H]; this.range = [x0, x1, y0, y1];
    const muted = css('--muted') || '#8e99bd', line = css('--line') || '#2a3352', ink = css('--ink') || '#e8ecf8';
    g.font = `11px ${MONO}`;
    // grid + ticks
    const xt = this.o.logx ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(2, Math.floor(W / 80)));
    const yt = this.o.logy ? logTicks(y0, y1) : niceTicks(y0, y1, Math.max(2, Math.floor(H / 45)));
    g.strokeStyle = line; g.lineWidth = 1; g.fillStyle = muted;
    g.textAlign = 'center'; g.textBaseline = 'top';
    for (const v of xt) { const X = sx(v); if (X < pl - 1 || X > pl + W + 1) continue; if (this.o.grid) { g.globalAlpha = .45; g.beginPath(); g.moveTo(X, pt); g.lineTo(X, pt + H); g.stroke(); g.globalAlpha = 1; } g.fillText(this.o.xfmt ? this.o.xfmt(v) : tickLabel(v), X, pt + H + 6); }
    g.textAlign = 'right'; g.textBaseline = 'middle';
    for (const v of yt) { const Y = sy(v); if (Y < pt - 1 || Y > pt + H + 1) continue; if (this.o.grid) { g.globalAlpha = .45; g.beginPath(); g.moveTo(pl, Y); g.lineTo(pl + W, Y); g.stroke(); g.globalAlpha = 1; } g.fillText(this.o.yfmt ? this.o.yfmt(v) : tickLabel(v), pl - 6, Y); }
    g.save(); g.beginPath(); g.rect(pl, pt, W, H); g.clip();
    const legend = [];
    for (const L of this.layers) {
      const col = L.color || ink;
      g.globalAlpha = L.alpha ?? 1; g.setLineDash(L.dash || []);
      if (L.type === 'band') { g.fillStyle = L.color || 'rgba(255,255,255,.05)'; g.fillRect(sx(L.x0), pt, sx(L.x1) - sx(L.x0), H); }
      else if (L.type === 'heat') {
        const z = L.z, ny = z.length, nx = z[0].length, [a0, a1, b0, b1] = L.dom;
        const cw = (sx(a1) - sx(a0)) / nx, ch = (sy(b0) - sy(b1)) / ny;
        for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) { g.fillStyle = L.cell(z[j][i], i, j); g.fillRect(sx(a0) + i * cw - .3, sy(b0) - (j + 1) * ch - .3, cw + .6, ch + .6); }
      } else if (L.type === 'line') {
        g.strokeStyle = col; g.lineWidth = L.width || 1.8; g.beginPath(); let pen = false;
        for (const p of L.points) {
          const ok = isFinite(p[1]) && (!this.o.logy || p[1] > 0) && (!this.o.logx || p[0] > 0);
          if (!ok) { pen = false; continue; }
          const X = sx(p[0]), Y = sy(p[1]);
          if (!pen) { g.moveTo(X, Y); pen = true; } else if (L.step) { g.lineTo(X, g._ly); g.lineTo(X, Y); } else g.lineTo(X, Y);
          g._ly = Y;
        }
        g.stroke();
        if (L.dots) { g.fillStyle = col; for (const p of L.points) if (isFinite(p[1])) { g.beginPath(); g.arc(sx(p[0]), sy(p[1]), L.dots, 0, 7); g.fill(); } }
      } else if (L.type === 'fn') {
        g.strokeStyle = col; g.lineWidth = L.width || 1.6; g.beginPath(); let pen = false;
        for (let i = 0; i <= 240; i++) {
          const xv = this.o.logx ? 10 ** (X0 + (X1 - X0) * i / 240) : x0 + (x1 - x0) * i / 240, yv = L.f(xv);
          if (!isFinite(yv) || (this.o.logy && yv <= 0)) { pen = false; continue; }
          const X = sx(xv), Y = sy(yv); if (!pen) { g.moveTo(X, Y); pen = true; } else g.lineTo(X, Y);
        }
        g.stroke();
      } else if (L.type === 'scatter') {
        g.fillStyle = col; const r = L.r || 1.6;
        for (const p of L.points) { if (!isFinite(p[1])) continue; if (p[2]) g.fillStyle = p[2]; g.beginPath(); g.arc(sx(p[0]), sy(p[1]), r, 0, 7); g.fill(); }
      } else if (L.type === 'hline') {
        g.strokeStyle = col; g.lineWidth = L.width || 1.2; const Y = sy(L.y); g.beginPath(); g.moveTo(pl, Y); g.lineTo(pl + W, Y); g.stroke();
        if (L.text) { g.setLineDash([]); g.fillStyle = col; g.textAlign = 'right'; g.textBaseline = 'bottom'; g.fillText(L.text, pl + W - 4, Y - 3); }
      } else if (L.type === 'vline') {
        g.strokeStyle = col; g.lineWidth = L.width || 1.2; const X = sx(L.x); g.beginPath(); g.moveTo(X, pt); g.lineTo(X, pt + H); g.stroke();
        if (L.text) { g.setLineDash([]); g.fillStyle = col; g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText(L.text, X + 4, pt + 4); }
      } else if (L.type === 'bars') {
        const bw = W / L.items.length * (L.width || .7);
        L.items.forEach((b, i) => {
          const X = sx(i) - bw / 2, Y = sy(Math.max(b.y, this.o.logy ? y0 : 0)), Y2 = sy(this.o.logy ? y0 : 0);
          g.fillStyle = b.color || col; g.fillRect(X + (b.off || 0) * bw, Math.min(Y, sy(b.y)), bw * (b.w || 1), Math.abs(Y2 - sy(b.y)));
        });
      } else if (L.type === 'hist') {
        const bins = L.bins || 40, [a, b] = L.range || [x0, x1], cnt = new Array(bins).fill(0);
        for (const v of L.values) { const k = Math.floor((v - a) / (b - a) * bins); if (k >= 0 && k < bins) cnt[k]++; }
        const mx = L.norm ? Math.max(...cnt) / L.norm : 1;
        g.fillStyle = col;
        cnt.forEach((n, k) => { const xa = a + (b - a) * k / bins, xb = a + (b - a) * (k + 1) / bins, yv = n / mx; g.fillRect(sx(xa) + .5, sy(yv), sx(xb) - sx(xa) - 1, sy(this.o.logy ? y0 : 0) - sy(yv)); });
      } else if (L.type === 'arrows') {
        g.strokeStyle = col; g.fillStyle = col; g.lineWidth = L.width || 1;
        for (const [x, y, dx, dy] of L.items) {
          const X = sx(x), Y = sy(y), X2 = sx(x + dx), Y2 = sy(y + dy), ang = Math.atan2(Y2 - Y, X2 - X), len = Math.hypot(X2 - X, Y2 - Y);
          if (len < .5) continue;
          g.beginPath(); g.moveTo(X, Y); g.lineTo(X2, Y2); g.stroke();
          const hd = Math.min(5, len * .4); g.beginPath(); g.moveTo(X2, Y2); g.lineTo(X2 - hd * Math.cos(ang - .45), Y2 - hd * Math.sin(ang - .45)); g.lineTo(X2 - hd * Math.cos(ang + .45), Y2 - hd * Math.sin(ang + .45)); g.fill();
        }
      } else if (L.type === 'text') {
        g.setLineDash([]); g.fillStyle = col; g.font = L.font || `11px ${MONO}`; g.textAlign = L.align || 'left'; g.textBaseline = L.base || 'middle'; g.fillText(L.s, sx(L.x) + (L.dx || 0), sy(L.y) + (L.dy || 0));
      } else if (L.type === 'custom') { g.setLineDash([]); L.fn(g, this); }
      g.globalAlpha = 1; g.setLineDash([]);
      if (L.label) legend.push([L.label, L.color || ink, L.dash]);
    }
    g.restore();
    g.strokeStyle = line; g.strokeRect(pl + .5, pt + .5, W - 1, H - 1);
    g.fillStyle = muted; g.textAlign = 'center'; g.textBaseline = 'bottom';
    if (this.o.xlabel) g.fillText(this.o.xlabel, pl + W / 2, h - 2);
    if (this.o.ylabel) { g.save(); g.translate(11, pt + H / 2); g.rotate(-Math.PI / 2); g.textBaseline = 'middle'; g.fillText(this.o.ylabel, 0, 0); g.restore(); }
    if (this.o.title) { g.fillStyle = ink; g.textAlign = 'left'; g.textBaseline = 'top'; g.font = `600 12px ${css('--display') || 'sans-serif'}`; g.fillText(this.o.title, pl, 8); g.font = `11px ${MONO}`; }
    if (this.o.legend && legend.length) {
      g.textAlign = 'left'; g.textBaseline = 'middle'; let X = pl + 8, Y = pt + 12;
      const maxW = W - 16;
      { // backing plate so the legend never sits directly on the curves
        let x = pl + 8, rows = 1, wMax = 0;
        for (const [s] of legend) { const tw = g.measureText(s).width + 30; if (x + tw > pl + maxW) { x = pl + 8; rows++; } x += tw; wMax = Math.max(wMax, x - pl - 8); }
        g.fillStyle = 'rgba(11,14,26,.78)'; g.fillRect(pl + 2, pt + 2, Math.min(maxW + 8, wMax + 10), rows * 16 + 6);
      }
      for (const [s, col, dash] of legend) {
        const tw = g.measureText(s).width + 30;
        if (X + tw > pl + maxW) { X = pl + 8; Y += 16; }
        g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dash || []); g.beginPath(); g.moveTo(X, Y); g.lineTo(X + 16, Y); g.stroke(); g.setLineDash([]);
        g.fillStyle = muted; g.fillText(s, X + 21, Y); X += tw;
      }
    }
  }
  /** data coords from a mouse event */
  toData(ev) {
    const r = this.c.getBoundingClientRect(), [pl, pt, W, H] = this.box, [x0, x1, y0, y1] = this.range;
    const fx = (ev.clientX - r.left - pl) / W, fy = 1 - (ev.clientY - r.top - pt) / H;
    const X = this.o.logx ? 10 ** (Math.log10(x0) + fx * (Math.log10(x1) - Math.log10(x0))) : x0 + fx * (x1 - x0);
    const Y = this.o.logy ? 10 ** (Math.log10(y0) + fy * (Math.log10(y1) - Math.log10(y0))) : y0 + fy * (y1 - y0);
    return [X, Y];
  }
}

function logTicks(a, b) {
  const out = [], lo = Math.floor(Math.log10(a)), hi = Math.ceil(Math.log10(b));
  const stride = Math.max(1, Math.ceil((hi - lo) / 6));
  for (let k = lo; k <= hi; k += stride) out.push(10 ** k);
  return out;
}

/** surface-style colour ramp for heat maps (indigo -> teal -> pale yellow) */
export function rampCss(t) {
  const S = [[0x1b, 0x1f, 0x4b], [0x24, 0x4f, 0x86], [0x2a, 0x9c, 0x96], [0x96, 0xd0, 0x78], [0xf4, 0xe8, 0xa4]];
  t = Math.min(1, Math.max(0, t)) * (S.length - 1);
  const i = Math.min(S.length - 2, Math.floor(t)), u = t - i, a = S[i], b = S[i + 1];
  return `rgb(${a.map((v, k) => Math.round(v + (b[k] - v) * u)).join(',')})`;
}
