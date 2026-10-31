// 2-D helpers for crash-course visuals on top of charts.js Plot: contour lines (marching squares), a height heat map,
// arrows in data units, and pointer dragging in data coordinates.
import { rampCss } from '../lib/charts.js';

/** Contour segments of f at the given levels over dom = [x0,x1,y0,y1] on an n×n grid. Returns [[level, [[ax,ay],[bx,by]]...]...]. */
export function contours(f, dom, levels, n = 90) {
  const [x0, x1, y0, y1] = dom, X = i => x0 + (x1 - x0) * i / n, Y = j => y0 + (y1 - y0) * j / n;
  const z = []; for (let j = 0; j <= n; j++) { const row = []; for (let i = 0; i <= n; i++) row.push(f(X(i), Y(j))); z.push(row); }
  const lerp = (a, b, va, vb, c) => a + (b - a) * (c - va) / (vb - va);
  return levels.map(c => {
    const segs = [];
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
      const v = [z[j][i], z[j][i + 1], z[j + 1][i + 1], z[j + 1][i]], p = [[X(i), Y(j)], [X(i + 1), Y(j)], [X(i + 1), Y(j + 1)], [X(i), Y(j + 1)]];
      const cross = [];
      for (let k = 0; k < 4; k++) {
        const a = v[k], b = v[(k + 1) % 4];
        if ((a < c) !== (b < c)) { const pa = p[k], pb = p[(k + 1) % 4]; cross.push([lerp(pa[0], pb[0], a, b, c), lerp(pa[1], pb[1], a, b, c)]); }
      }
      if (cross.length === 2) segs.push(cross);
      else if (cross.length === 4) { segs.push([cross[0], cross[1]], [cross[2], cross[3]]); }
    }
    return [c, segs];
  });
}

/** Draw contour segments as one custom layer. */
export function drawContours(P, sets, { color = 'rgba(232,236,248,.28)', width = 1, hi = null } = {}) {
  P.custom((g, p) => {
    for (const [c, segs] of sets) {
      g.strokeStyle = hi && hi.level === c ? hi.color : color; g.lineWidth = hi && hi.level === c ? (hi.width || 2) : width;
      g.beginPath();
      for (const [a, b] of segs) { g.moveTo(p.sx(a[0]), p.sy(a[1])); g.lineTo(p.sx(b[0]), p.sy(b[1])); }
      g.stroke();
    }
  });
}

/** Heat map of f over dom, coloured by the Atlas height ramp (t = (f - lo)/(hi - lo), optionally warped). */
export function heat(P, f, dom, { n = 70, lo = null, hi = null, warp = .6, alpha = .55 } = {}) {
  const [x0, x1, y0, y1] = dom, z = [];
  for (let j = 0; j < n; j++) { const row = []; for (let i = 0; i < n; i++) row.push(f(x0 + (x1 - x0) * (i + .5) / n, y0 + (y1 - y0) * (j + .5) / n)); z.push(row); }
  const flat = z.flat(), a = lo ?? Math.min(...flat), b = hi ?? Math.max(...flat);
  P.heat(z, dom, { cell: v => rampCss(Math.pow(Math.min(1, Math.max(0, (v - a) / (b - a || 1))), warp)), alpha });
}

/** Arrow from (x, y) by (dx, dy) in data units, with a filled head; o: {color, width, head}. */
export function arrow(P, x, y, dx, dy, { color = '#ff7a59', width = 2.4, head = 9 } = {}) {
  P.custom((g, p) => {
    const X = p.sx(x), Y = p.sy(y), X2 = p.sx(x + dx), Y2 = p.sy(y + dy), ang = Math.atan2(Y2 - Y, X2 - X), len = Math.hypot(X2 - X, Y2 - Y);
    if (len < 1) return;
    const hd = Math.min(head, len * .45);
    g.strokeStyle = color; g.fillStyle = color; g.lineWidth = width; g.lineCap = 'round';
    g.beginPath(); g.moveTo(X, Y); g.lineTo(X2 - hd * .6 * Math.cos(ang), Y2 - hd * .6 * Math.sin(ang)); g.stroke();
    g.beginPath(); g.moveTo(X2, Y2); g.lineTo(X2 - hd * Math.cos(ang - .42), Y2 - hd * Math.sin(ang - .42)); g.lineTo(X2 - hd * Math.cos(ang + .42), Y2 - hd * Math.sin(ang + .42)); g.closePath(); g.fill();
  });
}

/** Make a plot draggable: onMove([x, y]) is called with data coordinates while the pointer is down. */
export function draggable(ctx, onMove) {
  const c = ctx.plot.c; let down = false;
  const at = e => onMove(ctx.plot.toData(e));
  ctx.listen(c, 'pointerdown', e => { down = true; c.setPointerCapture(e.pointerId); at(e); });
  ctx.listen(c, 'pointermove', e => { if (down) at(e); });
  ctx.listen(c, 'pointerup', () => { down = false; });
  ctx.listen(c, 'pointercancel', () => { down = false; });
  c.style.cursor = 'crosshair';
}
