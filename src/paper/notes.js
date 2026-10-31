// In-page study notes: papers/<key>/notes.md -> HTML article (replaces the notes PDF viewer when present).
//
// Format (one file per paper):
//   # Title                 notes title
//   ## Heading              section (gets an anchor and a contents entry)        ### Heading   sub-heading
//   $$ ... $$               display math (may span lines)                         $..$          inline math
//   **bold**  *italic*  `code`  [text](url)
//   - item / 1. item        lists; indent continuation lines (incl. $$ blocks) by 3 spaces
//   | a | b |               tables (first row = header, second row = --- separator)
//   :::kind Title  ...  ::: callout; kinds: minute, key, theorem, warn, ours, plain
//   :::cols / :::col ... ::: side-by-side columns (stack on phones)
//   :::figure Caption ... ::: raw HTML/SVG figure with a caption
//   :::divider Label  :::   labelled rule between parts of the article
//   > text                  source / footnote line
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const slug = s => s.toLowerCase().replace(/<[^>]+>/g, '').replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');

function tex(t, display) {
  try { return window.katex.renderToString(t, { displayMode: display, throwOnError: false, strict: false }); }
  catch { return esc(t); }
}

export function inline(s) {
  const math = [];
  s = s.replace(/\$([^$\n]+?)\$/g, (_, t) => `\u0000${math.push(t) - 1}\u0000`);
  s = esc(s)
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*/g, '$1<i>$2</i>')
    .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => tex(math[+i], false));
}

// Collect the lines of a ::: block, honouring nested ::: blocks.
function takeFence(lines, i) {
  const body = []; let depth = 1;
  for (i++; i < lines.length; i++) {
    const l = lines[i].trim();
    if (/^:::\S/.test(l)) depth++;
    else if (l === ':::' && --depth === 0) return [body, i + 1];
    body.push(lines[i]);
  }
  return [body, i];
}

function table(rows) {
  const cells = r => r.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
  const head = cells(rows[0]), aligns = cells(rows[1]).map(c => c.endsWith(':') ? (c.startsWith(':') ? 'center' : 'right') : 'left');
  const label = j => esc(head[j].replace(/\$|\*\*/g, ''));
  const td = (tag, c, j) => `<${tag} style="text-align:${aligns[j] || 'left'}"${tag === 'td' ? ` data-label="${label(j)}"` : ''}>${inline(c)}</${tag}>`;
  return `<div class="nt-table"><table><thead><tr>${head.map((c, j) => td('th', c, j)).join('')}</tr></thead><tbody>${
    rows.slice(2).map(r => `<tr>${cells(r).map((c, j) => td('td', c, j)).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function list(lines, ordered) {
  const items = []; const marker = ordered ? /^\d+\.\s+/ : /^[-*]\s+/;
  for (const l of lines) {
    if (marker.test(l)) items.push([l.replace(marker, '')]);
    else items[items.length - 1].push(l.replace(/^ {1,3}/, ''));
  }
  const tag = ordered ? 'ol' : 'ul';
  return `<${tag}>${items.map(it => `<li>${blocks(it)}</li>`).join('')}</${tag}>`;
}

export function blocks(lines, toc) {
  const out = []; let i = 0;
  while (i < lines.length) {
    const raw = lines[i], l = raw.trim();
    if (!l) { i++; continue; }
    let m;
    if ((m = l.match(/^(#{1,3})\s+(.*)$/))) {
      const h = inline(m[2]), id = 'n-' + slug(m[2]);
      if (m[1] === '#') out.push(`<h2 class="nt-title">${h}</h2>`);
      else if (m[1] === '##') { toc?.push([id, m[2]]); out.push(`<h3 class="nt-h" id="${id}">${h}</h3>`); }
      else out.push(`<h4 class="nt-h4">${h}</h4>`);
      i++; continue;
    }
    if (l.startsWith('$$')) {
      let t = l.slice(2);
      if (t.endsWith('$$') && t.length >= 2) { out.push(`<div class="nt-eq">${tex(t.slice(0, -2), true)}</div>`); i++; continue; }
      const buf = [t];
      for (i++; i < lines.length && !lines[i].trim().endsWith('$$'); i++) buf.push(lines[i]);
      buf.push((lines[i] || '').trim().slice(0, -2)); i++;
      out.push(`<div class="nt-eq">${tex(buf.join('\n'), true)}</div>`); continue;
    }
    if ((m = l.match(/^:::(\w+)\s*(.*)$/))) {
      const [body, next] = takeFence(lines, i); i = next;
      const [kind, title] = [m[1], m[2]];
      if (kind === 'cols') out.push(`<div class="nt-cols">${blocks(body)}</div>`);
      else if (kind === 'col') out.push(`<div class="nt-col">${blocks(body)}</div>`);
      else if (kind === 'divider') out.push(`<div class="nt-div"><span>${inline(title)}</span></div>`);
      else if (kind === 'figure') out.push(`<figure class="nt-fig">${body.join('\n')}${title ? `<figcaption>${inline(title)}</figcaption>` : ''}</figure>`);
      else out.push(`<div class="nt-box nt-${kind}">${title ? `<div class="nt-box-h">${inline(title)}</div>` : ''}<div class="nt-box-b">${blocks(body)}</div></div>`);
      continue;
    }
    if (l.startsWith('|')) {
      const rows = []; for (; i < lines.length && lines[i].trim().startsWith('|'); i++) rows.push(lines[i]);
      out.push(table(rows)); continue;
    }
    const ord = /^\d+\.\s+/.test(l), un = /^[-*]\s+/.test(l);
    if (ord || un) {
      const buf = [];
      for (; i < lines.length; i++) {
        const x = lines[i];
        if (!x.trim()) { if (i + 1 < lines.length && /^ {3}|^\d+\.\s|^[-*]\s/.test(lines[i + 1])) { buf.push(''); continue; } break; }
        if (buf.length && !/^ {3}/.test(x) && !(ord ? /^\d+\.\s+/ : /^[-*]\s+/).test(x.trim())) break;
        buf.push(x);
      }
      out.push(list(buf, ord)); continue;
    }
    if (l.startsWith('>')) {
      const buf = []; for (; i < lines.length && lines[i].trim().startsWith('>'); i++) buf.push(lines[i].trim().replace(/^>\s?/, ''));
      out.push(`<p class="nt-src">${inline(buf.join(' '))}</p>`); continue;
    }
    const buf = [];
    for (; i < lines.length; i++) {
      const x = lines[i].trim();
      if (!x || /^(#{1,3}\s|\$\$|:::|\||>|[-*]\s|\d+\.\s)/.test(x)) break;
      buf.push(x);
    }
    out.push(`<p>${inline(buf.join(' '))}</p>`);
  }
  return out.join('\n');
}

export async function renderNotesArticle(text) {
  for (let i = 0; i < 100 && !window.katex; i++) await new Promise(r => setTimeout(r, 50));
  const toc = [], html = blocks(text.split(/\r?\n/), toc);
  const nav = toc.length > 3 ? `<nav class="nt-toc" aria-label="Notes contents">${toc.map(([id, t]) => `<a href="#${id}">${inline(t)}</a>`).join('')}</nav>` : '';
  return `${nav}<article class="nt">${html}</article>`;
}
